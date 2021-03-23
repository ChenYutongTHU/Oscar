# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import argparse
import base64
import numpy as np
import os
import os.path as op
import random, time, json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
sys.path.append('/data/private/chenyutong/Oscar')
from oscar.utils.logger import setup_logger
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import (tsv_writer, concat_tsv_files,
        delete_tsv_files, reorder_tsv_keys)
from oscar.utils.misc import (mkdir, set_seed, 
        load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption,
        ScstRewardCriterion)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader
from oscar.utils.cbs import FiniteStateMachineBuilder
from oscar.modeling.modeling_bert import BertForImageCaptioning
from oscar.modeling.modeling_VIVO import BertForVIVOPretraining
from oscar_transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar_transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from tensorboardX import SummaryWriter



class CaptionTSVDataset(Dataset):
    def __init__(self, yaml_file, tokenizer=None, add_od_labels=True,
            max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40, 
            is_train=True, mask_prob=0.15, max_masked_tokens=3, whole_word=True, **kwargs):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = op.dirname(yaml_file)
        self.label_file = find_file_path_in_yaml(self.cfg['label'], self.root)
        self.feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)

        assert op.isfile(self.feat_file)
        if add_od_labels: assert op.isfile(self.label_file)

        self.label_tsv = None if not self.label_file else TSVFile(self.label_file)
        self.feat_tsv = TSVFile(self.feat_file)

        self.tokenizer = tokenizer
        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                max_seq_length, mask_prob, max_masked_tokens, whole_word=whole_word,
                is_train=is_train)
        self.add_od_labels = add_od_labels
        self.is_train = is_train
        self.kwargs = kwargs
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.seek(i)[0] : i for i in range(tsv.num_rows())}

    def get_image_index(self, idx):
        return idx

    def get_image_key(self, idx):
        img_idx = self.get_image_index(idx)
        return self.image_keys[img_idx]

    def get_image_features(self, img_idx):
        num_boxes = int(self.feat_tsv.seek(img_idx)[1])
        features = np.frombuffer(base64.b64decode(self.feat_tsv.seek(img_idx)[2]), np.float32
                ).reshape((num_boxes, -1))[:]
        return torch.Tensor(features)

    def get_od_labels(self, img_idx):
        od_labels = None
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            #od_labels = " ".join([l['class'] for l in label_info])
            od_labels = [l['class'] for l in label_info]
        return od_labels

    def __getitem__(self, idx):
        img_idx = self.get_image_index(idx)
        img_key = self.image_keys[img_idx]
        features = self.get_image_features(idx)
        od_labels = self.get_od_labels(idx)
        example = self.tensorizer.tensorize_example(od_labels, features)
        return img_key, example

    def __len__(self):
        return self.get_valid_tsv().num_rows()


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            mask_prob=0.15, max_masked_tokens=3, whole_word=True, 
            is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_tags = self.max_seq_len-2
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self.whole_word = whole_word
        # torch.tril returns the lower triangular part, other elements are set to 0 (upper right set to 0)
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))


    def tokenize_tags(self, tags, max_seq_len):
        assert type(tags) == list
        tokens = []
        segments = []
        pt = 1 #take cls into accountß
        tags_valid = []
        for t in tags:
            tokens_ = self.tokenizer.tokenize(t)
            if len(tokens)+len(tokens_)>max_seq_len:
                break
            segments.append([pt, pt+len(tokens_)])
            pt = pt+len(tokens_)
            tokens.extend(tokens_)
            tags_valid.append(t)
        return tokens, segments, tags_valid

    def tensorize_example(self, text_a, img_feat,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        if self.whole_word:
            tokens_a, tokens_a_segment, tags = self.tokenize_tags(text_a, self.max_seq_len-2)
            num_tags = len(tags)
            #padding tags #num_tags <= len(tokens_a) <=self.max_seq_len==self.max_tags
            assert num_tags<=self.max_tags, tags
            #print(num_tags)
            #tags = tags + (self.max_tags-num_tags)*['PAD_tag']
            tags = ', '.join(tags)
        else:
            text_a = ' '.join(text_a)
            tokens_a = self.tokenizer.tokenize(text_a)
            tags = None

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_len = len(tokens)

        # pad text_a to keep it in fixed length for better inference.
        #padding_a_len = self.max_seq_a_len - seq_a_len
        #tokens += [self.tokenizer.pad_token] * padding_a_len
        #segment_ids += ([pad_token_segment_id] * padding_a_len)
        # ---- No padding for tag (by Yutong) cuz we don't need to generate caption word one by one

        masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
        if self.whole_word:
            candidate_masked_tags = list(range(0, num_tags)) # only mask text_a
            if self.is_train:
                random.shuffle(candidate_masked_tags)
            num_masked = min(max(round(self.mask_prob * num_tags), 1), self.max_masked_tokens) #--> tag
            num_masked = int(num_masked)   #when evaluating masking the first few tags
            assert num_masked<=num_tags
            masked_tag_idx = candidate_masked_tags[:num_masked]
            masked_tag_idx = sorted(masked_tag_idx)
            masked_token, masked_idx, masked_whole_word = [], [], []
            for i in masked_tag_idx:
                st, ed = tokens_a_segment[i]
                masked_token.extend(tokens[st:ed])
                st_,ed_ = len(masked_idx), len(masked_idx)+ed-st  
                masked_idx.extend(list(range(st,ed))) 
                masked_whole_word.append([st_,ed_]) 
            #padding masked_whole_word
            masked_whole_word = masked_whole_word + [[-1,-1]]*(self.max_masked_tokens - num_masked)
            for pos in masked_idx:
                if random.random() <= 0.8 or not self.is_train:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

        else:
            # randomly mask words for prediction, ignore [CLS]
            masked_whole_word = None
            candidate_masked_idx = list(range(1, seq_len)) # only mask text_a
            if self.is_train:
                random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8 or not self.is_train:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

        masked_pos[masked_idx] = 1 
        # pad masked tokens to the same length
        if not self.whole_word and num_masked < self.max_masked_tokens:
            masked_token = masked_token + ([self.tokenizer.pad_token] *
                    (self.max_masked_tokens - num_masked))
        elif self.whole_word: #padding
            assert len(masked_token) <= self.max_seq_len-2
            masked_token = masked_token + [self.tokenizer.pad_token]*(self.max_seq_len-2-len(masked_token))
        masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # bidirectional attention mask
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # T: tags R: image region
        t_start, t_end = 0, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # full attention for T-T R-R T-R
        attention_mask[t_start : t_end, t_start : t_end] = 1
        attention_mask[r_start : r_end, r_start : r_end] = 1
        attention_mask[t_start : t_end, r_start : r_end] = 1
        attention_mask[r_start : r_end, t_start : t_end] = 1
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        masked_whole_word = torch.tensor(masked_whole_word, dtype=torch.long)
        num_masked = torch.tensor(num_masked, dtype=torch.long)
        masked_ids = torch.tensor(masked_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids, masked_whole_word, num_masked,tags)
        #return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids, masked_whole_word, num_masked,tags)


def build_dataset(yaml_file, tokenizer, args, is_train=True):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    if is_train:
        return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,
            is_train=True, mask_prob=args.mask_prob, max_masked_tokens=args.max_masked_tokens,
            whole_word=args.whole_word)
    
    return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,
            is_train=False, mask_prob=args.mask_prob, max_masked_tokens=args.max_masked_tokens,
            whole_word=args.whole_word)


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, yaml_file, tokenizer, is_distributed=True, 
        is_train=True):
    dataset = build_dataset(yaml_file, tokenizer, args, 
        is_train=(is_train and not args.scst))
    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
    )
    return data_loader


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    scores = logits == labels 
    return scores


def train(args, train_dataloader, val_dataset, model, tokenizer, writer):
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.per_gpu_train_batch_size * get_world_size() * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if args.scst:
        scst_criterion = ScstRewardCriterion(
            cider_cached_tokens=op.join(args.data_dir, args.cider_cached_tokens),
            baseline_type=args.sc_baseline_type,
        )
        logger.info("  SCST training...")


    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, (img_keys, batch) in enumerate(train_dataloader):
            tags = batch[-1]
            batch = tuple(t.to(args.device) for t in batch[:-1])

            if not args.scst:
                model.train()
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3], 
                    'masked_pos': batch[4], 'masked_ids': batch[5],
                    'whole_word_masked': batch[6],
                    'num_masked_tags': batch[7]
                    #'tags': tags
                }

               #  debug
                torch.set_printoptions(profile="full")
                #print('input_ids shape ', batch[0].shape)
                # print('tag ', tags[0])
                # print('num_masked_tags', inputs['num_masked_tags'][0])
                # print('whole_word_masked ', inputs['whole_word_masked'][0])
                # print('input_ids\n',inputs['input_ids'][0])
                # # #print('attention_mask\n',inputs['attention_mask'][0])
                # # #print('token_type_ids\n',inputs['token_type_ids'][0])
                # print('img_feats\n',inputs['img_feats'].shape)
                # print('masked_pos',inputs['masked_pos'][0])
                # print('masked_ids',inputs['masked_ids'][0])
                # 
                #continue
                # torch.set_printoptions(profile="default")
                
                outputs = model(**inputs)
                loss, logits, ranked_masked_ids = outputs[:3]
                # masked_ids = inputs['masked_ids']
                # masked_ids = masked_ids[masked_ids != 0]
                batch_score = compute_score_with_logits(logits, ranked_masked_ids)
                batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])
            else:
                loss = scst_train_iter(args, train_dataloader, model, scst_criterion, img_keys, batch, tokenizer)
                batch_acc = scst_criterion.get_score()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, 
                        batch_acc, global_acc / global_step)
                    )
                    writer.add_scalar('loss', loss, global_step=global_step)
                    writer.add_scalar('batch_acc', batch_acc, global_step=global_step)

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        evaluate(args, val_dataset, model, tokenizer,
                                checkpoint_dir, writer, global_step)
                        # with open(evaluate_file, 'r') as f:
                        #     res = json.load(f)
                        # best_score = max(best_score, res['CIDEr'])
                        # res['epoch'] = epoch
                        # res['global_step'] = step
                        # res['best_CIDEr'] = best_score
                        # eval_log.append(res)
                        # with open(args.output_dir + '/eval_logs.json', 'w') as f:
                        #     json.dump(eval_log, f)
    return checkpoint_dir


def scst_train_iter(args, train_dataloader, model, scst_criterion, 
        img_keys, batch, tokenizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, 
        tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token]
    )
    inputs = {'is_decode': True,
        'input_ids': batch[0], 'attention_mask': batch[1],
        'token_type_ids': batch[2], 'img_feats': batch[3],
        'masked_pos': batch[4],
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.sc_beam_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": 1,
        "num_keep_best": 1,
    }

    def _ids_to_captions(all_ids):
        captions = []
        for ids in all_ids:
            c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            captions.append(c)
        return captions

    if args.sc_baseline_type == 'greedy':
        model.eval()
        with torch.no_grad():
            greedy_res_raw, _ = model(**inputs)
            greedy_res_raw.squeeze_(1)  # batch_size * max_len
        greedy_res = _ids_to_captions(greedy_res_raw)
    else:
        greedy_res = None

    model.train()
    inputs['do_sample'] = True
    inputs['num_return_sequences'] = args.sc_train_sample_n
    sample_res_raw, sample_logprobs = model(**inputs)
    sample_res_raw.squeeze_(1)
    sample_logprobs.squeeze_(1)
    assert sample_logprobs.requires_grad == True
    assert sample_res_raw.requires_grad == False
    sample_res = _ids_to_captions(sample_res_raw)

    gt_res = [train_dataloader.dataset.get_captions_by_key(k) for k in img_keys]
    loss = scst_criterion(gt_res, greedy_res, sample_res, sample_logprobs)
    return loss


def get_predict_file(output_dir, yaml_file, args):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
    data = op.basename(op.join(args.data_dir, '')[:-1])
    split = op.basename(yaml_file)
    assert split.endswith('.yaml')
    split = split[:-5]
    cc.append(data)
    cc.append(split)
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.add_od_labels:
        cc.append('odlabels')
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.use_cbs:
        cc.append('cbs{}'.format(args.min_constraints_to_satisfy))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    fpath = op.splitext(predict_file)[0]
    return fpath + '.eval.json'


def get_evaluate_method(predict_file):
    if 'nocaps' in op.basename(predict_file):
        return 'nocaps'
    else:
        return 'coco'


def evaluate(args, val_dataloader, model, tokenizer, output_dir, writer, global_step):
    # predict_file = get_predict_file(output_dir,
    #         val_dataloader.dataset.yaml_file, args)
    pred_results, labels, average_accuracy = test(args, val_dataloader, model, tokenizer, writer, global_step)
    with open(os.path.join(output_dir,'prediction_results.txt'),'w') as f:
        N = labels.shape[0]
        lbs = labels.detach().cpu().numpy()
        pds = pred_results.detach().cpu().numpy()
        for i in range(N):
            t1 = tokenizer.ids_to_tokens[lbs[i]]
            t2 = tokenizer.ids_to_tokens[pds[i]]
            f.writelines('{}\t{}\n'.format(t1, t2))

    return None


def test(args, test_dataloader, model, tokenizer, writer, global_step):
    if args.local_rank == 0:
        model.eval()
        pred_results = []
        labels = []
        accs,losses = [],[]

        #print(len(test_dataloader))
        for step, (img_keys, batch) in tqdm(enumerate(test_dataloader)):
            batch = tuple(t.to(args.device) for t in batch[:-1])
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                'token_type_ids': batch[2], 'img_feats': batch[3], 
                'masked_pos': batch[4], 'masked_ids': batch[5],
                'whole_word_masked': batch[6],
                'num_masked_tags': batch[7]
            }       
            outputs = model(**inputs)#only one token
            loss, logits, ranked_masked_ids = outputs[:3] #logits
            # masked_ids = inputs['masked_ids']
            # masked_ids = masked_ids[masked_ids != 0]
            batch_score = compute_score_with_logits(logits, ranked_masked_ids)
            batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])
            accs.append(batch_acc)
            pred_results.append(torch.max(logits, -1)[1].data) 
            labels.append(ranked_masked_ids)
            losses.append(loss)

        pred_results = torch.cat(pred_results, dim=0)
        labels = torch.cat(labels, dim=0)
        average_accuracy = torch.mean(torch.stack(accs, dim=0))
        average_loss = torch.mean(torch.stack(losses, dim=0))

        writer.add_scalar('validation_accuracy', average_accuracy, global_step)
        writer.add_scalar('validation_loss', average_loss, global_step)
    if get_world_size() > 1:
        torch.distributed.barrier()
    return pred_results, labels, average_accuracy


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                max_seq_length, args.max_gen_length, max_od_labels_len))


    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
            'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/data/private/NocapsData/VIVO_pretrain_data', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False, 
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False, 
                        help="yaml file used for validation during training.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    # parser.add_argument("--loss_type", default='sfmx', type=str, 
    #                     help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=15, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "default 15")
    parser.add_argument("--max_seq_a_length", default=15, type=int, 
                        help="The maximum sequence length for tag. should be the same with max_seq_length")
    parser.add_argument('--whole_word', action='store_true', help='mask whole word when masking tags')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--tie_weights", default=False, action='store_true', 
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=False, action='store_true', 
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float, 
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float, 
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int, 
                        help=".")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=40, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=5,
                        help="number of sampled captions for sc training")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--sc_beam_size', type=int, default=1,
                        help="beam size for scst training")
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p',
                        help="path to cached cPickle file used to calculate CIDEr scores")
    # for generation
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=20,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vivo_petrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args)

    if not args.distributed or (args.local_rank==0):
        writer = SummaryWriter(logdir=output_dir)
    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForVIVOPretraining, BertTokenizer
    if args.do_train:
        assert args.model_name_or_path is not None
        config = config_class.from_pretrained(args.config_name if args.config_name else \
                args.model_name_or_path, num_labels=args.num_labels, finetuning_task='VIVO_pretraining') #?
        config.whole_word = args.whole_word
        config.task = 'VIVO'
        if args.scst:
            # avoid using too much memory
            config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        #config.loss_type = args.loss_type
        config.tie_weights = args.tie_weights
        config.freeze_embedding = args.freeze_embedding
        #config.label_smoothing = args.label_smoothing # useless
        #config.drop_worst_ratio = args.drop_worst_ratio #useless
        #config.drop_worst_after = args.drop_worst_after #useless
        model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataloader = make_data_loader(args, args.train_yaml, tokenizer,
            args.distributed, is_train=True)
        val_dataloader = None
        if args.evaluate_during_training:
            val_dataloader = make_data_loader(args, args.val_yaml, tokenizer,
                args.distributed, is_train=False)
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer, writer)

        # test the last checkpoint after training
        if args.do_test:
            logger.info("Evaluate on dataset: " + args.test_yaml)
            test_dataloader = make_data_loader(args, args.test_yaml, 
                tokenizer, args.distributed, is_train=False)
            evaluate(args, test_dataloader, model, tokenizer, last_checkpoint)

    # inference and evaluation
    elif args.do_test or args.do_eval:
        logger.info("Evaluate on dataset: " + args.test_yaml)
        test_dataloader = make_data_loader(args, args.test_yaml,
            tokenizer, args.distributed, is_train=False)

        if not args.do_eval:
            predict_file = get_predict_file(checkpoint, test_dataloader.dataset.yaml_file, args)
            test(args, test_dataloader, model, tokenizer, predict_file)
            logger.info("Prediction results saved to: {}".format(predict_file))
        else:
            evaluate_file = evaluate(args, test_dataloader, model, tokenizer,
                    checkpoint)
            logger.info("Evaluation results saved to: {}".format(evaluate_file))

if __name__ == "__main__":
    main()
