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
        load_from_yaml_file, find_file_path_in_yaml,
        draw_position_embeddings)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption,
        ScstRewardCriterion,convert_tsv_to_coco_format)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader, load_wordforms
from oscar.utils.cbs import FiniteStateMachineBuilder
from oscar.modeling.modeling_bert import BertForImageCaptioning
from oscar_transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar_transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from tensorboardX import SummaryWriter
from oscar.utils.progressbar import ProgressBar
from kara_storage.abc import Serializer
import kara_storage
import struct

class FeatureSerializer(Serializer):
    def serialize(self, y):
        ind, x = y
        num_box = x.shape[0]
        #print(ind, num_box)
        #print(struct.pack("I", ind)+struct.pack("I", num_box))
        return struct.pack("I", ind) + struct.pack("I", num_box) + x.tobytes()
    
    def deserialize(self, x):
        import numpy as np
        #print(x[:8])
        ind = struct.unpack("I", x[:4])[0]
        num_box = struct.unpack("I", x[4:8])[0]
        #print(ind, num_box)
        return ind, np.frombuffer(x[8:], np.float32).reshape((num_box, -1))[:]

class FeatureSerializer_old(Serializer):
    #for version 2.0.4
    def serialize(self, x):
        num_box = x.shape[0]
        return struct.pack("I", num_box) + x.tobytes()
    
    def deserialize(self, x):
        import numpy as np
        num_box = struct.unpack("I", x[:4])[0]
        return np.frombuffer(x[4:], np.float32).reshape((num_box, -1))[:]

def read_wrapper(x):
    old_read = x.read
    def nw_read():
        v = old_read()
        if v is None:
            return None
        return x.tell()-1, v
    x.read = nw_read

class CaptionKaraDataset(torch.utils.data.IterableDataset):
    def __init__(self, yaml_file, tokenizer=None, add_od_labels=True,
            max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40, 
            is_train=True, mask_prob=0.15, max_masked_tokens=3, scst=False, is_distributed=False,
            match_threshold=1,
            **kwargs):
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.storage = kara_storage.KaraStorage(self.cfg['storage_path']) 

        self.storage_open = self.storage.open_dataset(self.cfg['split0'], self.cfg['split1'], 
            "r", serialization=FeatureSerializer())
        self.iterable_ImgDataset = kara_storage.make_torch_dataset(self.storage_open, 
            shuffle=is_train, auto_distributed=is_distributed and is_train) #shuffle!
 
        #print(len(self.iterable_ImgDataset))
        with open(self.cfg['label_path'],'r') as f:
            self.labels = json.load(f)  #[id,[labels]], [id, [labels],[id,[labels]]]
        # print('labels',len(self.labels))
        # input()
        if 'caption_path' in self.cfg:
            with open(self.cfg['caption_path'],'r') as f:
                self.captions = json.load(f)
            self.key2captions = {}
            for image_id, cap in self.captions:
                if not image_id in self.key2captions:
                    self.key2captions[image_id] = []
                self.key2captions[image_id].append(cap)

        else:
            self.captions = None
            self.key2captions = None

        if 'match_path' in self.cfg:
            with open(self.cfg['match_path'],'r') as f:
                self.match = json.load(f)
                self.match_threshold = match_threshold
        else:
            self.match = None
            self.match_threshold = 1

        self.tokenizer = tokenizer

        self.is_train = is_train 
        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                max_seq_length, max_seq_a_length, mask_prob, max_masked_tokens, 
                is_train=(self.is_train and not scst))

        self.scst = scst
        self.kwargs = kwargs

    def __iter__(self):
        for img in self.iterable_ImgDataset:
            index, features = img[0], img[1]
            assert index < len(self.labels), (index, len(self.labels), len(self.iterable_ImgDataset))
            img_id, od_labels = self.labels[index]
            if self.captions:
                _, caps = self.captions[index]
            else:
                caps = None
            example = self.tensorizer.tensorize_example(text_a=caps, text_b=od_labels, img_feat=torch.Tensor(features),
                sequence_a_segment_id=self.kwargs['cap_segment_id'], 
                sequence_b_segment_id=self.kwargs['tag_segment_id'],
                match=self.match[img_id],
                match_threshold=self.match_threshold)  #tag -> 0 caption -> 1
            yield img_id, example

    def set_epoch(self, epoch):
        self.iterable_ImgDataset.set_epoch(epoch)

    def get_caption_file_in_coco_format(self):
        assert self.is_train==False
        cap_file = op.splitext(self.cfg['caption_path'])[0] + '_coco_format.json' #root
        return cap_file

    def get_captions_by_key(self, key):
        return self.key2captions[key]

    def __len__(self):
        if self.is_train:
            n_gpu = get_world_size()
            return len(self.labels)//n_gpu
        else:
            return len(self.labels)

class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
        max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        #print('max_seq_a_len', self.max_seq_a_len)
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self.is_train = is_train
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

    def tokenize_tags(self, tags, max_seq_len):
        assert type(tags) == list
        tokens = []
        segments = []
        #pt = 1 #take cls into accountß
        pt=0
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


    def tensorize_example(self, text_a, text_b, img_feat,  #a caption; b tag
            cls_token_segment_id=0, pad_token_segment_id=0, sequence_a_segment_id=1, sequence_b_segment_id=0,
            match=None, match_threshold=1):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        padding_a_len = self.max_seq_a_len - seq_a_len
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)

        #text_b = ' '.join(text_b)
        #tokens_b = self.tokenizer.tokenize(text_b)

        self.max_tag_len = self.max_seq_len - len(tokens) - 1
        tokens_b, segments, tags_valid = self.tokenize_tags(text_b, self.max_seq_len - len(tokens) - 1)
        #segments len(segments)<=self.max_seq_len - len(tokens) - 1
        assert len(tokens_b) <= self.max_seq_len - len(tokens) - 1
        if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
            tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]

        padded_tokens_b = tokens_b[:]
        self.tag_len = len(tokens_b)
        if len(padded_tokens_b) < self.max_seq_len - len(tokens) - 1:
            padded_tokens_b = padded_tokens_b + [self.tokenizer.pad_token]*(self.max_seq_len - len(tokens) - 1 - len(padded_tokens_b))

        #match [[tag,tag_id->segment,region_id]]
        assert match!=None
        match_ids = []
        offset = 0#len(tokens) we plus offset in model forwarding (adapting to len(input_lens))
        #print('offset', len(tokens), self.max_seq_a_len)
        for ii, (seg, m, t) in enumerate(zip(segments, match[:len(segments)], tags_valid)): 
            if not m[0]==t:
                for jj, (m_,t_,tb) in enumerate(zip(match, tags_valid, text_b)):
                    print(jj, m_[0], t_,tb)
            assert m[0]==t, (m[0],t,i)

            if not m[-1]>match_threshold or m[2]>=self.max_img_seq_len:
                continue
            s, t = seg
            #assert s==len(match_ids), segments 
            for i in range(s,t):
                match_ids.append([i+offset, m[2]]) # m [tag, tag_id, region_id, region_iou]
        #padding
        self.match_len = len(match_ids)
        if len(match_ids) < self.max_tag_len:
            match_ids = match_ids + \
                [[0,0]]*(self.max_tag_len- len(match_ids))  #-1,-1??


        tokens += tokens_b + [self.tokenizer.sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        seq_len = len(tokens)

        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len)) # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
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
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

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

        if match_threshold<1:
            max_len = self.max_seq_len + self.max_img_seq_len + self.max_tag_len
        else:
            max_len = self.max_seq_len + self.max_img_seq_len
            self.match_len = 0
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region, M:match
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        m_start, m_end = self.max_seq_len + self.max_img_seq_len, self.max_seq_len + self.max_img_seq_len + self.match_len
        # triangle mask for caption to caption
        attention_mask[c_start : c_end, c_start : c_end].copy_(self._triangle_mask[0 : seq_a_len, 0 : seq_a_len])
        # full attention for C-L, C-R, C-M
        attention_mask[c_start : c_end, l_start : l_end] = 1
        attention_mask[c_start : c_end, r_start : r_end] = 1
        attention_mask[c_start : c_end, m_start : m_end] = 1
        # full attention for L-L, L-R, L-M
        attention_mask[l_start : l_end, l_start : l_end] = 1
        attention_mask[l_start : l_end, r_start : r_end] = 1
        attention_mask[l_start : l_end, m_start : m_end] = 1
        # full attention for R-L, R-R, R-M
        attention_mask[r_start : r_end, l_start : l_end] = 1
        attention_mask[r_start : r_end, r_start : r_end] = 1
        attention_mask[r_start : r_end, m_start : m_end] = 1        
        # full attention for M-L, M-R, M-M
        attention_mask[m_start : m_end, l_start : l_end] = 1
        attention_mask[m_start : m_end, r_start : r_end] = 1
        attention_mask[m_start : m_end, m_start : m_end] = 1


        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        padded_tokens_b = self.tokenizer.convert_tokens_to_ids(padded_tokens_b)
        padded_tokens_b = torch.tensor(padded_tokens_b, dtype=torch.long)

        match_ids = torch.tensor(match_ids, dtype=torch.long)


        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids,padded_tokens_b, match_ids)
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos,padded_tokens_b, match_ids)

class CaptionKaraDatasetWithConstraints(CaptionKaraDataset):
    def __init__(
        self, yaml_file,
        nms_threshold=0.85,
        max_given_constraints=3, **kwargs
    ):
        super().__init__(yaml_file, **kwargs)
        self.root=None
        boxes_tsvpath = find_file_path_in_yaml(self.cfg['cbs_box'], self.root)
        constraint2tokens_tsvpath = find_file_path_in_yaml(self.cfg['cbs_constraint'], self.root)
        tokenforms_tsvpath = find_file_path_in_yaml(self.cfg['cbs_tokenforms'], self.root)
        hierarchy_jsonpath = find_file_path_in_yaml(self.cfg['cbs_hierarchy'], self.root)
        self._boxes_reader = ConstraintBoxesReader(boxes_tsvpath)
        self._constraint_filter = ConstraintFilter(
            hierarchy_jsonpath, nms_threshold, max_given_constraints
        )
        self._fsm_builder = FiniteStateMachineBuilder(self.tokenizer,
                constraint2tokens_tsvpath, tokenforms_tsvpath,
                max_given_constraints)


    def __iter__(self):
        for img in self.iterable_ImgDataset:
            index, features = img[0], img[1]
            assert index < len(self.labels), (index, len(self.labels), len(self.iterable_ImgDataset))
            img_id, od_labels = self.labels[index]
            if self.captions:
                img_id, caps = self.captions[index]
            else:
                caps = None
            example = self.tensorizer.tensorize_example(text_a=caps, text_b=od_labels, img_feat=torch.Tensor(features),
                sequence_a_segment_id=1, sequence_b_segment_id=0)  #tag -> 0 caption -> 1
            
            constraint_boxes = self._boxes_reader[img_id]
            candidates = self._constraint_filter(
                constraint_boxes["boxes"], constraint_boxes["class_names"], constraint_boxes["scores"]
            )
            num_constraints = len(candidates)
            fsm, nstates = self._fsm_builder.build(candidates)            
            yield img_id, example[:-1] + (fsm, num_constraints, candidates, example[-1])







def build_dataset(yaml_file, tokenizer, args, is_train=True, scst=False, is_distributed=False):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file), yaml_file

    if is_train and not scst: #istrain and not args.scst
        return CaptionKaraDataset(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,
            is_train=True, mask_prob=args.mask_prob, max_masked_tokens=args.max_masked_tokens,
            cap_segment_id=args.cap_segment_id, tag_segment_id=args.tag_segment_id, scst=scst,
            is_distributed=is_distributed, match_threshold=args.match_threshold)
    if args.use_cbs:
        dataset_class = CaptionKaraDatasetWithConstraints
    else:
        dataset_class = CaptionKaraDataset
    return dataset_class(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_gen_length,
            is_train=is_train, cap_segment_id=args.cap_segment_id, tag_segment_id=args.tag_segment_id,
            scst=scst, is_distributed=is_distributed, match_threshold=args.match_threshold)


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
        is_train=is_train, scst=args.scst, is_distributed=is_distributed)
    if is_train:
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        #iters_per_batch = len(dataset) // images_per_batch
        #num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        # logger.info("Total training steps {}".format(num_iters))
    else:
        images_per_gpu = args.per_gpu_eval_batch_size

    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, 
        batch_size=images_per_gpu,
        pin_memory=True,
    )
    return data_loader


def save_checkpoint(model, tokenizer, args, epoch, iteration, optimizer, scheduler, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            torch.save({
                'epoch': epoch,
                'global_step': iteration,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, op.join(checkpoint_dir, 'epoch_step_opt_sc.bin'))
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
    if args.amp:
        from apex import amp
        try:
            # from apex.optimizers import FP16_Optimizer
            #from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(grouped_parameters,
                              lr=args.learning_rate,
                              eps=args.adam_epsilon,
                              bias_correction=False)
    else:
        optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')#'02')
    if args.distributed:
        if args.amp:
            try:
                from apex.parallel import DistributedDataParallel as DDP 
            except ImportError:
                raise ImportError(
                    'Please install apex from https://www.github.com/nvidia/apex to use distributed fp16 for training.')
            model = DDP(model)#,delay_allreduce=True)
        else:   
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], 
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    # restore scheduler, optimizer
    if os.path.exists(op.join(args.model_name_or_path, 'epoch_step_opt_sc.bin')) \
        and not 'VIVO' in args.model_name_or_path \
        and not args.only_load_ckpt: #no need to reload scheduler and optimizer for 
        training_state = torch.load(op.join(args.model_name_or_path, 'epoch_step_opt_sc.bin'),
            map_location=torch.device('cuda:{}'.format(args.local_rank)))
        if args.load_optimizer:
            optimizer.load_state_dict(training_state['optimizer'])
        scheduler.load_state_dict(training_state['scheduler'])
        if args.load_optimizer:
            logger.info("  Loading optimizer and scheduler from {}".format(op.join(args.model_name_or_path, 'epoch_step_opt_sc.bin')))
        else:
            logger.info("  Loading scheduler from {}".format(op.join(args.model_name_or_path, 'epoch_step_opt_sc.bin')))
        start_epoch = training_state['epoch']+1
    else:
        start_epoch = 0

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.per_gpu_train_batch_size * get_world_size() * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  len train_dataloader = %d", len(train_dataloader))
    logger.info("  Total optimization steps = %d", t_total)

    if args.scst:
        scst_criterion = ScstRewardCriterion(
            cider_cached_tokens=op.join(args.data_dir, args.cider_cached_tokens),
            baseline_type=args.sc_baseline_type,
        )
        logger.info("  SCST training...")


    global_step, global_loss, global_acc = start_epoch*t_total/args.num_train_epochs,  0.0, 0.0
    model.zero_grad()
    eval_log, best_score = {}, {}
    for name in val_dataset:
        eval_log[name] = []
        best_score[name] = 0
    checkpoint_dir = None

    if not args.distributed or args.local_rank == 0:
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')


    for epoch in range(start_epoch, int(args.num_train_epochs)):
        train_dataloader.dataset.set_epoch(epoch)
        for name in val_dataset:
            val_dataset[name].dataset.set_epoch(epoch)

        if epoch==start_epoch and is_main_process():
            checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step, 
                optimizer, scheduler, num_trial=10)

            
            if args.evaluate_during_training: 
                if args.img_embedding_type=='grid' and writer:
                    draw_position_embeddings(writer, model.module.bert, 
                        args.grid_n, grid_factor=args.grid_factor, global_step=global_step, save_dir=args.output_dir)
                for name in val_dataset:
                    logger.info(name+": Perform evaluation at step: %d epoch %d" % (global_step, epoch))
                    evaluate_file = evaluate(args, val_dataset[name], model, tokenizer,
                            checkpoint_dir, epoch-1, dataset=name)
                    with open(evaluate_file, 'r') as f:
                        res = json.load(f)
                    best_score[name] = max(best_score[name], res['CIDEr'])
                    res['epoch'] = epoch
                    res['global_step'] = 0
                    res['best_CIDEr'] = best_score[name]
                    eval_log[name].append(res)
                    with open(args.output_dir + '/eval_logs_{}_{}.json'.format(name, epoch), 'w') as f:
                        json.dump(eval_log[name], f)
                    val_dataset[name].dataset.set_epoch(epoch)

                    if writer:
                        writer.add_scalar('{}_CIDEr'.format(name), res['CIDEr'], global_step=global_step)
             

        if get_world_size() > 1:
            torch.distributed.barrier()

        for step, (img_keys, batch) in enumerate(train_dataloader):
            tag_tokens, match_ids = batch[-2], batch[-1]
            # print(img_keys[0])
            # print(tag_tokens[0])
            # print(match_ids[0])

            #input()
            #batch = batch[:-2]
            batch = tuple(t.to(args.device) for t in batch)
            #evaluate at the start!
            if not args.scst:
                model.train()
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3], 
                    'masked_pos': batch[4], 'masked_ids': batch[5],
                    'match_ids': batch[7]
                }
                torch.set_printoptions(profile="full")
                # print(img_keys[0])
                # print('input_ids', inputs['input_ids'][0])
                # print('attention_mask', inputs['attention_mask'][0])
                # input()
                # print('token_type_ids',inputs['token_type_ids'][0])
                # print('masked_pos',inputs['masked_pos'][0])
                # print('masked_ids',inputs['masked_ids'][0])
                # print('img_feats',inputs['img_feats'][0][0:5,0:10])
                # 4)
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                masked_ids = inputs['masked_ids']
                masked_ids = masked_ids[masked_ids != 0]
                batch_score = compute_score_with_logits(logits, masked_ids)
                batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])
            else:
                loss = scst_train_iter(args, train_dataloader, model, scst_criterion, img_keys, batch, tokenizer)
                batch_acc = scst_criterion.get_score()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            global_acc += batch_acc

            if not args.distributed or args.local_rank == 0:
                pbar(step)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f}, " \
                        "score: {:.4f}".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss,  batch_acc)
                    )
                    if writer != None:
                        writer.add_scalar('loss', loss, global_step=global_step)
                        writer.add_scalar('batch_acc', batch_acc, global_step=global_step)
                        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step=global_step)


        if is_main_process():
            checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step, 
                optimizer, scheduler, num_trial=10)
        # evaluation

        if args.evaluate_during_training and is_main_process(): 
            if args.img_embedding_type=='grid' and writer:
                draw_position_embeddings(writer, model.module.bert, 
                    args.grid_n, grid_factor=args.grid_factor, global_step=global_step, save_dir=args.output_dir)
            for name in val_dataset:
                logger.info(name+": Perform evaluation at step: %d epoch %d" % (global_step, epoch))
                evaluate_file = evaluate(args, val_dataset[name], model, tokenizer,
                        checkpoint_dir, epoch, dataset=name)
                with open(evaluate_file, 'r') as f:
                    res = json.load(f)
                if writer:
                    writer.add_scalar('{}_CIDEr'.format(name), res['CIDEr'], global_step=global_step)
                best_score[name] = max(best_score[name], res['CIDEr'])
                res['epoch'] = epoch
                res['global_step'] = 0
                res['best_CIDEr'] = best_score[name]
                eval_log[name].append(res)
                with open(args.output_dir + '/eval_logs_{}_{}.json'.format(name, epoch), 'w') as f:
                    json.dump(eval_log[name], f)

        if get_world_size() > 1:
            torch.distributed.barrier()
        print()
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
    # if is_main_process():
    #     for k, gt, greedy, sample in zip(img_keys, gt_res, greedy_res, sample_res):
    #         logger.info('k:{}'.format(k))
    #         logger.info('gt:{}'.format(gt))
    #         logger.info('greedy:{}'.format(greedy))
    #         logger.info('sample:{}'.format(sample))
    #         logger.info('\n')
    #         break
    return loss


def get_predict_file(output_dir, yaml_file, args, epoch=None,data='coco'):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
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
    if epoch:
        cc.append('epoch_{}'.format(epoch))
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


def evaluate(args, val_dataloader, model, tokenizer, output_dir, epoch, dataset='coco'):
    predict_file = get_predict_file(output_dir,
            val_dataloader.dataset.yaml_file, args, epoch, dataset) #name
    test(args, val_dataloader, model, tokenizer, predict_file)

    evaluate_file = get_evaluate_file(predict_file)
    caption_file = val_dataloader.dataset.get_caption_file_in_coco_format()
    data = val_dataloader.dataset.yaml_file.split('/')[-2]
    logger.info('Evaluate {}'.format(dataset))
    result, img2eval = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file,return_imgscores=True)
    logger.info('evaluation result: {}'.format(str(result)))
    logger.info('evaluation result saved to {}'.format(evaluate_file))
    with open(evaluate_file.split('.json')[0]+'img2eval.json','w') as f:
        json.dump(img2eval, f)
    return evaluate_file


def test(args, test_dataloader, model, tokenizer, predict_file):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token, 
        tokenizer.pad_token, tokenizer.mask_token, '.'])

    model.eval()
    inputs_param = {'is_decode': True,
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.num_beams,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_keep_best": args.num_keep_best,
    }
    if args.use_cbs:
        #assert False, 'Not support cbs now'
        inputs_param.update({'use_cbs': True,
            'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
        })
    def gen_rows(predict_file):
        time_meter = 0
        id2constraints = {}
        id2tags = {}
        id2traces = {}

        with open('select_ids.txt','r') as f:
            lines = f.readlines()
            select_ids = [l.strip() for l in lines]
        #select_ids = []
        # print(select_ids)

            


        with torch.no_grad():
            for step, (img_keys, batch) in tqdm(enumerate(test_dataloader)):
                # added by yutong to output constrain
                # if not str(img_keys[0]) in select_ids:
                #     #print(str(img_keys[0]))
                #     # input()
                #     yield img_keys[0], json.dumps(({'caption': '', 'conf': 0}))
                #     continue

                if args.use_cbs:
                    assert args.per_gpu_eval_batch_size==1, args.per_gpu_eval_batch_size
                    constraints = batch[-1]
                    id2constraints[img_keys[0]] = constraints
                    # print(img_keys[0])
                    # print('constraints', constraints)
                    if len(constraints)==0:
                        yield img_keys[0], json.dumps(({'caption': '', 'conf': 0}))
                        continue

                    batch = batch[:-1]
                    #!!!!
                    #continue
                tag_tokens = batch[5]
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4], 'match_ids': batch[6]
                }
                torch.set_printoptions(profile="full")

                if args.use_cbs:
                    inputs.update({
                        'fsm': batch[5],
                        'num_constraints': batch[6],
                    })
                # print(img_keys[0])
                # print(constraints)
                # input()
                inputs.update(inputs_param)
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])
                #traces = outputs[2]

                #assert args.per_gpu_eval_batch_size ==1
                #id2traces[img_keys[0].item() if isinstance(img_keys[0], torch.Tensor) else img_keys[0]] = traces

                for img_key, caps, confs, tag_token in zip(img_keys, all_caps, all_confs, tag_tokens):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    id2tags[img_key] = tokenizer.decode(tag_token.tolist(), skip_special_tokens=True)

                    # import pickle
                    # outputfile = op.join(args.output_dir,\
                    #     op.splitext(predict_file)[0] + '_{}_traces.pkl'.format(img_key))
                    # with open(outputfile,'wb') as f:
                    #     pickle.dump(traces, f)

                    yield img_key, json.dumps(res)
            with open(op.splitext(predict_file)[0] + '_inputtags.json','w') as f:
                json.dump(id2tags,f)
            with open(op.splitext(predict_file)[0] + '_constraints_{}.json'.format(3),'w') as f:
                json.dump(id2constraints,f)

            # import pickle
            # outputfile = op.join(args.output_dir,\
            #     op.splitext(predict_file)[0] + '_traces.pkl')
            # with open(outputfile,'wb') as f:
            #     pickle.dump(id2traces, f)

        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step+1)))

    tsv_writer(gen_rows(predict_file), predict_file)
    print('Save constraints to ', op.splitext(predict_file)[0] + '_constraints_{}.json'.format(3))
    # cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
    #     op.splitext(predict_file)[1] for i in range(world_size)]
    # concat_tsv_files(cache_files, predict_file)
    # delete_tsv_files(cache_files)
    # reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys, predict_file)


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args    
        else: # args.scst
            logger.info('SCST training max_seq_length{} max_gen_length{}'.format(args.max_seq_length,
                args.max_gen_length))
            return args
    else:
        #assert args.do_test or args.do_eval
        args_load = torch.load(op.join(args.model_name_or_path,'training_args.bin'))
        args_load.num_workers = 1
        args_load.output_dir = args.output_dir
        args_load.eval_model_dir = args.eval_model_dir
        args_load.max_gen_length, args_load.num_beams = args.max_gen_length, args.num_beams
        args_load.use_cbs = args.use_cbs
        if args.use_cbs:
            assert args.per_gpu_eval_batch_size==1
        args_load.num_keep_best = args.num_keep_best
        args_load.repetition_penalty, args_load.length_penalty = args.repetition_penalty, args.length_penalty
        args_load.data_dir = args.data_dir
        args_load.do_train, args_load.do_eval, args_load.do_test = args.do_train, args.do_eval, args.do_test
        args_load.model_name_or_path = args.model_name_or_path
        args_load.nocaps_evaluate_dir = args.nocaps_evaluate_dir
        args_load.evaluate_nocaps, args_load.evaluate_coco = args.evaluate_nocaps, args.evaluate_coco
        args_load.min_constraints_to_satisfy = args.min_constraints_to_satisfy
        args_load.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size
        args_load.nocaps_split = args.nocaps_split#.split(',')
        args_load.cap_segment_id = args.cap_segment_id
        args_load.tag_segment_id = args.tag_segment_id
        if hasattr(args_load, 'img_embedding_type'):
            assert args_load.img_embedding_type == args.img_embedding_type, \
                'resumed img_embedding_type {}, load img_embedding_type {}'.format(args_load.img_embedding_type,args.img_embedding_type)
        args_load.img_embedding_type = args.img_embedding_type

        if hasattr(args_load, 'grid_n'):
            assert args_load.grid_n == args.grid_n, \
                'resumed grid_n {}, load grid_n {}'.format(args_load.grid_n,args.grid_n)
        args_load.grid_n = args.grid_n

        if hasattr(args_load, 'grid_factor'):
            assert args_load.grid_factor == args.grid_factor, \
                'resumed grid_factor {}, load grid_factor {}'.format(args_load.grid_factor,args.grid_factor)
        args_load.grid_factor = args.grid_factor

        if hasattr(args_load, 'match_threshold'):
            assert args.match_threshold == args_load.match_threshold, \
                'resumed match_threshold {}, load match_threshold {}'.format(args_load.match_threshold,args.match_threshold)
        args_load.match_threshold = args.match_threshold

        if args_load.scst==False:
            logger.info('Inference Override --  ')
            logger.info('max_seq_length {} --> '.format(args_load.max_seq_length))
            max_od_labels_len = args_load.max_seq_length - args_load.max_seq_a_length
            args_load.max_seq_length = args_load.max_gen_length + max_od_labels_len
            logger.info('max_seq_length {}={}+{}'.format(args_load.max_seq_length,
                args_load.max_gen_length, max_od_labels_len))
        else:
            logger.info('Inference max_seq_length{} max_gen_length{}'.format(args_load.max_seq_length,
                args_load.max_gen_length))

        
    return args_load

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
    parser.add_argument("--data_dir", default='datasets/coco_caption', type=str, required=False,
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
    parser.add_argument("--loss_type", default='sfmx', type=str, 
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int, 
                        help="The maximum sequence length for caption.")
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
    parser.add_argument('--save_epochs', type=int, default=1, 
                        help="Save checkpoint every X epochs. Will also perform evaluatin.")
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
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--load_optimizer', action='store_true', help='whether to load optimizer')
    parser.add_argument('--only_load_ckpt', action='store_true', help='not to load optimizer, scheduler and epoch')
    parser.add_argument('--evaluate_nocaps', action='store_true')
    parser.add_argument('--evaluate_coco', action='store_true')
    parser.add_argument('--nocaps_evaluate_dir', type=str, default='Null')
    parser.add_argument('--nocaps_split',type=str,default='near,out,in')
    parser.add_argument('--cap_segment_id',type=int,default=1)
    parser.add_argument('--tag_segment_id',type=int,default=0)

    #---image_embedding
    parser.add_argument('--img_embedding_type', type=str, default='continuous')
    parser.add_argument('--grid_n', type=int, default=32)
    parser.add_argument('--grid_factor',  nargs='+',default=['width','height','cx','cy'])

    #---match
    parser.add_argument('--match_threshold',type=float, default=1.0) #1.0 -> no matching augmentation
    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.nocaps_split = args.nocaps_split.split(',')
    args.num_gpus = get_world_size()
    args.distributed = True#args.num_gpus > 1
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda:{}'.format(args.local_rank))
    if args.amp:
        from apex import amp
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("coco_finetune", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)

    args = restore_training_settings(args)
    if not args.distributed or (args.local_rank==0):
        writer = SummaryWriter(logdir=output_dir)
    else:
        writer = None
    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    if args.do_train:
        assert args.model_name_or_path is not None
        config = config_class.from_pretrained(args.config_name if args.config_name else \
                args.model_name_or_path, num_labels=args.num_labels, finetuning_task='image_captioning')
        if args.scst:
            # avoid using too much memory
            config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.tie_weights = args.tie_weights
        config.freeze_embedding = args.freeze_embedding
        config.label_smoothing = args.label_smoothing
        config.drop_worst_ratio = args.drop_worst_ratio
        config.drop_worst_after = args.drop_worst_after
        config.img_embedding_type = args.img_embedding_type
        config.grid_n = args.grid_n
        config.grid_factor = args.grid_factor
        config.use_match = args.match_threshold<1
        config.max_seq_a_len = args.max_seq_a_length
        model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        
        config.img_embedding_type = args.img_embedding_type
        config.grid_n = args.grid_n
        config.grid_factor = args.grid_factor
        config.use_match = args.match_threshold<1

        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataloader = make_data_loader(args, args.train_yaml, tokenizer,
            args.distributed, is_train=True)
        val_dataloader = {}
        if args.evaluate_during_training:
            val_dataloader['coco'] = make_data_loader(args, args.val_yaml, tokenizer,
                args.distributed, is_train=False)
        if args.evaluate_nocaps:
            assert op.isdir(args.nocaps_evaluate_dir)
            for split in args.nocaps_split:
                yaml_file = op.join(args.nocaps_evaluate_dir, '{}.yaml'.format(split))
                val_dataloader['nocaps_{}'.format(split)]=make_data_loader(
                    args, yaml_file, tokenizer, args.distributed, is_train=False)
                # print('nocaps_{}'.format(split), len(val_dataloader['nocaps_{}'.format(split)].dataset))
                # input()
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer, writer)

        # test the last checkpoint after training
        if args.do_eval:
            logger.info("Evaluate on dataset: " + args.test_yaml)
            test_dataloader = make_data_loader(args, args.test_yaml, 
                tokenizer, args.distributed, is_train=False)
            evaluate(args, test_dataloader, model, tokenizer, last_checkpoint, 'last')

    # inference and evaluation
    elif args.do_test or args.do_eval:
        test_dataloader = {}
        assert args.evaluate_coco or args.evaluate_nocaps
        if args.evaluate_coco:
            logger.info("Evaluate on dataset: " + args.test_yaml)
            test_dataloader['coco'] = make_data_loader(args, args.test_yaml,
                tokenizer, args.distributed, is_train=False)
        if args.evaluate_nocaps:
            assert op.isdir(args.nocaps_evaluate_dir)
            for split in args.nocaps_split:
                yaml_file = op.join(args.nocaps_evaluate_dir, '{}.yaml'.format(split))
                test_dataloader['nocaps_{}'.format(split)]=make_data_loader(
                    args, yaml_file, tokenizer, args.distributed, is_train=False)            

        for name in test_dataloader:
            if not args.do_eval:  #no need to produce evaluation results (bleu, meteor...)
                predict_file = get_predict_file(checkpoint, test_dataloader[name].dataset.yaml_file, args,
                    None,name)
                test(args, test_dataloader[name], model, tokenizer, predict_file)
                logger.info("Prediction {} results saved to: {}".format(name, predict_file))
                #coco format
                coco_file = predict_file.replace('.tsv','.coco_format.json')
                convert_tsv_to_coco_format(predict_file, coco_file)
                logger.info("Prediction {} results (coco format) saved to: {}".format(name, coco_file))

            else: # produce evaluation results
                evaluate_file = evaluate(args, test_dataloader[name], model, tokenizer,
                        checkpoint, None, dataset=name)
                logger.info("Evaluation results saved to: {}".format(evaluate_file))

if __name__ == "__main__":
    main()
