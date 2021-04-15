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
import pickle


class EmbeddingDataset(Dataset):
    def __init__(self, data_file, tokenizer=None, 
            max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40, **kwargs):
        self.data_file = data_file
        with open(self.data_file,'r') as f:
            self.raw_data = json.load(f)
        self.makelist()

        self.tokenizer = tokenizer
        self.tensorizer = EmbeddingTensorizer(self.tokenizer, max_img_seq_length, max_seq_length)

    def makelist(self):
        self.datalist = []
        for concept in self.raw_data:
            for dic in self.raw_data[concept]:
                if dic['tag_ind']==None:
                    continue
                info_ = {'concept':concept,
                'imgid': dic['id'],
                #'tags':[dic['labels'][dic['tag_ind']['class']]+[t['class'] for t in i,enumerate(dic['labels']) if not i==dic['tag_ind']],
                'tag_ind': 0, #dic['tag_ind'],
                'region_ind':dic['region_ind'],
                'imgfeatures': torch.Tensor(np.frombuffer(base64.b64decode(dic['features']),np.float32).reshape((-1,2054))[:])
                }
                #print(dic['tag_ind'])
                info_['tags'] = [dic['labels'][dic['tag_ind']]['class']]
                info_['tags'] = info_['tags'] + [t['class'] for i,t in enumerate(dic['labels']) if not i==dic['tag_ind']]
                self.datalist.append(info_)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        dataitem = self.datalist[idx]
        imgid, concept = dataitem['imgid'], dataitem['concept']
        example = self.tensorizer.tensorize_example(
                text_a=dataitem['tags'], img_feat=dataitem['imgfeatures'],
                tag_ind=dataitem['tag_ind'], region_ind=dataitem['region_ind'])
        return concept, imgid, example

class EmbeddingTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_img_seq_length = max_img_seq_length

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

    def tensorize_example(self, text_a, img_feat, tag_ind, region_ind,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a, tokens_a_segment, tags = self.tokenize_tags(text_a, self.max_seq_length-2)
        num_tags = len(tags)
        #assert num_tags<=self.max_tags, tags
        tags = ', '.join(tags)

        # tag_pos = torch.zeros(self.max_seq_length, dtype=torch.int)
        # for i in range(s,t):
        #     tag_pos[i] = 1
        s, t = tokens_a_segment[tag_ind]
        tag_pos = torch.tensor([i for i in range(s,t)])

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_len = len(tokens)

        padding_len = self.max_seq_length - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_length:
            if region_ind>=self.max_img_seq_length: #the target region is truncated
                img_feat[self.max_img_seq_length-1,:] = img_feat[region_ind,:]
                region_ind = self.max_img_seq_length-1
            img_feat = img_feat[0 : self.max_img_seq_length, ] 
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_length - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
        region_ind = self.max_seq_length+region_ind

        max_len = self.max_seq_length + self.max_img_seq_length
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # T: tags R: image region
        t_start, t_end = 0, seq_len
        r_start, r_end = self.max_seq_length, self.max_seq_length + img_len
        # full attention for T-T R-R T-R
        attention_mask[t_start : t_end, t_start : t_end] = 1
        attention_mask[r_start : r_end, r_start : r_end] = 1
        attention_mask[t_start : t_end, r_start : r_end] = 1
        attention_mask[r_start : r_end, t_start : t_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat, tag_pos, region_ind, tags)

def make_data_loader(args, tokenizer):
    dataset = EmbeddingDataset(data_file=args.data_file, tokenizer=tokenizer,
            max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=1,
        pin_memory=True,
    )
    return data_loader

def compute_embeddings(args, dataloader, model):
    results = []
    with torch.no_grad():
        for concept, imgid, instance in tqdm(dataloader):
            input_ids, attention_mask, segment_ids, img_feat, tag_pos, region_ind, tags = instance
            res = {'concept':concept, 'imgid':imgid,'tag_embed':None, 'region_embed':None}
            # print(res)
            # print('input_ids',input_ids)
            # print('segment_ids',input_ids)
            #print('tag_pos',tag_pos)
            # print('region_ind', region_ind)
            #print('tags', tags)
            # print(attention_mask)
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            img_feat = img_feat.to(args.device)
            tag_pos = tag_pos.to(args.device)
            region_ind = region_ind.to(args.device)
            #input()
            outputs = model.compute_embeddings(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, img_feats=img_feat)
            embeddings = outputs[0] # L, H
            #print(embeddings.shape)
            #print(tag_pos.shape, tag_pos)
            #print(region_ind.shape, region_ind)
            res['tag_embed'] = torch.mean(embeddings[tag_pos[0]], dim=0).cpu().detach().numpy()
            res['region_embed'] = embeddings[region_ind[0],:].cpu().detach().numpy()
            results.append(res)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',required=True)
    parser.add_argument("--args_dir", default=None, type=str, required=True,
                        help="path to train_args.bin directory")
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    #                     help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")

    args = parser.parse_args()
    tmp_dir = args.output_dir
    data_file = args.data_file
    mkdir(tmp_dir)
    global logger
    args = torch.load(op.join(args.args_dir,'training_args.bin'))
    args.output_dir = tmp_dir
    args.data_file = data_file

    logger = setup_logger("embedding_computing", args.output_dir, -1)
    config_class, model_class, tokenizer_class = BertConfig, BertForVIVOPretraining, BertTokenizer
    assert args.model_name_or_path is not None
    config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=args.num_labels, finetuning_task='VIVO_pretraining') #?
    config.whole_word = args.whole_word
    config.img_layer_norm_eps = args.img_layer_norm_eps
    config.use_img_layernorm = args.use_img_layernorm
    config.task = 'VIVO'
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    #config.hidden_dropout_prob = args.drop_out
    #config.tie_weights = args.tie_weights
    #config.freeze_embedding = args.freeze_embedding
    config.output_hidden_states = True
    model = model_class.from_pretrained(args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    model.to(args.device)
    model.eval()
    logger.info("Evaluation parameters %s", args)

    dataloader = make_data_loader(args, tokenizer)
    results = compute_embeddings(args,dataloader, model)
    with open(op.join(args.output_dir,'embedding.pkl'),'wb') as f:
        pickle.dump(results,f)


