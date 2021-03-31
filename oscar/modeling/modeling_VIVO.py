from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from oscar_transformers.pytorch_transformers.modeling_bert import (BertEmbeddings, 
        BertSelfAttention, BertAttention, BertEncoder, BertLayer, 
        BertSelfOutput, BertIntermediate, BertOutput,
        BertPooler, BertLayerNorm, BertPreTrainedModel,
        BertPredictionHeadTransform, BertOnlyMLMHead, BertLMPredictionHead,
        BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        load_tf_weights_in_bert)
from .modeling_utils import CaptionPreTrainedModel, ImgPreTrainedModel
from .modeling_bert import BertImgModel
from itertools import permutations

logger = logging.getLogger(__name__)

class BertHungarianLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.CE = torch.nn.CrossEntropyLoss(reduction='none')
        self.softmax = torch.nn.Softmax(dim=-1)
        self.whole_word = config.whole_word

    def forward(self, logits, target, whole_word_masked=None, num_masked=None):
        #whole_word_masked
        #[[st,ed],[st,ed],[-1,-1],[-1,-1]]
        eps = self.label_smoothing
        n_class = logits.size(1) #num_masked, n_class
        if self.whole_word:
            assert num_masked!=None
            M = num_masked
        else:
            M = logits.size(0)
        prob = self.softmax(logits) #num_masked, n_class
        C_max = -100
        target_best = None
        records = []
        for p in permutations(range(0,M)): #assign target i to the p[i]-th predictions
            p = list(p)
            if self.whole_word:
                target_p = []
                for i in p:
                    s, e = whole_word_masked[i][0],whole_word_masked[i][1]
                    target_p.append(target[s:e])
                target_p = torch.cat(target_p, dim=0)
            else:
                target_p = target[p] # num_masked
            C_ = torch.gather(prob,1,target_p.unsqueeze(1))  #num_masked, 1
            C_ = torch.sum(C_)
            records.append(C_)
            if C_>C_max:
                target_best = target_p
                C_max = C_
        assert target_best!=None, (M,records)
        loss = self.CE(input=logits, target=target_best) #num_masked, 
        return loss, target_best

class BertForVIVOPretraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForVIVOPretraining, self).__init__(config)
        self.config = config
        self.bert = BertImgModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.loss = BertHungarianLoss(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, input_ids, img_feats, attention_mask, masked_pos, 
            whole_word_masked=None, num_masked_tags=None, masked_ids=None, 
            token_type_ids=None, position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, img_feats=img_feats, attention_mask=attention_mask, 
            position_ids=position_ids, token_type_ids=token_type_ids,
            head_mask=head_mask,
            encoder_history_states=None)
        #masked_ids  B, nm
        #masked_pos B,len_a
        sequence_output = outputs[0][:, :masked_pos.shape[-1], :] # last layer hidden states B,L(len(text_a)),H 
        batch_size = sequence_output.shape[0]
        losses, ranked_masked_ids, class_logits_flatten = [],[],[]
        #print('masked_ids[:2]', masked_ids[:2])
        for b in range(batch_size):
            one_sequence_output = sequence_output[b] #len_a, h
            one_masked_pos = masked_pos[b]
            one_masked_ids = masked_ids[b][masked_ids[b]!=0]  #(nm,)  (num_masked,)
#             print(one_masked_pos)
#             print(one_masked_ids)
#             input()
            one_sequence_output_masked = one_sequence_output[one_masked_pos==1,:] #num_masked, h
            assert one_sequence_output_masked.shape[0] == one_masked_ids.shape[0]
            class_logits = self.cls(one_sequence_output_masked) # num_masked, h -> num_masked, V
            one_loss, ids = self.loss(class_logits, one_masked_ids, whole_word_masked[b], num_masked_tags[b]) #scalar
            losses.append(one_loss) #num_masked,
            ranked_masked_ids.append(ids)
            class_logits_flatten.append(class_logits)
        #losses [n1,n2,n3,]
        masked_losses = torch.mean(torch.cat(losses, dim=0))
        ranked_masked_ids = torch.cat(ranked_masked_ids, dim=0)
        class_logits_flatten = torch.cat(class_logits_flatten, dim=0)
        outputs = (masked_losses, class_logits_flatten, ranked_masked_ids) + outputs[2:]
        return outputs



