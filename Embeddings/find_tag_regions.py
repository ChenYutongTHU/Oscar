import json, os, csv
import numpy as np
from tqdm import tqdm
import base64
def read_features(id_, id2lineidx, fp,lineidx):
    l = id2lineidx[id_]
    pos = lineidx[l]
    fp.seek(pos)
    return fp.readline()

def read_labels(id_, id2lineidx, fp,lineidx):
    l = id2lineidx[id_]
    pos = lineidx[l]
    fp.seek(pos)
    return fp.readline()

def makegrid(p,res):
    grid = np.zeros([res,res])
    p = (p*res).astype(np.int32)
    x1,y1,x2,y2 = p
    grid[y1:y2+1,x1:x2+1] = 1
    return grid
    
def compute_iou(p1, p2, res=1000,eps=1e-9):
    grid1 = makegrid(p1,res)
    grid2 = makegrid(p2,res)
    iou = np.sum(grid1*grid2)/np.sum((eps+(grid1+grid2).astype(np.bool).astype(np.int32)))
    return iou

with open('/data/private/NocapsData/VIVO_pretrain_data/train.label.lineidx','r') as f:
    lineidx2pos = f.readlines()
    lineidx2pos = [int(l.strip()) for l in lineidx2pos]
with open('/data/private/NocapsData/VIVO_pretrain_data/features.lineidx','r') as f:
    lineidx2pos_feature = f.readlines()
    lineidx2pos_feature = [int(l.strip()) for l in lineidx2pos_feature]
with open('/data/private/NocapsData/model_0060000/imageid2idx.json','r') as f:
    id2lineidx = json.load(f)  
fp = open('/data/private/NocapsData/VIVO_pretrain_data/train.label.tsv','r')
fp_feature = open('/data/private/NocapsData/model_0060000/features.tsv','r')

with open('class2imageid_.json','r') as f:
    cls2id = json.load(f)

cls2info = {} # 'id':,'tag_ind','region_ind':
threshold=0.8
for c,imglist in tqdm(cls2id.items()):
    cls2info[c] = []
    for img in imglist:
        it = {'id':img, 'tag_ind':None, 'region_ind':None}
        labels = read_labels(img, id2lineidx, fp, lineidx2pos)
        labels = json.loads(labels.split('\t')[1])
        it['labels'] = labels
        features = read_features(img, id2lineidx, fp_feature, lineidx2pos_feature)
        num_boxes, features = int(features.split('\t')[1]), features.split('\t')[2] 
        it['features'] = features
        features = np.frombuffer(base64.b64decode(features), np.float32).reshape((num_boxes, -1))[:]
        rects = [f[-6:-2] for f in features]
        for j,d in enumerate(labels):
            if d['class']==c and d['rect'] != []:
                it['tag_ind'] = j
                tag_rec = np.array(d['rect']) #x1,y1,x2,y2
                break
        ious = [compute_iou(tag_rec, r) for r in rects]
        ious_sorted_idx = np.argsort(ious)[::-1]
        region_ind = ious_sorted_idx[0]
        if ious[region_ind]>threshold and len(cls2info[c])<=20:
            it['region_ind'] = int(region_ind)
            it['region_rect'] = [float for r in rects[region_ind]]
            cls2info[c].append(it)
        if len(cls2info[c])>=20:
            break
            #url = id2url[it['id']]
            #os.system('wget {} -O /data/private/chenyutong/Oscar/Embeddings/debug/sheep/{}.jpg'.format(url, it['id']))
            # print(it, ious[region_ind])
    break
with open('class2imageid.json','w') as f:
    json.dump(cls2info,f)