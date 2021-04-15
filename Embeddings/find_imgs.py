import json, os, csv
import numpy as np
from tqdm import tqdm
label_file = '/data/private/NocapsData/VIVO_pretrain_data/train.label.tsv'
id2detection = {}
with open(label_file,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    for r in reader:
        id_, detc = r
        id2detection[id_] = json.loads(detc)
print(id2detection['8e3f3b1d936e01ef'])
print(id2detection['8ef2eabfad01c8c0'])
print(id2detection['8d863055019e4105'])
input()
classes = ['sheep','zebra','swan','elephant','horse','bat',
           'motorcycle','truck','car','taxi',
           'kite','flag','pizza','fries','pasta','peach','pumpkin','strawberry','grape',
           'lavender','rose',
           'violin','drum','phone',
           'desk','chair']

cls2imgs = {}
for c in classes:
    cls2imgs[c] = []
    for id_ in tqdm(id2detection):
        for d in id2detection[id_]:
            if d['class'].lower()==c and not id_ in cls2imgs[c]:
                cls2imgs[c].append(id_)
        if len(cls2imgs[c])>=2000:
            break
    print(c, len(cls2imgs[c]))

with open('class2imageid_.json','w') as f:
    json.dump(cls2imgs,f)