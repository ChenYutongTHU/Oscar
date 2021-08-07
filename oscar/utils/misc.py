# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

import errno
import os
import os.path as op
import yaml
import random
import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt

def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def cosine_similarity(arr): #N,d
    #normalize
    eps = 1e-10
    norm = np.sqrt(np.sum(arr*arr, axis=1,keepdims=True))+eps
    arr = arr/norm
    cos = np.dot(arr, np.transpose(arr))
    return cos

def draw_position_embeddings(writer, model, N, global_step, grid_factor, save_dir=None):
    if not save_dir:
        save_dir = op.join(save_dir, 'grid_embeddings')
        mkdir(save_dir)

    for name in grid_factor:
        embed_name = 'img_embedding_{}'.format(name)
        embedding = getattr(model,embed_name)
        tensor = embedding.weight
        arr = tensor.detach().cpu().numpy()
        N_ = N*2 if name=='area' else N
        sim = cosine_similarity(arr[:N_,])
        fig, ax = plt.subplots()
        ax.matshow(sim)
        writer.add_figure(embed_name,fig,global_step=global_step)
        fig.savefig(op.join(save_dir, '{}_{}.png'.format(embed_name, global_step)))
    return 



def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
