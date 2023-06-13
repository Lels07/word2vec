
import torch, os, pickle, shutil
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
from config import *
from utils import *
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    write_summary = True
except:
    write_summary = False

from cbow import CBOWModeler


test = torch.load("./fr/cbow100-fat/model4.pth")

idx_to_word = test["idx_to_word"]
word_to_idx = test["word_to_idx"]

model = CBOWModeler(len(idx_to_word), 100)
model.load_state_dict(test["cbow_state_dict"])

embeds = model.embeddings.weight.data.cpu()

def vec( word):
    return embeds[word_to_idx[word]]

inp = vec("jeune") - vec("vieux") + vec("léger")                                
print('inp.shape: ', inp.shape)

emb_ranking_top, euclidean_dis_top = nearest_neighbour(inp, embeds, 13)
print('emb_ranking_top: ', emb_ranking_top, type(emb_ranking_top))

for idx, t in enumerate(emb_ranking_top):
    print(idx_to_word[t], euclidean_dis_top[idx])
