
import torch, os, pickle, shutil
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
from config import *
from preprocess import Preprocess
from utils import *
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    write_summary = True
except:
    write_summary = False

from cbow import CBOWModeler


test = torch.load("./model499.pth")

idx_to_word = test["idx_to_word"]
word_to_idx = test["word_to_idx"]

model = CBOWModeler(len(idx_to_word), EMBEDDING_DIM)
model.load_state_dict(test["cbow_state_dict"])

embeds = model.embeddings.weight.data.cpu()

def vec( word):
    return embeds[word_to_idx[word]]

inp = vec("walking") - vec("walk") + vec("swim")                                
print('inp.shape: ', inp.shape)

emb_ranking_top, euclidean_dis_top = nearest_neighbour(inp, embeds, 10)
print('emb_ranking_top: ', emb_ranking_top, type(emb_ranking_top))

for idx, t in enumerate(emb_ranking_top):
    print(idx_to_word[t], euclidean_dis_top[idx])
