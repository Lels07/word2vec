from utils import *
from config import *
from cbow import CBOWModeler
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

test = torch.load("./fr/cbow100-fat/model4.pth")

idx_to_word = test["idx_to_word"]
word_to_idx = test["word_to_idx"]
total_loss = test["total_loss"]

model = CBOWModeler(len(idx_to_word), 100)
model.load_state_dict(test["cbow_state_dict"])

embeds = model.embeddings.weight.data.cpu()

tsne = TSNE(n_components = 2).fit_transform(embeds.cpu())
test_words = ["roi", "reine", "lourd","léger", "france", "allemagne", "paris", "berlin"]
x, y = [], []
annotations = []
for idx, coord in enumerate(tsne):
    # print(coord)
    annotations.append(idx_to_word[idx])
    x.append(coord[0])
    y.append(coord[1])

plt.figure(figsize = (10, 10))
for i in range(len(test_words)):
    word = test_words[i]
    #print('word: ', word)
    vocab_idx = word_to_idx[word]
    # print('vocab_idx: ', vocab_idx)
    plt.scatter(x[vocab_idx], y[vocab_idx])
    plt.annotate(word, xy = (x[vocab_idx], y[vocab_idx]), \
        ha='right',va='bottom')

plt.savefig("w2v.png")
plt.show()
