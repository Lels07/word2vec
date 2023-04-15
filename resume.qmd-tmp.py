







































































































text = """Denise était venue à pied de la gare Saint-Lazare, où un train de
Cherbourg l'avait débarquée avec ses deux frères, après une nuit
passée sur la dure banquette d'un wagon de troisième classe. Elle
tenait par la main Pépé, et Jean la suivait, tous les trois brisés
du voyage, effarés et perdus, au milieu du vaste Paris, le nez
levé sur les maisons, demandant à chaque carrefour la rue de la
Michodière, dans laquelle leur oncle Baudu demeurait. Mais, comme
elle débouchait enfin sur la place Gaillon, la jeune fille
s'arrêta net de surprise."""




#| code-fold: true

import torch, string
from nltk.stem.wordnet import WordNetLemmatizer
from torch._C import dtype
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk


class Preprocess(Dataset):

    def __init__(self, raw_data, window_size, fraction_size): 

        vocab, word_to_idx, idx_to_word, exemples_data, training_exemples, training_data_raw = self.tokenize(raw_data, window_size, fraction_size)

        self.training_exemples = training_exemples
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.window_size = window_size
        self.exemples_data = exemples_data
        self.training_data_raw = training_data_raw

        # self.data = torch.tensor(training_exemples, dtype=torch.long)
        self.data_x = torch.tensor([ex[0] for ex in training_exemples], dtype=torch.long)
        self.data_y = torch.tensor([ex[1] for ex in training_exemples], dtype=torch.long)
        # self.data = training_exemples


    def __len__(self):
        
        return len(self.data_x)
       

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def get_training_data(self, list_token, word_to_idx, window_size):

        data = []
        data_raw = []
        for ex in list_token:
            for i in range(len(ex) - window_size*2):
                window = ex[i: i + window_size*2+1]
                target = window.pop(window_size)
                context = window
                data.append(([word_to_idx[c] for c in context], word_to_idx[target]))
                data_raw.append((context, target))

        return data, data_raw


    def tokenize(self, corpus, window_size, fraction_size):
    
        """
        prend en entré le corpus brut
        et renvoie une liste de token après nettoyage des données

        """
        if corpus == 'toy_corpus':
            corpus = TOY
            sentences_list = sent_tokenize(corpus)

        
        elif corpus == 'gensim':
            import gensim.downloader as api
            dataset = api.load("20-newsgroups")
            data = [d for d in dataset][:int(fraction_size*len([d_ for d_ in dataset]))]
            print(f'fraction of data taken: {fraction_size}/1')
            
            sentences_list = []
            print("forming sentences..")
            for s in data:
                sentences_list.append(" ".join(s))
        else:
            sentences_list = sent_tokenize(corpus)[:int(fraction_size*len(sent_tokenize(corpus)))]


        print("tokenize the data...")


        punctation = string.punctuation + "``" 

        list_token = []
        vocab = set()
        for s in sentences_list:

            token_words_s = word_tokenize(s)
            for i in range(window_size): 
                token_words_s.append("<end>")
                token_words_s.insert(0, "<bg>")
                
            tok = []

            for w in token_words_s:
                if w not in punctation and not w.isdigit():
                    tok.append(w.lower())
                    vocab.add(w.lower())

            list_token.append(tok)
        
        
        list_token, vocab, word_to_idx, idx_to_word = self.get_data(list_token, vocab)

        training_data, training_data_raw = self.get_training_data(list_token, word_to_idx, window_size)

        return vocab, word_to_idx, idx_to_word, list_token, training_data, training_data_raw 

    def get_data(self, list_token, vocab): 

        '''
        prend en entré le corpus tokénisé et la window

        returns:
         - le vocabulaire : {word: count}
         - word_to_idx : {word: index}
         - idx_to_word : list[word]
        '''
        
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = list(word_to_idx.keys())

        return list_token, vocab, word_to_idx, idx_to_word



#| code-fold: true
import os
import torch

#############################
# Config all parameters here 
#############################

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
RAW_DATA = 'gensim' 
MODEL_ID = RAW_DATA
WINDOW_SIZE = 2 
DISPLAY_LOSS = True 


if RAW_DATA == 'gensim': 

    #general parameters
    DISPLAY_N_BATCH = 2000
    SAVE_N_EPOCH = 1
    BATCH_SIZE = 16
    N_SAVE = 1
    EPOCH = 500

    #preprocess parameters
    WINDOW_SIZE = 2
    FRACTION_SIZE = 1

    # Model parameters
    EMBEDDING_DIM = 10
    LEARNING_RATE = 0.001

    #eval settings
    TEST_WORDS = ["elle", "jeune", "sur", "la"]

if RAW_DATA == 'toy_corpus':

    #toy corpus 
    TOY = "word1 word2 word3 word4. word5 word6 word7 word8 word9. word10 word12 word13 word14. word15 word16 word17 word18"
    #general parameters
    DISPLAY_N_BATCH = 3000
    SAVE_N_EPOCH = 100
    BATCH_SIZE = 16
    N_SAVE = 50
    EPOCH = 300

    #preprocess parameters
    WINDOW_SIZE = 2
    FRACTION_SIZE = 1

    # Model parameters
    EMBEDDING_DIM = 5
    LEARNING_RATE = 0.001

    # eval settings
    TEST_WORDS = ["word2", "word4", "word12"]
    

##### Main trainer and utils settings

PREPROCESS_DATA_DIR = os.path.join(MODEL_ID, 'preprocessed')
PREPROCESS_DATA_PATH = os.path.join(PREPROCESS_DATA_DIR, 'preprocessed_' + MODEL_ID + "_" + str(FRACTION_SIZE) + '.pickle')
MODEL_DIR = os.path.join(MODEL_ID, "cbow")




test = Preprocess(text, WINDOW_SIZE, FRACTION_SIZE)
print(test.exemples_data)





print(test.vocab)
print(test.word_to_idx)
print(test.idx_to_word)

print('la taille du vocabulaire est de: ', len(test.vocab))





print(test.training_data_raw)
print(test.training_exemples)
print("le nombre de couple est de: ", len(test.training_exemples))




print(test.data_x.size())





print(test.data_y.size())









import torch
import torch.nn as nn 
import torch.nn.functional as F 

class CBOWModeler(nn.Module):
  
  def __init__(self, vocab_size, embedding_dim): 
    super(CBOWModeler, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(embedding_dim, vocab_size)

    initrange = 0.5 / embedding_dim
    self.embeddings.weight.data.uniform_(-initrange, initrange)

  def forward(self, input):
    '''
    calcul de la somme des contextes à laquelle on applique la transformation linéaire (tensor : [1, len(vocab)])

    returns: log_softmax appliqué à la transformation 

    '''
    embedding = self.embeddings(input)
    embedding = torch.sum(embedding, dim=1)

    Z_1 = self.linear1(embedding)
    out = F.log_softmax(Z_1, dim=1)

    return out






vocab = test.vocab
word_to_idx = test.word_to_idx
idx_to_word = test.idx_to_word
train_data = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = True)



#| code-fold: true
import torch.nn as nn
import numpy as np 
from numpy.linalg import norm

def nearest_neighbour(X, embeddings, k):
    distance = nn.CosineSimilarity()
    dist = distance(X, embeddings)
    all_idx = np.argsort(-dist)[:k]
    all_cos = dist[all_idx]

    return all_idx, all_cos

def k_n_nn(model, words, word_to_idx, idx_to_word, k):
    model.eval()
    matrix = model.embeddings.weight.data.cpu()

    print(f"process to determine the {k} nearest words of {words}")

    for word in words:
        input = matrix[word_to_idx[word]]

        ranking, _ = nearest_neighbour(input, matrix, k=k+1)

        print(word.ljust(10), ' | ', ', '.join([idx_to_word[i] for i in ranking[1:]]))
    
    return {}



import torch.optim as optim

loss_function = nn.NLLLoss()

total_loss = []

cbow = CBOWModeler(len(vocab), EMBEDDING_DIM).to(DEVICE)

grad = optim.Adam(cbow.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):

    # print(i)
    for batch_idx, (context, target) in enumerate(train_data):

        cbow.train()

        grad.zero_grad()
        context = context.to(DEVICE)
        target = target.to(DEVICE)

        y_pred = cbow(context)
        loss = loss_function(y_pred, target)

        loss.backward()
        grad.step()

        total_loss.append(loss.item())

print(total_loss)



import matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 10))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch")

plt.plot(total_loss)
plt.show()





def predict(context):

  cont = [word_to_idx[word] for word in context]
  tensor = torch.tensor([cont])
  pred = cbow(tensor)

  return idx_to_word[torch.argmax(pred)]

correct = 0
for context, target in test.training_data_raw :
  if target == predict(context):
    correct += 1
print(correct/len(test.training_exemples))
  
test_words = ["tenait", "par", "main", "pépé"]
print(predict(test_words))







```{python}

from cbow import CBOWModeler
from sklearn.manifold import TSNE

mod = torch.load("./model499.pth", map_location=torch.device('cpu'))

idx_to_word = mod["idx_to_word"]
word_to_idx = mod["word_to_idx"]
total_loss = mod["total_loss"]

model = CBOWModeler(len(idx_to_word), 100)
model.load_state_dict(mod["cbow_state_dict"])

embeds = model.embeddings.weight.data.cpu()

tsne = TSNE(n_components = 2).fit_transform(embeds.cpu())
test_words = ['france', 'paris', 'berlin', 'king', 'queen', 'men', 'women', 'he', "she", "car", "trunk", "cake", "sandwich", "cook", ]
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
