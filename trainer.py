from numpy import mean
import torch 
import torch.nn as nn 
import torch.optim as optim
from utils import *
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    write_summary = True
except:
    write_summary = False

from CBOW import CBOWModeler
from Prepro import Preprocess

WINDOW_SIZE = 5 
BATCH_SIZE = 64
MIN_FREQ = 50
EMBEDDING_DIM = 100
DEVICE = torch.device("mps")
LEARNING_RATE = 0.005
EPOCH = 20
DISPLAY_LOSS = True
DISPLAY_N_BATCH = 1000
TEST_WORDS = ["she", "fight", "car", "work", "france"]

dataset = Preprocess("WikiText2", "train", WINDOW_SIZE, MIN_FREQ, BATCH_SIZE)

train_data = dataset.train_data
vocab = dataset.vocab

word_to_idx = vocab.get_stoi()
idx_to_word = vocab.get_itos()

loss_function = nn.NLLLoss()

total_loss = []

cbow = CBOWModeler(len(vocab), EMBEDDING_DIM).to(DEVICE)

grad = optim.Adam(cbow.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, EPOCH))   

    # print(i)
    for batch_idx, (context, target) in enumerate(train_data):
        print('batch# ' + str(batch_idx+1).zfill(len(str(len(train_data)))) + '/' + str(len(train_data)), end = '\r')

        cbow.train()

        grad.zero_grad()
        context = context.to(DEVICE)
        target = target.to(DEVICE)

        y_pred = cbow(context)
        loss = loss_function(y_pred, target)

        loss.backward()
        grad.step()

        total_loss.append(loss.item())

        if batch_idx % DISPLAY_N_BATCH == 0 and DISPLAY_LOSS:
            print(f'Batch: {batch_idx+1}/{len(train_data)}, Loss: {mean(total_loss)}')
            
            k_n_nn(cbow, TEST_WORDS, word_to_idx, idx_to_word, 5) 

    print(mean(total_loss))

