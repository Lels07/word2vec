from numpy import mean
import torch 
import os, shutil
import torch.nn as nn 
import torch.optim as optim
from utils import *
from config import *
from torch.optim.lr_scheduler import LambdaLR
# try:
#     from torch.utils.tensorboard.writer import SummaryWriter
#     write_summary = True
# except:
#     write_summary = False


from cbow import CBOWModeler
from Prepro import Frpreprocess, Preprocess

MODEL_DIR = os.path.join(MODEL_ID, "cbow" + str(EMBEDDING_DIM))

if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

os.makedirs(MODEL_DIR)

if RAW_DATA == "frcow":
    dataset = Frpreprocess(DATA_PATH, MIN_FREQ, WINDOW_SIZE, BATCH_SIZE)
else:
    dataset = Preprocess(RAW_DATA, DATA_TYPE, WINDOW_SIZE, MIN_FREQ, BATCH_SIZE)


train_data = dataset.train_data
vocab = dataset.vocab

word_to_idx = vocab.get_stoi()
idx_to_word = vocab.get_itos()

loss_function = nn.NLLLoss()

total_loss = []

cbow = CBOWModeler(len(vocab), EMBEDDING_DIM).to(DEVICE)

grad = optim.Adam(cbow.parameters(), lr=LEARNING_RATE)

# lr_lambda = lambda epoch: (EPOCH - epoch) / EPOCH
# lr_scheduler = LambdaLR(grad, lr_lambda=lr_lambda, verbose=True)

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

        if batch_idx % SAVE_N_EPOCH == 0:
            torch.save({'cbow_state_dict': cbow.state_dict(),
                        'total_loss': total_loss,
                        'word_to_idx': word_to_idx,
                        'idx_to_word': idx_to_word,
                        },'{}/model{}.pth'.format(MODEL_DIR, epoch) )
    # lr_scheduler.step()

    print(mean(total_loss))
