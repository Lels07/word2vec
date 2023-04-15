import torch, os, pickle, shutil
import torch.nn as nn 
import torch.optim as optim
from config import *
from preproccess2 import Preprocess
from utils import *
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    write_summary = True
except:
    write_summary = False

from cbow import CBOWModeler

################################################################
#the training code will be place in this file
#
#Check config.py for changing parameters
#
#Model parameters and data are stored in a new dir if not exits
################################################################
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

os.makedirs(MODEL_DIR)


# def read(data):
#     with open(data, "r") as f:
#         a = f.read()
#     return a
# text = read("./wikitext-2-raw/wiki.train.raw")

# def read(text):
#     with open(text, "r") as f:
#         r = f.read()
#
#     return r
#
# corpus = read("../legifrance.gouv.fr.txt")

# Load the data
if not os.path.exists(PREPROCESS_DATA_PATH):
    dataset = Preprocess(RAW_DATA, WINDOW_SIZE, FRACTION_SIZE)

    if not os.path.exists(PREPROCESS_DATA_DIR):
        os.makedirs(PREPROCESS_DATA_DIR)

    print('\ndumping pickle...')
    out_file = open(PREPROCESS_DATA_PATH,'wb')
    pickle.dump(dataset, out_file)
    out_file.close()
    print('pickle dumped\n')

else:
    print('\nloading pickle...')
    infile = open(PREPROCESS_DATA_PATH,'rb')
    dataset = pickle.load(infile)
    infile.close()
    print('pickle loaded\n')

vocab = dataset.vocab
word_to_idx = dataset.word_to_idx
idx_to_word = dataset.idx_to_word



train_data = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
print('len(train_dataset): ', len(dataset))
print('len(train_loader): ', len(train_data))
print('len(vocab): ', len(vocab), '\n')

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
            print(f'Batch: {batch_idx+1}/{len(train_data)}, Loss: {loss.item()}')
            
            k_n_nn(cbow, TEST_WORDS, word_to_idx, idx_to_word, 5) 
    
    if epoch % SAVE_N_EPOCH == 0:
        torch.save({'cbow_state_dict': cbow.state_dict(),
                    'total_loss': total_loss,
                    'word_to_idx': word_to_idx,
                    'idx_to_word': idx_to_word,
                    },'{}/model{}.pth'.format(MODEL_DIR, epoch) )
