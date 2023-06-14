import os
import torch

#############################
# Config all parameters here 
#############################

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
RAW_DATA = 'WikiText2' # toy_corpus ou wikitext2 ou frcow 
DATA_PATH = "./corpus/frcow-lemmatized-100000sent.xml" # only for french corpus
MODEL_ID = RAW_DATA
DISPLAY_LOSS = True 



#general parameters
DISPLAY_N_BATCH = 2000
SAVE_N_EPOCH = 1
BATCH_SIZE = 64
N_SAVE = 1
EPOCH = 20
DATA_TYPE = 'train' #train or test or valid

#preprocess parameters
WINDOW_SIZE = 5
MIN_FREQ = 30

# Model parameters
EMBEDDING_DIM = 250
LEARNING_RATE = 0.025

#eval settings
if RAW_DATA == "WikiText103" or "wikitext2":
    TEST_WORDS = ["pain", "paris", "king", "day", "fight"]
if RAW_DATA == "frcow":
    TEST_WORDS = ["paris", "berlin", "guerre", "il", "triste"]

    

##### Main trainer and utils settings

PREPROCESS_DATA_DIR = os.path.join(MODEL_ID, 'preprocessed')
MODEL_DIR = os.path.join(MODEL_ID, "cbow" + str(EMBEDDING_DIM))


    
