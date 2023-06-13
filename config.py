import os
import torch

#############################
# Config all parameters here 
#############################

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
RAW_DATA = 'WikiText103' # toy_corpus ou wikitext2 ou frcow 
DATA_PATH = "./corpus/frcow-lemmatized-100000sent.xml" # only for french corpus
MODEL_ID = RAW_DATA
WINDOW_SIZE = 2 
DISPLAY_LOSS = True 


if RAW_DATA == 'WikiText103' or RAW_DATA == 'frcow' or RAW_DATA == 'WikiText2': 

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
    # FRACTION_SIZE = 0.01

    # Model parameters
    EMBEDDING_DIM = 250
    LEARNING_RATE = 0.025

    #eval settings
    TEST_WORDS = ["pain", "paris", "king", "day", "fight"]

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
# PREPROCESS_DATA_PATH = os.path.join(PREPROCESS_DATA_DIR, 'preprocessed_' + MODEL_ID + "_" + str(FRACTION_SIZE) + '.pickle')
MODEL_DIR = os.path.join(MODEL_ID, "cbow" + str(EMBEDDING_DIM))


    
