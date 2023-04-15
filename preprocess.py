import torch, string
from nltk.stem.wordnet import WordNetLemmatizer
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from config import *
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lem = WordNetLemmatizer()

text = "ceci est un test 04290, ; ; , fancy et encore d'autre test pour voir. on va mettre des phrase\n et autre. Voila qui va mieux"


class Preprocess(Dataset):

    def __init__(self, raw_data, window_size, fraction_size): 

        vocab, word_to_idx, idx_to_word, training_exemples, training_exemples_tuples = self.tokenize(raw_data, window_size, fraction_size)

        self.training_exemples = training_exemples
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.window_size = window_size
        self.training_exemples_tuples = training_exemples_tuples

        self.data_x = torch.tensor([exemples[0] for exemples in training_exemples_tuples], dtype=torch.long)
        self.data_y = torch.tensor([exemples[1] for exemples in training_exemples_tuples], dtype=torch.long)


    def __len__(self):
        
        return len(self.data_y)
       

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def get_training_data(self, data, word_to_idx, window_size):
        training_data = []
        # vocab_indices = list(range(len(word_to_idx)))
        
        #pour toutes les phrases
        print('preparing training data (x, y)...')
        for sentence in data:
            indices = [word_to_idx[word] for word in sentence]
            
            #pour tout mot compris comme target
            for center_word_pos in range(len(indices)):
                if not indices[center_word_pos] == 0:
                
                #pour toutes les positions de notre fenetre
                    for w in range(-window_size, window_size+1):                
                        context_word_pos = center_word_pos + w

                    #on s'assure que l'on ne dépasse pas la phrase
                        if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                            continue
                    
                        context_word_idx = indices[context_word_pos]
                        center_word_idx  = indices[center_word_pos]
                    
                        if center_word_idx == context_word_idx:
                            continue

                        training_data.append([context_word_idx, center_word_idx])
        training_data_tuple = []
        x = [0 for i in range(window_size*2)]
        y = training_data[0][1]
        for i in range(len(training_data)):
            if training_data[i][1] == y:
                x.append(training_data[i][0])
                x.pop(0)
            else:
                training_data_tuple.append((x, y))
                x = [0 for i in range(window_size*2)]
                y = training_data[i][1]
                x.append(training_data[i][0])
                x.pop(0)

        return training_data, training_data_tuple


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
            dataset = api.load("text8")
            data = [d for d in dataset][:int(fraction_size*len([d_ for d_ in dataset]))]
            print(f'fraction of data taken: {fraction_size}/1')
            
            sentences_list = []
            print("forming sentences..")
            for s in data:
                sentences_list.append(" ".join(s))
        else:
            sentences_list = sent_tokenize(corpus)


        print("tokenize the data...")


        stop_words = set(stopwords.words("english"))
        punctation = [s for s in string.punctuation.replace("#","")]
        digit = string.digits
        for digi in digit:
            punctation.append(digi)

        list_token = [word_tokenize(sent) for sent in sentences_list]
        

        print("lemmatize and remove stopwords...")
        clean_list_token = []
        for sent in list_token:
            clean_list_token.append([lem.lemmatize(w.lower()) for w in sent if w not in punctation and not w.isdigit()])


        for sent in clean_list_token:
            for i in range(window_size):
                sent.insert(i, "#")
                sent.append("#")
        
        clean_list_token, vocab, word_to_idx, idx_to_word = self.get_data(clean_list_token)

        # print(clean_list_token)
        training_data, training_data_tuple = self.get_training_data(clean_list_token, word_to_idx, window_size)



                
        return vocab, word_to_idx, idx_to_word, training_data, training_data_tuple

    def get_data(self, list_token): 

        '''
        prend en entré le corpus tokénisé et la window

        returns:
         - le vocabulaire : {word: count}
         - word_to_idx : {word: index}
         - idx_to_word : list[word]
        '''
        
        vocab = {}
        word_to_idx = {}

        for token_words in list_token: #pour toute les phrases tokeniser en mot
            for word in token_words: #pour tout les mots
                if word not in vocab:
                    vocab[word] = 0
                    word_to_idx[word] = len(word_to_idx)
                vocab[word] += 1
        idx_to_word = list(word_to_idx.keys())

        return list_token, vocab, word_to_idx, idx_to_word



