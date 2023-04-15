import torch, string
from nltk.stem.wordnet import WordNetLemmatizer
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from config import *
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lem = WordNetLemmatizer()


class Preprocess(Dataset):

    def __init__(self, raw_data, window_size, fraction_size): 

        vocab, word_to_idx, idx_to_word, exemples_data, training_exemples = self.tokenize(raw_data, window_size, fraction_size)

        self.training_exemples = training_exemples
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.window_size = window_size
        self.exemples_data = exemples_data

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
        for ex in list_token:
            for i in range(len(ex) - window_size*2):
                window = ex[i: i + window_size*2+1]
                target = window.pop(window_size)
                context = window
                data.append(([word_to_idx[c] for c in context], word_to_idx[target]))

        return data


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
        elif corpus == 'frcow':
            import re 
            with open(DATA_PATH, 'r',  encoding = "utf-8") as file:
                text = file.read()
                sentences_ = re.split("</s>", text)
                sentences_list = []
                for sentence in sentences_:
                    new_sentence = []
                    for token in sentence.split("\n"):
                        if re.match("<s*", token) or len(token)==0: #empty line or remaining xml tag
                            continue
                        if re.match(".+_.+", token): #le_DET
                            new_sentence.append(token[0:token.rfind("_")])
                    sentences_list.append(" ".join(new_sentence))
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
        

        # print("lemmatize and remove stopwords...")
        # clean_list_token = []
        # for sent in list_token:
        #     clean_list_token.append([lem.lemmatize(w.lower()) for w in sent if w not in punctation and not w.isdigit()])
        #
        #
        # for sent in clean_list_token:
        #     for i in range(window_size):
        #         sent.insert(i, "#")
        #         sent.append("#")
        
        list_token, vocab, word_to_idx, idx_to_word = self.get_data(list_token, vocab)

        # print(clean_list_token)
        training_data = self.get_training_data(list_token, word_to_idx, window_size)



                
        return vocab, word_to_idx, idx_to_word, list_token, training_data 

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

