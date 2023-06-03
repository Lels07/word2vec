from functools import partial
import torch
import re
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText103, WikiText2
from torchtext.data.utils import get_tokenizer

class Preprocess:

    def __init__(self, dataset_name: str,
                 dataset_type: tuple[str]|str,
                 window_size: int,
                 min_freq_vocab: int,
                 batch_size, 
                 tokenizer_name="basic_english"):

        self.window_size = window_size

        train_data, vocab = self.get_dataloader_vocab(dataset_name, dataset_type, batch_size, min_freq_vocab, tokenizer_name)

        self.train_data = train_data
        self.vocab = vocab

    def get_tokenizer(self, tokenizer_name):

        tokenizer = get_tokenizer(tokenizer_name)

        return tokenizer

    def get_data(self, data_name: str, data_type: tuple[str]|str):
        '''
        get data from torchtext language modelling dataset  
        '''

        if data_name == "WikiText2":
            data = WikiText2(split=data_type)

        elif data_name == "WikiText103":
            data = WikiText103(split=data_type)

        data = to_map_style_dataset(data)

        return data

    def vocab_builder(self, data, tokenizer, min_freq_vocab):
        """build vocab from data"""

        vocab = build_vocab_from_iterator(map(tokenizer, data), specials=["<unk>"], min_freq=min_freq_vocab)
        vocab.set_default_index(vocab["<unk>"])
        
        return vocab

    def get_batch(self, batch, text_pipeline):

        '''batch to be use with pytorch dataloader'''

        
        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = text_pipeline(text)
            
            if len(text_tokens_ids) < self.window_size * 2 + 1:
                continue

            for idx in range(len(text_tokens_ids) - self.window_size * 2):
                token_id_sequence = text_tokens_ids[idx : (idx + self.window_size * 2 + 1)]
                output = token_id_sequence.pop(self.window_size)
                input_ = token_id_sequence
                batch_input.append(input_)
                batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output
    
    def get_dataloader_vocab(self, dataset_name, dataset_type, batch_size, min_freq_vocab, tokenizer_name, shuffle=False):

        data = self.get_data(dataset_name, dataset_type)
        tokenizer = self.get_tokenizer(tokenizer_name)

        vocab = self.vocab_builder(data, tokenizer, min_freq_vocab) 

        
        text_pipeline = lambda x: vocab(tokenizer(x))

        collate_function = self.get_batch

        train_data = DataLoader(data, batch_size=batch_size, 
                                shuffle=shuffle, 
                                collate_fn=partial(collate_function, text_pipeline=text_pipeline))

        return train_data, vocab


class Frpreprocess:
    def __init__(self, path, min_freq, window_size, batch_size):

        train_data, vocab = self.get_dataloader_vocab(path,batch_size, min_freq, shuffle=False)
        self.train_data = train_data
        self.vocab = vocab
        self.window_size = window_size
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.path = path

    def get_data(self, path):
        with open(path, 'r',  encoding = "utf-8") as file:

            sentences_list = []
            sent = []
            for line in file:
                if re.match("<s.+", line):
                    continue
                elif re.match("</s>", line):
                    sentences_list.append(" ".join(sent))
                    sent = []
                else:
                    sent.append(line[0:line.rfind("_")])

            sentences_list = to_map_style_dataset(sentences_list)
            return sentences_list

    def yieldtok(self, path):
        for ex in self.get_data(path):
            tokens = ex.lower().split(" ")
            yield tokens

    def vocab_builder(self, path, data, min_freq):
        vocab = build_vocab_from_iterator(self.yieldtok(path), specials=["<unk>"],min_freq=min_freq)
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def get_batch(self, batch, text_pipeline):

        '''batch to be use with pytorch dataloader'''

        
        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = text_pipeline(text)
            
            if len(text_tokens_ids) < self.window_size * 2 + 1:
                continue

            for idx in range(len(text_tokens_ids) - self.window_size * 2):
                token_id_sequence = text_tokens_ids[idx : (idx + self.window_size * 2 + 1)]
                output = token_id_sequence.pop(self.window_size)
                input_ = token_id_sequence
                batch_input.append(input_)
                batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output
    
    def get_dataloader_vocab(self, path, batch_size, min_freq_vocab, shuffle=True):

        data = self.get_data(path)

        vocab = self.vocab_builder(path, data, min_freq_vocab) 
        tokenizer = get_tokenizer(None)

        
        text_pipeline = lambda x: vocab(tokenizer(x))

        collate_function = self.get_batch

        train_data = DataLoader(data, batch_size=batch_size, 
                                shuffle=shuffle, 
                                collate_fn=partial(collate_function, text_pipeline=text_pipeline))

        return train_data, vocab


