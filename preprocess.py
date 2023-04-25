import torch
from CBOW import CBOWModeler
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
import statistics 
from utils import k_n_nn

tokenizer = get_tokenizer("basic_english")

data_iter = WikiText2(split="train")

vocab = build_vocab_from_iterator(map(tokenizer, data_iter), min_freq=40, specials=["<unk>"] )
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))

# print(text_pipeline("this is a test"))


# print(vocab.get_stoi())
# print("taille: ", len(vocab.get_stoi()))

def collate(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)
        
        if len(text_tokens_ids) < 2 * 2 + 1:
            continue

        for idx in range(len(text_tokens_ids) - 2 * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + 2 * 2 + 1)]
            output = token_id_sequence.pop(2)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


collate = collate
dataloader = DataLoader(data_iter, batch_size=64, collate_fn=partial(collate, text_pipeline=text_pipeline))

loss_function = nn.NLLLoss()

total_loss = []

device = torch.device("mps")
print(len(vocab))


cbow = CBOWModeler(len(vocab), 70).to(device)

grad = optim.Adam(cbow.parameters(), lr=0.001)

for epoch in range(30):
    print(epoch)

    for batch_idx, (context, target) in enumerate(dataloader):

        cbow.train()

        grad.zero_grad()

        context = context.to(device)
        target = target.to(device)


        y_pred = cbow(context)
        loss = loss_function(y_pred, target)

        loss.backward()
        grad.step()

        total_loss.append(loss.item())

    k_n_nn(cbow, ["he", "work", "is", "actor", "war", "paris"], vocab.get_stoi(), vocab.get_itos(), 5)

    print(statistics.mean(total_loss))
