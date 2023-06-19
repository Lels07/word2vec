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

  # def predict(self, input):
  #   '''
  #   returns : le mot avec la log_probabilité la plus grande
  #   '''
  #   context = torch.tensor([[word_to_idx[w.lower()] for w in input]])
  #   probs = self.forward(context)
  #
  #   out = torch.argmax(probs)
  #
  #   return idx_to_word[out.item()]

# def get_word_embs(self, word):
#     word = torch.LongTensor([word_to_idx[word]])
#     return self.embeddings(word)



