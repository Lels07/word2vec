import torch.nn as nn
import numpy as np 
from numpy.linalg import norm

def nearest_neighbour(X, embeddings, k):
    distance = nn.CosineSimilarity()
    dist = distance(X, embeddings)
    all_idx = np.argsort(-dist)[:k]
    all_cos = dist[all_idx]

    return all_idx, all_cos

def k_n_nn(model, words, word_to_idx, idx_to_word, k):
    model.eval()
    matrix = model.embeddings.weight.data.cpu()

    print(f"process to determine the {k} nearest words of {words}")

    for word in words:
        input = matrix[word_to_idx[word]]

        ranking, _ = nearest_neighbour(input, matrix, k=k+1)

        print(word.ljust(10), ' | ', ', '.join([idx_to_word[i] for i in ranking[1:]]))
    
    return {}

    
# def nearest_neighbour(input, embedding, k):
#     cosine_sim = np.dot(input, embedding) / (norm(input)*norm(embedding)) 
#     rank = np.argsort(-cosine_sim)
#
#     distance_rank = cosine_sim[rank[:k]]
#     target_rank = rank[:k]
#     cosine_sim_k = cosine_sim[target_rank]
#
#     return target_rank, cosine_sim_k

# def k_nearest_neighbour(model, words, word_to_idx, idx_to_word, k):
#
#     model.eval()
#     matrix = model.embeddings.weight.data.cpu()
#
#     words_dict = {}
#
#     print(f"process to determine the {k} nearest words of {words}")
#     for word in words:
#         input = matrix[word_to_idx[word], :]
#
#         ranking, _ = nearest_neighbour(matrix, input, k)
#         print(word.ljust(10), ' | ', ', '.join([idx_to_word[i] for i in ranking[1:]]))
#
#     return words_dict

# def nearest_word(inp, emb, top = 5, debug = False):
#     euclidean_dis = np.linalg.norm(inp - emb, axis = 1)    
#     emb_ranking = np.argsort(euclidean_dis)
#     emb_ranking_distances = euclidean_dis[emb_ranking[:top]]
#
#     emb_ranking_top = emb_ranking[:top]
#     euclidean_dis_top = euclidean_dis[emb_ranking_top]
#
#     if debug:
#         print('euclidean_dis: ', euclidean_dis)
#         print('emb_ranking: ', emb_ranking)
#         print(f'top {top} embeddings are: {emb_ranking[:top]} with respective distances\n {euclidean_dis_top}')
#
#     return emb_ranking_top, euclidean_dis_top
def print_nearest_words(model, test_words, word_to_ix, ix_to_word, top = 5):
    
    model.eval()
    emb_matrix = model.embeddings.weight.data.cpu()
    
    nearest_words_dict = {}

    print('==============================================')
    for t_w in test_words:
        
        inp_emb = emb_matrix[word_to_ix[t_w], :]  

        emb_ranking_top, _ = nearest_word(inp_emb, emb_matrix, top = top+1)
        print(t_w.ljust(10), ' | ', ', '.join([ix_to_word[i] for i in emb_ranking_top[1:]]))

    return nearest_words_dict
