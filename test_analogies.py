from cbow import CBOWModeler
import torch
from config import *
from utils import nearest_neighbour

def main():

    checkpoint = torch.load("model4.pth", map_location=torch.device('cpu'))

    idx_to_word = checkpoint["idx_to_word"]
    word_to_idx = checkpoint["word_to_idx"]

    model = CBOWModeler(len(idx_to_word), 100)
    model.load_state_dict(checkpoint["cbow_state_dict"])
    embeds = model.embeddings.weight.data.cpu()

    test_embeddings_on_analogies(idx_to_word, word_to_idx, embeds)

def test_embeddings_on_analogies(idx_to_word:list[str], word_to_idx:dict[str:str], embeds:torch.Tensor):
    def vec(word):
        return embeds[word_to_idx[word]]

    counter_all_examples = 0
    counter_correct_prediction = 0

    # retrieve analogies
    all_analogies_examples = get_analogies_examples("questions-words-fr.lemm.txt")

    ban_list = ["capital-common-countries",
    "capital-world",
    "city-in-state",
    "gram6-nationality-adjective",
    "currency",
    "gram1-adjective-to-adverb",
    "gram2-opposite"]

    for analogy_type, analogies in all_analogies_examples.items():
        if analogy_type in ban_list:
            continue

        #try every valid analogy
        for analogy in analogies:
            if all(item in idx_to_word for item in analogy):

                counter_all_examples+=1

                inp = vec(analogy[1]) - vec(analogy[0]) + vec(analogy[2])

                #get k most similar token...
                emb_ranking_top, euclidean_dis_top = nearest_neighbour(inp, embeds, 13)
                #...and remove tokens that come from the analogy
                emb_ranking_top = [idx_to_word[elt] for elt in emb_ranking_top if idx_to_word[elt] not in analogy[:-1]]

                #if expected result
                if analogy[3] in emb_ranking_top[:10]:
                    counter_correct_prediction+=1

    print(f"accuracy{counter_correct_prediction/counter_all_examples}")


def get_analogies_examples(path:str) -> dict[str:list[str]]:

    with open(path, "r", encoding="utf-8") as file:
        #load whole file
        text = file.read().strip()
        #split into each class of analogies
        analogies = text.split(":")[1:]
        #split each example
        analogies = {analogy.split("\n")[0].strip(): analogy.split("\n")[1:] for analogy in analogies}
        #split each token within each example
        analogies = {analogy_type:[example.split(" ") for example in analogies.get(analogy_type)] for analogy_type in analogies.keys()}
    return analogies

if __name__ == "__main__":
    main()
