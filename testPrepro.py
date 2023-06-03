
import torchtext
from torchtext.vocab import build_vocab_from_iterator
import re 
def preprocess(path):
    with open(path, 'r',  encoding = "utf-8") as file:
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
        return sentences_list


def yieldtok():
    for ex in preprocess("./corpus/frcow-lemmatized-100000sent.xml"):
        tokens = ex.lower().split(" ")
        yield tokens

tokens_gen = yieldtok()

vocab = build_vocab_from_iterator(tokens_gen)

print(vocab.get_stoi())
