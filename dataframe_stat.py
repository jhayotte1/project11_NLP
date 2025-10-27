import nltk
import os
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tok = RegexpTokenizer(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOPWORDS = set(stopwords.words("english"))

TEXT_PATH = "data/texts"

def id_to_text_path(id, type: str):
    return(f"data/texts/{type}/{id}")

def read_text(file_path: str):
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()
    return text

def tokenize_text(txt):
    tokens = tok.tokenize(txt)
    tokens = [t.lower() for t in tokens if len(t) > 1 and t.lower() not in STOPWORDS]
    return tokens

def tot_tokens_and_vocab_size(type: str):
    tot_vocab = []
    tot_tokens = []
    for file in os.listdir(os.path.join(TEXT_PATH, type)):
        file_path = os.path.join(TEXT_PATH, type, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenize_text(text)
        tot_tokens.extend(tokens)
        tot_vocab.update(tokens)
    return len(tot_vocab), len(tot_vocab)

def number_ebooks(type: str):
    with open(f"{type}_ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)
    return len(ids)

love_tok, love_vocab = tot_tokens_and_vocab_size("love")
hate_tok, hate_vocab = tot_tokens_and_vocab_size("hate")

print("Number of Love related Poetry ebooks : ", number_ebooks("love"))
print("Number of Hate related Poetry ebooks: ", number_ebooks("hate"))
print("Number tokens in Love dataframe: ", love_tok)
print("Vocabulary size of Love dataframe: ", love_vocab)
print("Number tokens in Hate dataframe: ", hate_tok)
print("Vocabulary size in Hate dataframe: ", love_vocab)