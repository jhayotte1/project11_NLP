from hate_and_love_terms_proportions import love_vocab,hate_vocab, CSV_DIR
import os
from pandas import read_csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import fasttext
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt


MODEL_PATH = os.path.join("data/models/fasttext.bin")

nltk.download('stopwords')

def top_frequency(poem_list, n=100):
    stop_words = set(stopwords.words('english'))
    all_words = []

    for poem in tqdm(poem_list, desc="Frequency analysis"):
        words = nltk.word_tokenize(str(poem).lower())
        filtered = [w for w in words if w.isalpha() and w not in stop_words]
        all_words.extend(filtered)

    return [word for word,_ in Counter(all_words).most_common(n)]

def cosine_sim(A,B): 
    return (np.dot(A,B))/(np.linalg.norm(A)*np.linalg.norm(B))

def average_semantic_similarity(text_words, vocab_words, model):
    sims = []

    for w1 in tqdm(text_words, desc="Iterating through words"):
        for w2 in vocab_words:
            sims.append(cosine_sim(model.get_word_vector(w1),model.get_word_vector(w2)))

    return np.mean(sims)

def matching_strings(text_words: set, vocab_words: set):
    return fuzz.ratio(text_words, vocab_words)

if __name__ == "__main__":
    love = read_csv(os.path.join(CSV_DIR, "love.csv"))
    hate = read_csv(os.path.join(CSV_DIR, "hate.csv"))
    love_texts,hate_texts = love.iloc[:,4],hate.iloc[:,4]
    
    if os.path.exists(MODEL_PATH):
            model = fasttext.load_model(MODEL_PATH)
    else:
        with open("temp.txt","w") as temp:
            for poem in love_texts:
                temp.write('\n')
                temp.write(poem)
            for poem in hate_texts:
                temp.write('\n')
                temp.write(poem)

        model = fasttext.train_unsupervised('temp.txt', model='skipgram')
        model.save_model(MODEL_PATH)
        os.remove("temp.txt")

    top_love: set = set(top_frequency(love_texts))
    top_hate: set = set(top_frequency(hate_texts))

    comparisons = ['Love & Love Vocab', 'Hate & Love Vocab', 'Hate & Hate Vocab', 'Love & Hate Vocab']
    top_lists = [top_love, top_hate, top_hate, top_love]
    vocab_lists = [love_vocab, love_vocab, hate_vocab, hate_vocab]

    df = pd.DataFrame({
        'Comparison': comparisons,
        'Semantic Similarity': [average_semantic_similarity(top, vocab, model) 
                               for top, vocab in zip(top_lists, vocab_lists)],
        'Fuzzy Matching': [matching_strings(top, vocab) 
                          for top, vocab in zip(top_lists, vocab_lists)]
    })
    
    ig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, 
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')

    plt.savefig(os.path.join("img/", "frequent_tokens_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(df)

