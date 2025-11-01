from hate_and_love_terms_proportions import love_vocab,hate_vocab, CSV_DIR
import os
from pandas import read_csv
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('punkt')
def distances_between_differing_words(poem_list, vocab1, vocab2):
    distances = []
     
    for poem in tqdm(poem_list, desc="Going through poems"):
        distance = 0
        words = word_tokenize(poem.lower())
        
        vocabs = [vocab1,vocab2] 
        switch = 0
        for word in words:
            if word in vocabs[switch]:
                distances.append(distance) 
                distance = 0
                switch ^= 1
            else:
                distance += 1
    
    return distances

if __name__ == "__main__":
    love = read_csv(os.path.join(CSV_DIR, "love.csv"))
    hate = read_csv(os.path.join(CSV_DIR, "hate.csv"))
    love_texts,hate_texts = love.iloc[:,4],hate.iloc[:,4]

    htl_in_love_distances = distances_between_differing_words(love_texts,hate_vocab, love_vocab)
    htl_in_hate_distances = distances_between_differing_words(hate_texts,hate_vocab, love_vocab)
    
    data = pd.DataFrame({
        'Distances': htl_in_love_distances + htl_in_hate_distances,
        'Type': ['Love Poems']*len(htl_in_love_distances) + \
                ['Hate Poems']*len(htl_in_hate_distances) 
    })
    
    # We here only include the the 90% interval of these values
    # This is so that our graph does not include outliers that make it unreadable and does not contribute with any meaningful statistics
    max_distance = data['Distances'].quantile(0.95)
    data_filtered = data[data['Distances'] <= max_distance]
                                           
    sns.histplot(data=data_filtered, x='Distances', hue='Type', bins=10, multiple='dodge', palette=['r','g'])
    plt.xlabel("Distances")
    plt.ylabel("Occurences")
    plt.title("Alternating Love and Hate related words ")
    plt.savefig(os.path.join("img", "love_and_hate_distances_differing.png"))

