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
def distances_between_words(poem_list, vocab):
    distances = []
     
    for poem in tqdm(poem_list, desc="Going through poems"):
        distance = 0
        words = word_tokenize(poem.lower())

        for word in words:
            if word in vocab:
                distances.append(distance) 
                distance = 0
            else:
                distance += 1
    
    return distances

if __name__ == "__main__": 
    love = read_csv(os.path.join(CSV_DIR, "love.csv"))
    hate = read_csv(os.path.join(CSV_DIR, "hate.csv"))
    love_texts,hate_texts = love.iloc[:,4],hate.iloc[:,4]
    
    love_in_love_distances = distances_between_words(love_texts,love_vocab)
    hate_in_love_distances = distances_between_words(love_texts,hate_vocab)
    love_in_hate_distances = distances_between_words(hate_texts,love_vocab)
    hate_in_hate_distances = distances_between_words(hate_texts,hate_vocab)

    # print(love_in_love_distances[:20])
    # print(hate_in_love_distances[:20])
    # print(love_in_hate_distances[:20])
    # print(hate_in_hate_distances[:20])


    data = pd.DataFrame({
        'Distances': love_in_love_distances + hate_in_love_distances + love_in_hate_distances + hate_in_hate_distances,
        'Type': ['Love in Love']*len(love_in_love_distances) + \
                ['Hate in Love']*len(hate_in_love_distances) + \
                ['Love in Hate']*len(love_in_hate_distances) + \
                ['Hate in Hate']*len(hate_in_hate_distances)
    })
    
    # We here only include the the 90% interval of these values
    # This is so that our graph does not include outliers that make it unreadable and does not contribute with any meaningful statistics
    max_distance = data['Distances'].quantile(0.95)
    data_filtered = data[data['Distances'] <= max_distance]
                                           
    sns.histplot(data=data_filtered, discrete=True, x='Distances', hue='Type', bins=10, multiple='dodge', palette=['r','g','b','y'])
    plt.xlabel("Distances")
    plt.ylabel("Occurences")
    plt.title("Distances between Love and Hate related words in Love and Hate poems")
    plt.savefig(os.path.join("img", "love_and_hate_distances.png"))

