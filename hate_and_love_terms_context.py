import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from pandas import read_csv
from dataframe_stat import tokenize_text
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from wordcloud import WordCloud


# Collected from: https://wordnik.com/words/hate 
hate_vocab = {
    'hate','abhor', 'abhorrence', 'abhorrent', 'abominable', 'abominate', 'abomination',
    'accursed', 'acrimonious', 'allergy', 'anathema', 'animosity', 'animus',
    'antagonism', 'antipathy', 'aversion', 'bear a grudge', 'bear ill will',
    'bear malice', 'belligerence', 'bete noire', 'bitchy', 'bitter', 'black',
    'blasphemous', 'catty', 'clash', 'clashing', 'cold sweat', 'collision',
    'conflict', 'contemn', 'contemptible', 'contention', 'creeping flesh',
    'damnable', 'deprecate', 'despicable', 'despise', 'despite', 'despiteful',
    'despitefulness', 'detest', 'detestable', 'detestation', 'disapprove',
    'disdain', 'disfavor', 'disgust', 'dislike', 'disrelish', 'distasteful',
    'distressing', 'enmity', 'evil', 'execrable', 'execrate', 'execration',
    'foul', 'friction', 'hateful', 'hatred', 'hold in abomination',
    'hold it against', 'horrid', 'horror', 'hostility', 'ill will',
    'ill-natured', 'infamous', 'loathe', 'loathing', 'malevolence',
    'malevolent', 'malice', 'malign', 'malignity', 'mean', 'mislike',
    'mortal horror', 'nasty', 'nausea', 'not care for', 'obnoxious',
    'odious', 'odium', 'opprobrious', 'owe a grudge', 'peeve', 'pet peeve',
    'phobia', 'quarrelsomeness', 'rancor', 'repellent', 'reprehensible',
    'repugnance', 'repulsion', 'repulsive', 'resent', 'resentful', 'resist',
    'revulsion', 'scorn', 'scurvy', 'shrink from', 'shudder at', 'shuddering',
    'spite', 'spiteful', 'spitefulness', 'trouble', 'unspeakable',
    'utterly detest', 'vicious', 'vile', 'hate'
}

# Collected from: https://wordnik.com/words/love
love_vocab = {
    'love', 'fondness', 'fortitude', 'freak out on', 'frictionlessness', 'friendliness',
    'friendship', 'frigidity', 'fuck', 'generosity', 'get high on', 'girl',
    'giving', 'gloat over', 'go for', 'good vibes', 'good vibrations',
    'good wishes', 'goodwill', 'grace', 'greatheartedness', 'greetings',
    'groove on', 'gust', 'gusto', 'guy', 'happy family', 'harmony', 
    'have deep feelings', 'have designs on', 'have eyes for', 'have it bad',
    'have sex', 'hold dear', 'hon', 'honey', 'honey bunch', 'honey child',
    'hope', 'humanitarianism', 'hump', 'identity', 'idolatry', 'idolize',
    'impotence', 'inamorata', 'inamorato', 'inclination', 'indulge in',
    'infatuation', 'intended', 'intrigue', 'justice', 'kind regards',
    'kindest regards', 'kindness', 'kinship', 'know', 'lamb', 'lambkin',
    'largeheartedness', 'leaning', 'liaison', 'libido', 'light of love',
    'like', 'like-mindedness', 'likes', 'liking', 'love affair', 'love bird',
    'love of mankind', 'loved one', 'lovemaking', 'lover', 'loyalty', 'lust',
    'lust after', 'luxuriate in', 'make out', 'man', 'mania', 'marriage',
    'mate', 'mutuality', 'natural virtues', 'neck', 'neighborlikeness',
    'neighborliness', 'nothing', 'oneness', 'partiality', 'passion', 'peace',
    'peaceableness', 'pet', 'petkins', 'philanthropism', 'philanthropy',
    'piety', 'pleasure', 'potency', 'precious', 'precious heart',
    'predilection', 'prefer', 'preference', 'prize', 'proclivity', 'prudence',
    'rapport', 'rapprochement', 'rapture', 'reciprocity', 'regard', 'regards',
    'rejoice in', 'relationship', 'relish', 'remembrances', 'respects',
    'revel in', 'revere', 'riot in', 'romance', 'savor', 'screw', 'sensuality',
    'sentiment', 'sex drive', 'sexiness', 'sexual instinct', 'sexual urge',
    'sexualism', 'sexuality', 'sharing', 'sisterhood', 'smack the lips',
    'snookums', 'sociability', 'solicitude', 'solidarity', 'sugar', 'suitor',
    'supernatural virtues', 'swain', 'sweet', 'sweetheart', 'sweetie',
    'sweetkins', 'sweets', 'swim in', 'sympathy', 'symphony', 'take',
    'take pleasure in', 'take to', 'tally', 'taste', 'team spirit',
    'temperance', 'tenderness', 'theological virtues', 'thing', 'treasure',
    'truelove', 'turtledove', 'understanding', 'unhostility', 'union',
    'unison', 'unity', 'utilitarianism', 'value', 'venerate', 'voluptuousness',
    'wallow in', 'want', 'warmth', 'weakness', 'welfarism', 'well-affectedness',
    'well-beloved', 'well-disposedness', 'wish', 'wish to goodness',
    'wish very much', 'woman', 'worship', 'would fain do', 'yearning',
    'young man', 'zeal'
}

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 6_000_000

CSV_DIR = "data/csv"
lemmatizer = WordNetLemmatizer()

#Vocab preparation
def prepare_vocab(vocab, nlp):
    singles = set()
    phrases = set()
    max_len = 1
    
    for expr in vocab:
        doc = nlp(expr)
        lems = [t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space)]
        if not lems:
            continue
        if len(lems) == 1:
            singles.add(lems[0])
        else:
            phrases.add(" ".join(lems))
            if len(lems) > max_len:
                max_len = len(lems)
    return singles, phrases, max_len

love_singles_vocab, love_phrases_vocab, love_max_len_vocab = prepare_vocab(love_vocab, nlp)
hate_singles_vocab, hate_phrases_vocab, hate_max_len_vocab = prepare_vocab(hate_vocab, nlp)





def lemmatize_text(text: str) -> list:
    doc = nlp(text)
    lemmed = [token.lemma_.lower() for token in doc if not (token.is_punct or token.is_space)]
    return lemmed


def hate_context_in_text(lemmed_text: list) -> list:
    cont = []
    n=len(lemmed_text)

    for i, word in enumerate(lemmed_text):
        if word in hate_singles_vocab:
            for off in (-2, -1, 1, 2):
                if i + off >= 0 and i + off < len(lemmed_text):
                    cont.append(lemmed_text[i + off])
        for L in range(min(love_max_len_vocab, n-i), 1, -1):
            span = tuple(lemmed_text[i:i+L])
            if span in love_phrases_vocab:
                start, end = i, i+L-1
                for k in (start-2, start-1, end+1, end+2):
                    if k >= 0 and k < n:
                        cont.append(lemmed_text[k])
                break
    cont = set(cont)
    return cont


def love_context_in_text(lemmed_text: list) -> list:
    cont = []
    n = len(lemmed_text)
    
    for i, word in enumerate(lemmed_text):
        if word in love_singles_vocab:
            for off in (-2, -1, 1, 2):
                k = i + off
                if 0 <= k < n:
                    cont.append(lemmed_text[k])
        for L in range(min(love_max_len_vocab, n - i), 1, -1):  
            span = tuple(lemmed_text[i:i+L])
            if span in love_phrases_vocab:
                start, end = i, i+L-1
                for k in (start-2, start-1, end+1, end+2):
                    if 0 <= k < n:
                        cont.append(lemmed_text[k])
                break
    cont = set(cont)
    return cont


def love_cont_freq_df(love_context: set, lemmed_texts: list) -> dict:
    freq = {}
    for word in love_context:
        freq[word] = sum(text.count(word) for text in lemmed_texts)
    return freq

def hate_cont_freq_df(hate_context: set, lemmed_texts: list) -> dict:
    freq = {}
    for word in hate_context:
        freq[word] = sum(text.count(word) for text in lemmed_texts)
    return freq


def plot_wordcloud(freq_dict: dict, title: str):
    wc = WordCloud(width=1000, height=500, background_color='white', colormap='viridis').generate_from_frequencies(freq_dict)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

if __name__ == "__main__":
    love = read_csv(os.path.join(CSV_DIR, "love.csv"))
    hate = read_csv(os.path.join(CSV_DIR, "hate.csv"))
    love_texts,hate_texts = love.iloc[:,4],hate.iloc[:,4]
    
    love_lemmed = [lemmatize_text(text) for text in love_texts]
    hate_lemmed = [lemmatize_text(text) for text in hate_texts]

    love_context = [love_context_in_text(t) for t in love_lemmed]
    hate_context = [hate_context_in_text(t) for t in hate_lemmed]
    #remove duplicates
    love_context = set(word for context in love_context for word in context)
    hate_context = set(word for context in hate_context for word in context)

    love_context_freq_in_love = love_cont_freq_df(love_context, love_lemmed)
    hate_context_freq_in_love = hate_cont_freq_df(hate_context, love_lemmed)

    plot_wordcloud(love_context_freq_in_love, "Love Context in Love Texts")
    plot_wordcloud(hate_context_freq_in_love, "Hate Context in Love Texts")
