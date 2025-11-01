import os
from pandas import read_csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import *

CSV_DIR = "data/csv"

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

stemmer = PorterStemmer()

hate_vocab = [stemmer.stem(word) for word in hate_vocab if ' ' not in word]  
love_vocab = [stemmer.stem(word) for word in love_vocab if ' ' not in word]  


if __name__ == "__main__":
    
    love = read_csv(os.path.join(CSV_DIR, "love.csv"))
    hate = read_csv(os.path.join(CSV_DIR, "hate.csv"))
    love_texts,hate_texts = love.iloc[:,4],hate.iloc[:,4]

    
    # Summing over all texts in Hate and Love and the occurences of Hate and Love words
    # We normalize by the length of the respective set to make the size of the related words list less relevant 
    love_in_love = sum([sum([text.count(w) for w in love_vocab])/len(love_vocab) for text in love_texts])
    hate_in_love = sum([sum([text.count(w) for w in hate_vocab])/len(hate_vocab) for text in love_texts])
    
    love_in_hate = sum([sum([text.count(w) for w in love_vocab])/len(love_vocab) for text in hate_texts])
    hate_in_hate = sum([sum([text.count(w) for w in hate_vocab])/len(hate_vocab) for text in hate_texts])

    data = pd.DataFrame({
        'Category': ['Love Poems', 'Love Poems', 'Hate Poems', 'Hate Poems'],
        'Emotion': ['Love Words', 'Hate Words', 'Love Words', 'Hate Words'],
        'Proportion': [love_in_love, hate_in_love, love_in_hate, hate_in_hate]
    })

    sns.barplot(data=data, x='Category', y='Proportion', hue='Emotion')
    plt.title("Hate and Love based words in Hate and Love poems normalized")
    plt.savefig(os.path.join("img", "love_and_hate_words.png"))
