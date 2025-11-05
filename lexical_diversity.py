import os
import json
import fasttext
import fasttext.util
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np
import requests
from tqdm import tqdm
from nltk import pos_tag, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')


MODEL_BIN = "cc.en.300.bin"
DATA_PATH_CSV = "data/csv"
DATA_PATH_LD = "data/lexical_diversity"
TEXT_DIR = "data/texts"


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

def title_to_vector(title):
    return ft.get_sentence_vector(title)

def sim_to_love_vocab(title_vector):
    max_sim = -1.0
    title_arr = np.asarray(title_vector, dtype=np.float32)
    t_norm = np.linalg.norm(title_arr)
    if t_norm == 0:
        return 0.0
    for elt in love_vocab:
        voc_vector = ft.get_sentence_vector(elt)
        voc_arr = np.asarray(voc_vector, dtype=np.float32)
        denom = t_norm * np.linalg.norm(voc_arr)
        if denom == 0:
            sim = 0.0
        else:
            sim = float(np.dot(title_arr, voc_arr) / denom)
        if sim > max_sim:
            max_sim = sim
    return max_sim

def sim_to_hate_vocab(title_vector):
    max_sim = -1.0
    title_arr = np.asarray(title_vector, dtype=np.float32)
    t_norm = np.linalg.norm(title_arr)
    if t_norm == 0:
        return 0.0
    for elt in hate_vocab:
        voc_vector = ft.get_sentence_vector(elt)
        voc_arr = np.asarray(voc_vector, dtype=np.float32)
        denom = t_norm * np.linalg.norm(voc_arr)
        if denom == 0:
            sim = 0.0
        else:
            sim = float(np.dot(title_arr, voc_arr) / denom)
        if sim > max_sim:
            max_sim = sim
    return max_sim

def get_max_sim_to_vocab(df: pd.DataFrame):
    max_val = df["sim_to_vocab"].max()
    return max_val

def get_rand_poem_maxsim_vocab(df: pd.DataFrame):
    max_val = df["sim_to_vocab"].max()
    rows = df[df["sim_to_vocab"] == max_val]
    poem_id = rows.iloc[:, 0].sample().iloc[0]
    return poem_id


def poem_lexical_diversity(id, love: bool):
    START_STRING = "*** START OF THE PROJECT GUTENBERG EBOOK"
    END_STRING = "*** END OF THE PROJECT GUTENBERG EBOOK"

    poem_LD = []

    txt_path = os.path.join(TEXT_DIR, f"{id}.txt")
    if not os.path.exists(txt_path):
        return None
    with open(os.path.join(TEXT_DIR, f"{id}.txt"), "r") as poem:
        text = ""
        for line in poem:
            if line.startswith(START_STRING):
                break
        i = 1
        for line in tqdm(poem):
            if line.startswith(END_STRING):
                break
            tokens = nltk.word_tokenize(line)
            tagged = nltk.pos_tag(tokens)
            adj_or_adv = 0
            vb = 0
            for tag in tagged:
                if tag[1].startswith('VB'):
                    vb += 1
                if tag[1].startswith('JJ') or tag[1].startswith('RB'):
                    adj_or_adv += 1
            if vb==0:
                poem_LD.append((i, -1))
            else:
                poem_LD.append((i, adj_or_adv/vb))
            i += 1
        
    return poem_LD
            

def save_list_to_csv(li, love: bool):
    os.makedirs(DATA_PATH_LD, exist_ok=True)
    if love:
        csv_path= os.path.join(DATA_PATH_LD, "LD_love_poem.csv")
    else:
        csv_path = os.path.join(DATA_PATH_LD, "LD_hate_poem.csv")
    df = pd.DataFrame(li, columns=["line_index", "LD_value"])
    df.to_csv(csv_path, index=False)
    return None


def plot_LD(data: str, love: bool):
    df = pd.read_csv(os.path.join(DATA_PATH_LD, data))
    plt.figure(figsize=(10, 5))
    plt.scatter(df["line_index"], df["LD_value"], color="steelblue", alpha=0.7, s=5)
    plt.title("Lexical Diversity (ADJ+ADV / VERB) per line")
    plt.xlabel("Line index")
    plt.ylabel("LD value")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if love:
        plt.savefig(os.path.join("img", "LD_love_graph.png"))
    else:
        plt.savefig(os.path.join("img", "LD_hate_graph.png"))
    plt.show()


def bin_subdivision(love: bool):
    if love:
        df = pd.read_csv(os.path.join(DATA_PATH_LD, "LD_love_poem.csv"))
    else:
        df = pd.read_csv(os.path.join(DATA_PATH_LD, "LD_hate_poem.csv"))
    ld_max = df["LD_value"].max()
    ld_min = df["LD_value"][df["LD_value"] != -1].min()
    bin = (ld_max - ld_min)/10
    return [ld_min + k*bin for k in range(11)]

def get_number_per_bin_love(df):
    bin_sub = bin_subdivision(True)
    count = []
    for k in range(len(bin_sub) - 1):
        x, y = bin_sub[k], bin_sub[k+1]
        mask = (df["LD_value"] >= x) & (df["LD_value"] < y)
        count.append(mask.sum())
    return count


def plot_hist(bin_sub, count, love):
    midpoints = [(bin_sub[i] + bin_sub[i+1])/2 for i in range(len(bin_sub) - 1)]
    plt.bar(midpoints, count, width=(bin_sub[1] - bin_sub[0]), edgecolor="black", color="steelblue", alpha = 0.7)
    plt.xlabel("LD value")
    plt.ylabel("Number of lines")
    if love:
        plt.title("Histogram of LD Love poem : number of line for 10 subdivisions")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(os.path.join("img", "LD_love_subdiv_histogram.png"))
    else:
        plt.title("Histogram of LD Hate poem : number of line for 10 subdivisions")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(os.path.join("img", "LD_hate_subdiv_histogram.png"))

    plt.show()



if __name__ == "__main__":

##Uncomment this section to generate the lexical diversity of each poem
    
    if not os.path.exists(MODEL_BIN):
        print("Downloading FastText model...")
        fasttext.util.download_model('en', if_exists='ignore')
        print("Download complete.")
    else:
        print("FastText model already exists.")

    ft = fasttext.load_model(MODEL_BIN)

    df_love = pd.read_csv(os.path.join(DATA_PATH_CSV, "love.csv"), usecols=[0,1])
    df_hate = pd.read_csv(os.path.join(DATA_PATH_CSV, "hate.csv"), usecols=[0,1])

    df_love["vectors"] = df_love.iloc[:, 1].apply(title_to_vector)
    df_hate["vectors"] = df_hate.iloc[:, 1].apply(title_to_vector)

    df_love["sim_to_vocab"] = df_love["vectors"].apply(sim_to_love_vocab)
    # apply sim_to_hate_vocab to hate vectors (was mistakenly using df_love)
    df_hate["sim_to_vocab"] = df_hate["vectors"].apply(sim_to_hate_vocab)

    max_love_sim_to_vocab = get_max_sim_to_vocab(df_love)
    max_hate_sim_to_vocab = get_max_sim_to_vocab(df_hate)
    print(max_love_sim_to_vocab)
    print(max_hate_sim_to_vocab)

    rand_hate_id = get_rand_poem_maxsim_vocab(df_hate)
    rand_love_id = get_rand_poem_maxsim_vocab(df_love)
    print(rand_hate_id)
    print(rand_love_id)

    hate_poem_LD = poem_lexical_diversity(rand_hate_id, love=False)
    love_poem_LD = poem_lexical_diversity(rand_love_id, love=True)

    save_list_to_csv(love_poem_LD, love=True)
    save_list_to_csv(hate_poem_LD, love=False)

    plot_LD("LD_love_poem.csv", love=True)
    plot_LD("LD_hate_poem.csv", love=False)

    df_love_LD = pd.read_csv(os.path.join(DATA_PATH_LD, "LD_love_poem.csv"))
    df_hate_LD = pd.read_csv(os.path.join(DATA_PATH_LD, "LD_hate_poem.csv"))

    plot_hist(bin_subdivision(love=True), get_number_per_bin_love(df_love_LD), love=True)
    plot_hist(bin_subdivision(love=False), get_number_per_bin_love(df_hate_LD), love=False)

##END OF SECTION
