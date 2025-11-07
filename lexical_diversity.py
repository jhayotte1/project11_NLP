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
from scipy.optimize import curve_fit


try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


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

def expo(x, a, b, c):
    return a*np.exp(-b*x) + c

def log_func(x, a, b):
    return a * np.log(x + 1e-6) + b #to avoid log(0)

def poly(x, y, deg):
    try:
        poly_coeffs = np.polyfit(x, y, deg)
        poly_fit = np.poly1d(poly_coeffs)
        return poly_fit
    except np.linalg.LinAlgError:
        print(f"Polyfit failed for degree {deg}")
        return None

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

def parametric_fitting(bin_sub, count, title):
    x = np.array([(bin_sub[i] + bin_sub[i+1])/2 for i in range(len(bin_sub)-1)])
    y = np.array(count)

    x_data = x
    y_data = y
    x_smooth = np.linspace(x_data.min(), x_data.max(), 300)

    try:
        params_exp, _ = curve_fit(expo, x, y, p0=(max(y), 1, min(y)))
        y_exp = expo(x, *params_exp)
    except RuntimeError:
        y_exp = np.zeros_like(y)
    
    try:
        params_log, _ = curve_fit(log_func, x, y, p0=(1, 1))
        y_log = log_func(x, *params_log)
    except RuntimeError:
        y_log = np.zeros_like(y)
    
    poly_fit_2 = poly(x, y, 2)
    poly_fit_3 = poly(x, y, 3)
    poly_fit_4 = poly(x, y, 4)
    poly_fit_5 = poly(x, y, 5)
    poly_fit_8 = poly(x, y, 8)
    y_poly_2 = poly_fit_2(x)
    y_poly_3 = poly_fit_3(x)
    y_poly_4 = poly_fit_4(x)
    y_poly_5 = poly_fit_5(x)
    y_poly_8 = poly_fit_8(x)


    r2_exp = r2_score(y, y_exp)
    r2_log = r2_score(y, y_log)
    r2_poly_2 = r2_score(y, y_poly_2)
    r2_poly_3 = r2_score(y, y_poly_3)
    r2_poly_4 = r2_score(y, y_poly_4)
    r2_poly_5 = r2_score(y, y_poly_5)
    r2_poly_8 = r2_score(y, y_poly_8)

    print(f"Parametric fitting for {title}")
    print(f"R² Exponential : {r2_exp:.4f}")
    print(f"R² Logarithmic : {r2_log:.4f}")
    print(f"R² Polynomial(2) : {r2_poly_2:.4f}")
    print(f"R² Polynomial(3) : {r2_poly_3:.4f}")
    print(f"R² Polynomial(4) : {r2_poly_4:.4f}")
    print(f"R² Polynomial(5) : {r2_poly_5:.4f}")
    print(f"R² Polynomial(8) : {r2_poly_8:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(x_data, y_data, label="Observed", color="steelblue")
    plt.plot(x_smooth, expo(x_smooth, *params_exp), color='crimson', linestyle="-", label=f'Exponential fit (R²={r2_exp:.3f})')
    plt.plot(x_smooth, log_func(x_smooth, *params_log), color='seagreen', linestyle="-", label=f'Logarithmic fit (R²={r2_log:.3f})')
    plt.plot(x_smooth, poly_fit_2(x_smooth), color='royalblue', linestyle="-", label=f'Polynomial(2) fit (R²={r2_poly_2:.3f})')
    plt.plot(x_smooth, poly_fit_3(x_smooth), color='mediumpurple', linestyle="-", label=f'Polynomial(3) fit (R²={r2_poly_3:.3f})')
    plt.plot(x_smooth, poly_fit_4(x_smooth), color='darkorange', linestyle="-", label=f'Polynomial(4) fit (R²={r2_poly_4:.3f})')
    plt.plot(x_smooth, poly_fit_5(x_smooth), color='gold', linestyle="-", label=f'Polynomial(5) fit (R²={r2_poly_5:.3f})')
    plt.plot(x_smooth, poly_fit_8(x_smooth), color='black', linestyle="-", label=f'Polynomial(8) fit (R²={r2_poly_8:.3f})')

    plt.title(f"Parametric fitting of LD histogram ({title})")
    plt.xlabel("LD value (bin center)")
    plt.ylabel("Number of lines")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join("img", f"LD_parametric_fitting_{title}.png"))
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

    bin_love = bin_subdivision(love=True)
    count_love = get_number_per_bin_love(df_love_LD)
    bin_hate = bin_subdivision(love=False)
    count_hate = get_number_per_bin_love(df_hate_LD)

    plot_hist(bin_love, count_love, love=True)
    plot_hist(bin_hate, count_hate, love=False)

    parametric_fitting(bin_love, count_love, "love_poem")
    parametric_fitting(bin_hate, count_hate, "hate_poem")

##END OF SECTION
