import os
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 20_000_000

import pandas as pd
import matplotlib.pyplot as plt
from spacy.matcher import Matcher
from pandas import read_csv
from dataframe_stat import tokenize_text
from wordcloud import WordCloud
from typing import Iterable, List, Tuple, Set, Dict
from collections import defaultdict, Counter
from tqdm import tqdm


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




CSV_DIR = "data/csv"


#Vocab preparation
def prepare_vocab(vocab_raw: Iterable[str], nlp_model) -> Tuple[Set[str], Set[Tuple[str, ...]], int]:
    prepared = set()
    
    for expr in vocab_raw:
        doc = nlp_model(expr)
        lems = tuple(t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space))
        if not lems:
            continue
        prepared.add(lems)
    return prepared


def build_matcher_from_lemmas(lemmas_patterns: Set[Tuple[str, ...]], nlp_model) -> Matcher:
    m = Matcher(nlp_model.vocab)
    patterns = [[{"LEMMA": lemma} for lemma in lemmas] for lemmas in lemmas_patterns]
    if patterns:
        m.add("TARGET", patterns)
    return m

def iter_doc_chunk(text: str, nlp_model, max_tokens: int = 50_000, overlap: int = 2):
    base_doc = nlp_model(text)
    N = len(base_doc)
    if N==0:
        return
    i = 0
    while i < N:
        j = min(i+max_tokens, N)
        usable_start = i + (overlap if i > 0 else i)
        usable_end = j - (overlap if j < N else j)
        subtext = base_doc[i:j].text
        yield subtext, i, j, usable_start, usable_end
        i = j - overlap

def count_context_with_matcher(texts, matcher, nlp_model):
    cnt = Counter()
    for text in tqdm(texts, desc="Processing books", unit="book"):
        if not isinstance(text, str) or not text:
            continue
        doc = nlp_model(text)  # un livre complet
        n = len(doc)
        matches = matcher(doc)
        for _, start, end in matches:           # end exclusif
            for k in (start-2, start-1, end, end+1):
                if 0 <= k < n:
                    tok = doc[k]
                    if not (tok.is_space or tok.is_punct or tok.is_stop):
                        cnt.update([tok.lemma_.lower()])
    return dict(cnt)


def count_context_with_matcher_chunked(texts, matcher: Matcher, nlp_model, max_tokens: int = 50_000, overlap=2):
    cnt = Counter()
    for text in tqdm(texts, desc="Processing (chunked)", unit="book"):
        if not isinstance(text, str) or not text:
            continue
        for subtext, start_tok, end_tok, usable_start, usable_end in iter_doc_chunk(text, nlp_model, max_tokens, overlap):
            doc = nlp_model(subtext)
            n = len(doc)
            matches = matcher(doc)
        
            for _, start, end in matches:
                g_s = start_tok + start
                g_e = start_tok + end

                if not (usable_start <= g_s and g_e <= usable_end):
                    continue

                for off in (end, end+1, start-1, start-2):
                    if 0 <= off < n:
                        tok=doc[off]
                        if not (tok.is_punct or tok.is_space or tok.is_stop):
                            cnt.update([tok.lemma_.lower()])
    return cnt


def plot_wordcloud(freq_dict: dict, title: str, top_n: int | None = None):
    if top_n is not None and top_n < len(freq_dict) and top_n > 0:
        freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)[:top_n])
    
    wc = WordCloud(width=1200, height=600, background_color='white', colormap='viridis').generate_from_frequencies(freq_dict)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.savefig(os.path.join("img", f"{title.lower().replace(' ', '_')}_wordcloud.png"))
    plt.show()



if __name__ == "__main__":
    love = read_csv(os.path.join(CSV_DIR, "love.csv"))
    hate = read_csv(os.path.join(CSV_DIR, "hate.csv"))
    love_texts,hate_texts = love.iloc[:,4],hate.iloc[:,4]
    
    love_patterns = prepare_vocab(love_vocab, nlp)
    love_matcher = build_matcher_from_lemmas(love_patterns, nlp)

    try:
        love_freq = count_context_with_matcher(love_texts, love_matcher, nlp)
    except Exception as e:
        print("Chunked book")
        love_freq = count_context_with_matcher_chunked(love_texts, love_matcher, nlp, max_tokens=50_000, overlap=2)

    hate_patterns = prepare_vocab(hate_vocab, nlp)
    hate_matcher = build_matcher_from_lemmas(hate_patterns, nlp)

    try:
        hate_freq = count_context_with_matcher(hate_texts, hate_matcher, nlp)
    except Exception as e:
        print("Chunked book")
        hate_freq = count_context_with_matcher_chunked(hate_texts, hate_matcher, nlp, max_tokens=50_000, overlap=2)

    plot_wordcloud(love_freq, "Love Context Word Cloud", top_n = 200)
    plot_wordcloud(hate_freq, "Hate Context Word Cloud", top_n = 200)
