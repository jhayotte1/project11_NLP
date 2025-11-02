import os, re
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from spacy.matcher import Matcher
from collections import Counter
from tqdm import tqdm
from typing import Iterable, Tuple, Dict, Set, Optional
from wordcloud import WordCloud


CPU_COUNT = os.cpu_count() or 4
N_PROC = max(1, min(CPU_COUNT // 2, 6))
CHUNK_TOKENS = 40_000
BATCH_SIZE = 6
SAVE_EVERY = 100_000

CSV_DIR = "data/csv"
LOVE_CSV = os.path.join(CSV_DIR, "love.csv")
HATE_CSV =  os.path.join(CSV_DIR, "hate.csv")
TEXT_COLUMN = 4

PARTIAL_DIR = "data/csv/partial_context_count"
OUT_DIR = "data/csv/context_counts"

#spaCy Pipeline
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
nlp.max_length = 20_000_000

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

LOVE_SET = {s.strip().lower() for s in love_vocab if s.strip()}
HATE_SET = {s.strip().lower() for s in hate_vocab if s.strip()}
ALL_SET = LOVE_SET | HATE_SET

SINGLE = {w for w in ALL_SET if " " not in w}
MULTI = {w for w in ALL_SET if " " in w}

AFFECTIVE_VERBS = {
    "love","hate","adore","idolize","like","dislike","detest","despise","loathe","resent","prefer","prize","value","want","wish","worship","revere"
} & SINGLE

NOISY_ANCHORS = {
    "take","know","man","woman","guy","girl","thing","nothing","black"
}

SINGLE = SINGLE - NOISY_ANCHORS

def expand_for_prefilter(tokens: set[str]) -> set[str]:
    extras = set()
    for t in tokens:
        L = len(t)
        if not t.isalpha() or L <= 3:
            continue
        low = t.lower()
        if low.endswith("e"):
            if L>=3: extras.add(low + "d")
        else:
            if L>=3: extras.add(low + "ed")
        if not low.endswith("s") or (L >= 3 and not low.endswith("ss")):
            extras.add(low + "s")
        if L >= 4:
            extras.add(low + "ing")
    return tokens | extras


SINGLE_EXPANDED = (SINGLE - AFFECTIVE_VERBS) | expand_for_prefilter(AFFECTIVE_VERBS)
SURFACE_FORMS = SINGLE_EXPANDED


def build_surface_regex_multi(phrases: set[str]) -> re.Pattern | None:
    parts = []
    for expr in phrases:
        esc = re.escape(expr).replace(r"\ ", r"\s+")
        parts.append(rf"\b{esc}\b")
    return re.compile("(?i)(?:%s)" % "|".join(parts)) if parts else None


SURFACE_RE = build_surface_regex_multi(MULTI)


def should_process(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if any(sf in low for sf in SURFACE_FORMS):
        return True
    return bool(SURFACE_RE and SURFACE_RE.search(text))


def max_pattern_len(lemma_patterns: Set[Tuple[str, ...]]) -> int:
    return max((len(p) for p in lemma_patterns), default=1)


def prepare_vocab(vocab_raw: Iterable[str], nlp_model) -> Set[Tuple[str, ...]]:
    out = set()
    for expr in vocab_raw:
        expr = expr.strip()
        if not expr:
            continue
        doc = nlp_model(expr)
        lems = tuple(t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space))
        if not lems:
            continue
        out.add(lems)
    return out


def build_matcher_from_lemmas(lemmas_patterns: Set[Tuple[str, ...]], nlp_model) -> Matcher:
    m = Matcher(nlp_model.vocab)
    patterns = [[{"LEMMA": lemma} for lemma in lemmas] for lemmas in lemmas_patterns]
    if patterns:
        m.add("TARGET", patterns)
    return m


def iter_sub_texts(text: str, nlp_model, max_tokens: int, overlap: int):
    L = len(text)
    if L >= nlp_model.max_length:
        nlp_model.max_length = L + 1000

    base_doc = nlp_model.make_doc(text)
    N = len(base_doc)
    if N==0:
        return
    i = 0
    while i < N:
        j = min(i+max_tokens, N)
        usable_start = i + (overlap if i > 0 else 0)
        usable_end = j - (overlap if j < N else 0)
        subtext = base_doc[i:j].text
        yield subtext, i, j, usable_start, usable_end
        i = j - overlap if j < N else j


def counter_context_stream(csv_path: str, matcher: Matcher, nlp_model, text_col: int, max_tokens: int, overlap: int, out_partial: Optional[str] = None) -> Counter:
    counter = Counter()
    updates_since_save = 0

    for df in pd.read_csv(csv_path, usecols=[text_col], chunksize=8):
        texts = df.iloc[:, 0].astype(str).tolist()
        for text in texts:
            if not should_process(text):
                continue

            subtexts = list(iter_sub_texts(text, nlp_model, max_tokens, overlap))
            if not subtexts: 
                continue

            for k in range(0, len(subtexts), BATCH_SIZE):
                batch = subtexts[k:k+BATCH_SIZE]
                docs = list(
                    nlp_model.pipe(
                        (st for st, *_ in batch),
                        batch_size = BATCH_SIZE,
                        n_process = N_PROC
                    )
                )
                for doc, (_, start_tok, end_tok, usable_start, usable_end) in zip(docs, batch):
                    n = len(doc)
                    for _, s, e in matcher(doc):
                        g_s = start_tok + s
                        g_e = start_tok + e
                        if not (usable_start <= g_s and g_e <= usable_end):
                            continue

                        for off in range(s-2, s):
                            if 0 <= off < n:
                                tok = doc[off]
                                if not (tok.is_space or tok.is_punct or tok.is_stop):
                                    counter.update([tok.lemma_.lower()])
                                    updates_since_save += 1
                        for off in range(e, e+2):
                            if 0 <= off < n:
                                tok = doc[off]
                                if not (tok.is_space or tok.is_punct or tok.is_stop):
                                    counter.update([tok.lemma_.lower()])
                                    updates_since_save += 1
            if out_partial and updates_since_save >= SAVE_EVERY:
                pd.Series(counter).sort_values(ascending=False).to_csv(out_partial)
                updates_since_save = 0
    return counter


def make_wordcloud(counter, title, save_path, max_words=200):
    freqs = dict(counter)
    wc = WordCloud(
        width=1600, 
        height=1000, 
        background_color='white',
        max_words=max_words,
        colormap="viridis",
        random_state=42
    ).generate_from_frequencies(freqs)
    
    plt.figure(figsize=(14, 9))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=24, pad=20)
    plt.tight_layout()
    
    plt.savefig(os.path.join("img", save_path))
    plt.show()


if __name__ == "__main__":
    print("Preparing patterns...")
    love_patterns = prepare_vocab(love_vocab, nlp)
    hate_patterns = prepare_vocab(hate_vocab, nlp)

    overlap_love = max_pattern_len(love_patterns)
    overlap_hate = max_pattern_len(hate_patterns)

    print("Building matchers...")
    love_matcher = build_matcher_from_lemmas(love_patterns, nlp)
    hate_matcher = build_matcher_from_lemmas(hate_patterns, nlp)

    print("Counting love contexts...")

    print("LOVE-in-LOVE ...")
    love_in_love = counter_context_stream(
        LOVE_CSV, love_matcher, nlp, TEXT_COLUMN, CHUNK_TOKENS, overlap_love, out_partial=os.path.join(PARTIAL_DIR, "love_in_love_partial.csv")
    )
    pd.Series(love_in_love).sort_values(ascending=False).to_csv(os.path.join(OUT_DIR, "love_in_love_counts.csv"))

    print("HATE-in-LOVE ...")
    hate_in_love = counter_context_stream(
        LOVE_CSV, hate_matcher, nlp, TEXT_COLUMN, CHUNK_TOKENS, overlap_hate, out_partial=os.path.join(PARTIAL_DIR, "hate_in_love_partial.csv")
    )

    print("LOVE-in-HATE ...")
    love_in_hate = counter_context_stream(
        HATE_CSV, love_matcher, nlp, TEXT_COLUMN, CHUNK_TOKENS, overlap_hate, out_partial=os.path.join(PARTIAL_DIR, "love_in_hate_partial.csv")
    )
    pd.Series(love_in_hate).sort_values(ascending=False).to_csv(os.path.join(OUT_DIR, "love_in_hate_counts.csv"))

    print("HATE-in-HATE ...")
    hate_in_hate = counter_context_stream(
        HATE_CSV, hate_matcher, nlp, TEXT_COLUMN, CHUNK_TOKENS, overlap_hate, out_partial=os.path.join(PARTIAL_DIR, "hate_in_hate_partial.csv")
    )
    pd.Series(hate_in_hate).sort_values(ascending=False).to_csv(os.path.join(OUT_DIR, "hate_in_hate_counts.csv"))


    print("Generating word clouds...")
    make_wordcloud(love_in_love, "Love context in LOVE dataframe", "love_in_love_context_wordcloud.png", max_words=200)
    make_wordcloud(hate_in_love, "Hate context in LOVE dataframe", "hate_in_love_context_wordcloud.png", max_words=200)
    make_wordcloud(love_in_hate, "Love context in HATE dataframe", "love_in_hate_context_wordcloud.png", max_words=200)
    make_wordcloud(hate_in_hate, "Hate context in HATE dataframe", "hate_in_hate_context_wordcloud.png", max_words=200)