import glob, re
import json, os
from rdflib import Graph, Namespace
from rdflib.namespace import DCTERMS, RDF
from tqdm import tqdm

CACHE_DIR = "data/cache"
POETRY_IDS_PATH = os.path.join(CACHE_DIR, "poetry_ids.json")
LOVE_IDS_PATH = os.path.join(CACHE_DIR, "love_ids.json")
HATE_IDS_PATH = os.path.join(CACHE_DIR, "hate_ids.json")

RDF_PATH = "gutenberg_rdf/cache/epub/*/*.rdf"
RDF_BASE_DIR = "gutenberg_rdf/cache/epub"

PGTERMS = Namespace("http://www.gutenberg.org/2009/pgterms/")

LOVE = re.compile(r"\b(love|loves|loved|loving)\b", re.I)
HATE = re.compile(
    r"\b("
    r"hate|hates|hated|hating|hatred|hateful|hater|haters|"
    r"despise|despises|despised|despising|"
    r"loathe|loathes|loathed|loathing|"
    r"detest|detests|detested|detesting|detestation|"
    r"abhor|abhors|abhorred|abhorring|abhorrence|"
    r"abominate|abominates|abominated|abominating|abomination|"
    r"disdain|disdains|disdained|disdaining|disdainful|"
    r"scorn|scorns|scorned|scorning|scornful|"
    r"contempt|contemptuous|"
    r"disgust|disgusts|disgusted|disgusting|"
    r"revulsion|repulsion|repulsive|repugnance|repugnant|"
    r"execration|"
    r"animosity|animosities|"
    r"hostility|hostilities|hostile|"
    r"enmity|"
    r"antipathy|"
    r"antagonism|antagonistic|"
    r"animus|"
    r"rancor|rancorous|rankling|"
    r"malice|malicious|malevolence|malevolent|malignity|"
    r"spite|spiteful|"
    r"venom|venomous|"
    r"revenge|revengeful|"
    r"aversion|"
    r"dislike|dislikes|disliked|disliking|"
    r"odium|"
    r"anathema|"
    r"resentment|resentful|"
    r"grievance|grievances|"
    r"gripe|gripes|griped|griping|"
    r"irritant|irritants|"
    r"nuisance|nuisances|"
    r"bugbear|bugbears|"
    r"bother|bothers|bothered|bothering|bothersome|"
    r"trouble|troubles|troubled|troubling|troublesome|"
    r"horror|horrors|horrific|"
    r"pain|painful|"
    r"frost|frosty|"
    r"objection|objections|objectionable"
    r")\b",
    re.I
)


# From all the ebook available, we want only the english ones
def is_english_ebook(g: Graph):
    for node in g.objects(None, DCTERMS.language):
        for val in g.objects(node, RDF.value):
            v = str(val).lower()
            if v == "en" or v.startswith("en-"):
                return True
    return False


# From all the ebook available, we want only the Poetry ones
def is_poetry(g: Graph):
    for node in g.objects(None, PGTERMS.bookshelf):
        for val in g.objects(node, RDF.value):
            if "poetry" in str(val).lower():
                return True
    return False


# Used to retrieve only the Poetry about Love or Hate
def extract_fields(g: Graph):
    fields = {"title": set(), "subject": set(), "bookshelves": set(), "summary":set()}

    for o in g.objects(None, DCTERMS.title):
        fields["title"].add(str(o))

    for node in g.objects(None, DCTERMS.subject):
        for val in g.objects(node, RDF.value):
            fields["subject"].add(str(val))

    for node in g.objects(None, PGTERMS.bookshelf):
        for val in g.objects(node, RDF.value):
            fields["bookshelves"].add(str(val))
    
    for summary in g.objects(None, PGTERMS.marc520):
        fields["summary"].add(str(summary))
    return fields


# Test if a file contains Love or Hate topic
def has_love_hate(g: Graph):  # return a tuple (has_love: bool, has_hate: bool)
    fields = extract_fields(g)

    bag = " ".join(
        list(fields["title"]) + list(fields["subject"]) + list(fields["bookshelves"]) + list(fields["summary"])
    ).lower()

    has_love = bool(LOVE.search(bag))
    has_hate = bool(HATE.search(bag))

    return (has_love, has_hate)


# Retrieve the list of ids of Poetry ebook
def get_poetry_ids_from_rawdata():

    poetry_ids = []
    love_ids = []
    hate_ids = []

    rdf_files = glob.glob(RDF_PATH)
    print("RDF files found : ", len(rdf_files))

    for rdf_file in tqdm(rdf_files):
        g = Graph()

        try:
            g.parse(rdf_file)
        except Exception:
            continue  # skip corrupted files

        if not is_english_ebook(g):
            continue
        if not is_poetry(g):
            continue

        ebook_id = rdf_file.split("/")[-2]
        poetry_ids.append(ebook_id)

    return poetry_ids, love_ids, hate_ids


# Save the list of Poetry ids in a JSON file, so that we don't have to look at the 76000 files each time while we just want to focus on the poetry
def save_poetry_ids(poetry_ids, path=POETRY_IDS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    uniq_sorted = sorted(set(poetry_ids), key=lambda x: int(x))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(uniq_sorted, f, ensure_ascii=False, indent=2)
    print(f"Saved{len(uniq_sorted)} poetry_ids -> {path}")


# Return the path of a corresponding file having it's id
def ids_to_path(ids):
    return [os.path.join(RDF_BASE_DIR, i, f"pg{i}.rdf") for i in ids]


# Load the JSON file containing the list of Poetry ids
def load_poetry_ids(path=POETRY_IDS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Get the Poetry ids from the JSON file, if it does not exist, create it and return it
def get_or_build_poetry_ids():
    if os.path.exists(POETRY_IDS_PATH):
        ids = load_poetry_ids()
        print(f"Loaded {len(ids)} poetry ids from cache")
        return ids
    else:
        poetry_ids, _, _ = get_poetry_ids_from_rawdata()
        save_poetry_ids(poetry_ids)
        return poetry_ids


# Among all the Poetry ebook, retrieve the ones about Love or Hate
def get_love_hate_from_poetry_ids(ids):
    love, hate = [], []
    for rdf_path in tqdm(ids_to_path(ids), total=len(ids)):
        if not os.path.exists(rdf_path):
            continue
        g = Graph()
        try:
            g.parse(rdf_path)
        except Exception:
            continue

        has_love, has_hate = has_love_hate(g)
        ebook_id = os.path.basename(os.path.dirname(rdf_path))
        if has_love:
            love.append(ebook_id)
        if has_hate:
            hate.append(ebook_id)
    return love, hate


# poetry_ids, love_ids, hate_ids = get_poetry_ids_from_rawdata()
# save_poetry_ids(poetry_ids)

poetry_ids = get_or_build_poetry_ids()
love_ids, hate_ids = get_love_hate_from_poetry_ids(poetry_ids)

# Save the results to a json file to prepare for scraping
# Balance dataset by limiting to the first 250

limit = 250
save_poetry_ids(love_ids[:limit], LOVE_IDS_PATH)
save_poetry_ids(hate_ids[:limit], HATE_IDS_PATH)

print("Total number of poetry books found : ", len(poetry_ids))
print("Total number of Love poetry books found : ", len(love_ids))
print("Total number of Hate poetry books found : ", len(hate_ids))
print("Example of poetry ids : ", poetry_ids[:20])
