import os
import csv
import sys
from df_constr import load_poetry_ids
from rdflib import Graph
from rdflib.namespace import DCTERMS
import requests
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

CACHE_DIR = "data/cache"
LOVE_IDS_PATH = os.path.join(CACHE_DIR, "love_ids.json")
HATE_IDS_PATH = os.path.join(CACHE_DIR, "hate_ids.json")
RDF_BASE_DIR = "gutenberg_rdf/cache/epub"
CSV_DIR = "data/csv"


if os.path.exists(LOVE_IDS_PATH) and os.path.exists(HATE_IDS_PATH):
    hate_ids = load_poetry_ids(HATE_IDS_PATH)
    love_ids = load_poetry_ids(LOVE_IDS_PATH)
    print("Loaded LOVE and HATE poetry ids from cache")
else:
    raise FileNotFoundError(
        "LOVE and HATE cache files not found, please run df_constr.py first"
    )


def id_to_path(id):
    return os.path.join(RDF_BASE_DIR, str(id), f"pg{id}.rdf")


def get_url_for_download(id):
    rdf_path = id_to_path(id)

    if not os.path.exists(rdf_path):
        print(f"Error: this id ({id}) does not have a corresponding file")
        return None
    g = Graph()

    try:
        g.parse(rdf_path)
    except Exception as e:
        print(f"Error parsing RDF for id {id}: {e}")
        return None

    for file_node in g.objects(None, DCTERMS.hasFormat):
        url = str(file_node)
        if url.endswith(f"{id}.txt.utf-8"):
            return url
    return None


def download_text(id, url, save_dir="data/texts"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{id}.txt")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        #print(f"Saved ebook {id}.txt")
        return save_path
    except Exception as e:
        print(f"Error downloading ebook {id}: {e}")
        return None


def clean_ebook(id: int, ebook_dir: str = "data/texts") -> (str, str, str, str):
    TITLE_STRING = "Title: "
    AUTHOR_STRING = "Author: "
    CONTRIBUTOR_STRING = "Contributor: "
    TRANSLATOR_STRING = "Translator: "
    CREATOR_STRING = "Creator: "
    ILLUSTRATOR_STRING = "Illustrator: "
    RELEASE_STRING = "Release date: "
    COMPILER_STRING = "Compiler: "
    PUBLISHER_STRING = "Publisher: "
    DUBIOUS_STRING = "Dubious author: "

    START_STRING = "*** START OF THE PROJECT GUTENBERG EBOOK"
    END_STRING = "*** END OF THE PROJECT GUTENBERG EBOOK"
    EDITOR_STRING = "Editor: "
    
    txt_path = os.path.join(ebook_dir, f"{id}.txt")
    if not os.path.exists(txt_path):
        return None

    with open(os.path.join(ebook_dir, f"{id}.txt"), "r") as ebook:
        title = ""
        author = ""
        release = ""
        text = ""

        for line in ebook:
            if line.startswith(TITLE_STRING):
                title = line[7:].strip("\n")
                break

        for line in ebook:
            if line.startswith(AUTHOR_STRING):
                author = line[8:].strip("\n")
                break
            elif line.startswith(CONTRIBUTOR_STRING):
                author = line[13:].strip("\n")
                break
            elif line.startswith(TRANSLATOR_STRING):
                author = line[12:].strip("\n")
                break
            elif line.startswith(EDITOR_STRING):
                author = line[8:].strip("\n")
                break
            elif line.startswith(CREATOR_STRING):
                author = line[9:].strip("\n")
                break
            elif line.startswith(COMPILER_STRING):
                author = line[10:].strip("\n")
                break
            elif line.startswith(ILLUSTRATOR_STRING):
                author = line[13:].strip("\n")
                break
            elif line.startswith(PUBLISHER_STRING):
                author = line[11:].strip("\n")
                break
            elif line.startswith(DUBIOUS_STRING):
                author = line[16:].strip("\n")
                break

        for line in ebook:
            if line.startswith(RELEASE_STRING):
                release = list(line[14:])
                release = "".join(release[: release.index("[") - 1]).strip("\n")
                break

        for line in ebook:
            if line.startswith(START_STRING):
                break

        for line in ebook:
            if line.startswith(END_STRING):
                break
            text += " " + line.strip("\n\r")

        return title, author, release, text


def add_to_csv(
    id: int,
    tart: tuple,
    csv_file: str,
    ebook_dir: str = "data/texts",
    csv_dir: str = CSV_DIR,
):
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_file)

    with open(csv_path, "a", newline='') as csvfile:
        write = csv.writer(csvfile)
        write.writerow((id, *tart))



def load_existing_ids(csv_file: str, csv_dir:str = CSV_DIR) -> list:
    skip = []
    with open(CSV_DIR + f"/{csv_file}", "r", newline='') as csvfile:
        read = csv.reader(csvfile)
        for row in read:
            skip.append(row[0])
    return skip

if __name__ == "__main__":
    generate = True
    skip = []
    if os.path.exists(CSV_DIR + "/love.csv"):
        if not (input("CSV file for love found, do you want to regenerate it? Y/N: ").lower().strip() == "y"):
            generate = False
        skip = load_existing_ids("love.csv")  
    
    if generate:
        print("Downloading and parsing love texts...")
        for id in tqdm(love_ids):
            if id not in skip:
                url = get_url_for_download(id)
                saved_path = download_text(id,url)
                cleaned = clean_ebook(id)
                if cleaned is not None:
                    add_to_csv(id,clean_ebook(id),"love.csv")
    
    generate = True
    skip = []
    if os.path.exists(CSV_DIR + "/hate.csv"):
        if not (input("CSV file for hate found, do you want to regenerate it? Y/N: ").lower().strip() == "y"):
            generate = False
        skip = load_existing_ids("hate.csv")  
    
    if generate:
        print("Downloading and parsing hate texts...")
        for id in tqdm(hate_ids):
            if id not in skip:
                url = get_url_for_download(id)
                saved_path = download_text(id,url)
                cleaned = clean_ebook(id)
                if cleaned is not None:
                    add_to_csv(id,clean_ebook(id),"hate.csv")


##Test
# id = love_ids[0]
# print(id)
# url = get_url_for_download(id)
# print(url)
# saved_path = download_text(id, url)
# print(saved_path)
# add_to_csv(id, clean_ebook(id), "love.csv")
