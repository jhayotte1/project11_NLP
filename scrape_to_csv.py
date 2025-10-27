import os, glob
from df_constr import load_poetry_ids
from rdflib import Graph, Namespace
from rdflib.namespace import DCTERMS, RDF
import requests


CACHE_DIR = "data/cache"
LOVE_IDS_PATH = os.path.join(CACHE_DIR, "love_ids.json")
HATE_IDS_PATH = os.path.join(CACHE_DIR, "hate_ids.json")
RDF_BASE_DIR = "gutenberg_rdf/cache/epub"

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

    try :
        g.parse(rdf_path)
    except Exception as e:
        print(f"Error parsing RDF for id {id}: {e}")
        return None
    
    for file_node in g.objects(None, DCTERMS.hasFormat):
        url = str(file_node)
        if url.endswith(f"{id}.txt.utf-8"):
            return url
    return None

def download_text(id, url, save_dir = "data/texts"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{id}.txt")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Saved ebook {id}.txt")
        return save_path
    except Exception as e:
        print(f"Error downloading ebook {id}: {e}")
        return None

##Test
#id = love_ids[0]
#print(id)
#url = get_url_for_download(id)
#print(url)
#saved_path = download_text(id, url)
#print(saved_path)
