import os
from df_constr import load_poetry_ids
import requests


CACHE_DIR = "data/cache"
LOVE_IDS_PATH = os.path.join(CACHE_DIR, "love_ids.json")
HATE_IDS_PATH = os.path.join(CACHE_DIR, "hate_ids.json")

if os.path.exists(LOVE_IDS_PATH) and os.path.exists(HATE_IDS_PATH):
    hate_ids = load_poetry_ids(HATE_IDS_PATH)
    love_ids = load_poetry_ids(LOVE_IDS_PATH)
    print("Loaded LOVE and HATE poetry ids from cache")
else:
    raise FileNotFoundError(
        "LOVE and HATE cache files not found, please run df_constr.py first"
    )

