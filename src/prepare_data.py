import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Languages we use from SemRel2024:
# kin = Kinyarwanda (Bantu — our training language)
# hau = Hausa (Afro-Asiatic Chadic — cross-family test)
# amh = Amharic Afro-Asiatic Semitic — cross-family test)

SEMREL_LANGS = {
    "kin": "kinyarwanda",
    "hau": "hausa",
    "amh": "amharic",
}

print("Loading SemRel2024")
semrel_data = {}
