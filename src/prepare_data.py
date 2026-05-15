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

for code, name in SEMREL_LANGS.items():
    print(f" Loading {name} ({code}) …")
    try:
        ds = load_dataset("SemRel/SemRel2024", name, trust_remote_code=True)
        rows = []
        for split_name in ["train", "dev", "test"]:
            if split_name not in ds:
                continue
            for item in ds[split_name]:
                rows.append({
                    "sentence1": item["sentence1"],
                    "sentence2": item["sentence2"],
                    "score": float(item["label"]),
                    "split": split_name,
                    "lang": code,
                })
        df = pd.DataFrame(rows)
        df.to_csv(f"{DATA_DIR}/semrel_{code}.csv", index=False)
        semrel_data[code] = df
        print(f"-> {len(df)} pairs saved")
    except Exception as e:
        print(f"Unable to load {name}: {e}")
        # print("Falling back to manual download instructions below.")