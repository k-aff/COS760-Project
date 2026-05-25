!pip install "datasets<3.0.0"

# import os
# from google.colab import userdata
# from huggingface_hub import login

# try:
#     hf_token = userdata.get('HF_TOKEN')

#     # Karabo: Makes things easier and faster with Hugging Face
#     login(hf_token)
#     print("Successfully logged in to the Hugging Face Hub!")
# except Exception as e:
#     print(f"Login failed. Make sure 'HF_TOKEN' is set in your Secrets tab and notebook access is toggled on: {e}")

# For local dev
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

try:
    hf_token = os.environ["HF_TOKEN"]
    login(hf_token)
    print("Successfully logged in to the Hugging Face Hub!")
except KeyError:
    print("HF_TOKEN not found. Add it to your .env file.")

import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. SemRel2024 - labelled relatedness pairs

# Languages from SemRel2024:
# kin = Kinyarwanda (Bantu-our training language)
# hau = Hausa (Afro-Asiatic Chadic-cross-family test)
# amh = Amharic (Afro-Asiatic Semitic-cross-family test)

SEMREL_LANGS = {
    "kin": "Kinyarwanda",
    "hau": "Hausa",
    "amh": "Amharic",
}

print("Loading SemRel2024")
semrel_data = {}

for code, name in SEMREL_LANGS.items():
    print(f"Loading {name}({code})")
    try:
        ds = load_dataset("SemRel/SemRel2024", code)
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
        print(f"{len(df)} pairs saved")
    except Exception as e:
        print(f"Unable to load {name}: {e}")

# 2. FLORES-200 - unlabelled South African sentences
from datasets import load_dataset
from google.colab import userdata
from huggingface_hub import login

# Make sure we're logged in
login(os.environ["HF_TOKEN"])

FLORES_LANGS = {
    "zul_Latn": "isiZulu",
    "xho_Latn": "isiXhosa",
    "nso_Latn": "Sepedi",
    "tsn_Latn": "Setswana",
    "tso_Latn": "Xitsonga",
}

print("Loading FLORES+ from HuggingFace")
flores_data = {}

for code, name in FLORES_LANGS.items():
    print(f"Loading {name} ({code})")
    try:
        ds = load_dataset("openlanguagedata/flores_plus", code)
        rows = []
        for split_name in ["dev", "devtest"]:
            if split_name not in ds:
                continue
            for item in ds[split_name]:
                rows.append({
                    "sentence": item.get("text", item.get("sentence", "")),
                    "split": split_name,
                    "lang": code,
                    "lang_name": name,
                    "domain": item.get("domain", "general"),
                })
        df = pd.DataFrame(rows)
        df.to_csv(f"{DATA_DIR}/flores_{code}.csv", index=False)
        flores_data[code] = df
        print(f"  {len(df)} sentences saved")
    except Exception as e:
        print(f"  ERROR: {e}")

# 3. Mafand-MT Dataset - unlabelled translation pairs
# Available Mafand languages: zul, xho and tsn

MAFAND_LANGS = {
    "zul": "zul",
    "xho": "xho",
    "tsn": "tsn",
}

print("\nLoading Mafand-MT")
mafand_data = {}

for code, name in MAFAND_LANGS.items():
    print(f"Loading {name}({code})")
    try:
        #Mafand pairs are always English->target language
        ds = load_dataset("masakhane/mafand", f"en-{code}")
        rows = []
        for split_name in ["train", "validation", "test"]:
            if split_name not in ds:
                continue
            for item in ds[split_name]:
                translation = item.get("translation", {})
                target_sent = translation.get(code, "")
                if target_sent:
                    rows.append({
                        "sentence": target_sent,
                        "en_sentence": translation.get("en", ""),
                        "split": split_name,
                        "lang": code,
                        "lang_name": name,
                    })
        df = pd.DataFrame(rows)

        #Let's use sample max 5000 per language to keep things manageable
        if len(df) > 5000:
            df = df.sample(5000, random_state=42).reset_index(drop=True)
        df.to_csv(f"{DATA_DIR}/mafand_{code}.csv", index=False)
        mafand_data[code] = df
        print(f"{len(df)} sentences saved")
    except Exception as e:
        print(f"Error: {e}")

print("\nDataset Summary")
print("\nSemRel2024:")
for code, df in semrel_data.items():
    train_n = len(df[df["split"] == "train"])
    dev_n = len(df[df["split"] == "dev"])
    test_n = len(df[df["split"] == "test"])
    print(f"  {SEMREL_LANGS[code]:15s} ({code})  train={train_n:4d}  dev={dev_n:4d}  test={test_n:4d}")

print("\nFLORES-200:")
for code, df in flores_data.items():
    print(f"  {FLORES_LANGS[code]:12s} ({code:10s})  sentences={len(df):4d}")

print("\nMafand-MT:")
for code, df in mafand_data.items():
    print(f"  {MAFAND_LANGS[code]:12s} ({code})  sentences={len(df):5d}")

print("\nData preparation complete. Files saved to data/")
