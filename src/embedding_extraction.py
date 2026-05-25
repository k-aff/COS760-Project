"""# Embedding Extraction
### South African language sentences from FLORES-200 and Mafand-MT are encoded using the trained models
"""

SA_FAMILY = {
    "zul_Latn": "isiZulu",   "xho_Latn": "isiXhosa",
    "nso_Latn": "Sepedi",    "tsn_Latn": "Setswana",
    "tso_Latn": "Xitsonga",
    "zul": "isiZulu",        "xho": "isiXhosa",
    "tsn": "Setswana",
}

def get_sentence_embeddings(sentences, tokenizer, model, batch_size=32):
    all_embs = []
    model.eval()
    for start in range(0, len(sentences), batch_size):
        batch_sents = sentences[start: start + batch_size]
        enc = tokenizer(
            batch_sents, max_length=MAX_LEN,
            padding=True, truncation=True, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            emb = model.encode(enc["input_ids"], enc["attention_mask"])
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)

def extract_embeddings(dataset_name, lang_dict, model_key, max_per_lang=500):
    save_path = f"{MODEL_DIR}/{model_key}_best"
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model     = SentenceSimilarityModel(save_path).to(DEVICE)

    all_embs, all_labels = [], []
    lang_codes = list(lang_dict.keys())

    for idx, code in enumerate(lang_codes):
        csv_path = f"{DATA_DIR}/{dataset_name}_{code}.csv"
        if not os.path.exists(csv_path):
            print(f"  Skipping {code} — file not found")
            continue
        df    = pd.read_csv(csv_path)
        sents = df["sentence"].dropna().tolist()
        if len(sents) > max_per_lang:
            import random; random.seed(42)
            sents = random.sample(sents, max_per_lang)
        print(f"  {SA_FAMILY.get(code, code):12s} ({code})  {len(sents)} sentences")
        embs = get_sentence_embeddings(sents, tokenizer, model)
        all_embs.append(embs)
        all_labels.extend([idx] * len(embs))

    return np.vstack(all_embs), np.array(all_labels), lang_codes

print("Extracting FLORES embeddings...")
flores_embs_afro, flores_labels_afro, flores_codes = extract_embeddings("flores", FLORES_LANGS, "afroxlmr")
flores_embs_xlmr, flores_labels_xlmr, _            = extract_embeddings("flores", FLORES_LANGS, "xlmr")

print("\nExtracting Mafand embeddings...")
mafand_embs_afro, mafand_labels_afro, mafand_codes = extract_embeddings("mafand", MAFAND_LANGS, "afroxlmr")
mafand_embs_xlmr, mafand_labels_xlmr, _            = extract_embeddings("mafand", MAFAND_LANGS, "xlmr")
print("\nDone.")

"""# Silhouette Score Analysis
### Measures how well sentence embeddings cluster by language. Scores range from -1 to 1 where higher is better
"""

from sklearn.metrics import silhouette_score

def compute_silhouette(embs, labels, name=""):
    if len(set(labels)) < 2:
        print(f"  {name}: only 1 class — cannot compute silhouette")
        return float("nan")
    score = silhouette_score(embs, labels, metric="cosine")
    print(f"  {name:30s}  silhouette = {score:.4f}")
    return score

print("=== Silhouette Scores ===")
print("\nFLORES-200 (5 SA Bantu languages):")
sil_flores_afro = compute_silhouette(flores_embs_afro, flores_labels_afro, "AfroXLMR")
sil_flores_xlmr = compute_silhouette(flores_embs_xlmr, flores_labels_xlmr, "XLM-R")

print("\nMafand-MT (3 SA Bantu languages):")
sil_mafand_afro = compute_silhouette(mafand_embs_afro, mafand_labels_afro, "AfroXLMR")
sil_mafand_xlmr = compute_silhouette(mafand_embs_xlmr, mafand_labels_xlmr, "XLM-R")

print("\nSummary:")
print(f"  {'Dataset':<12} {'AfroXLMR':>12} {'XLM-R':>12} {'Winner':>10}")
print("  " + "-" * 48)
for ds, af, xl in [("FLORES", sil_flores_afro, sil_flores_xlmr),
                   ("Mafand",  sil_mafand_afro, sil_mafand_xlmr)]:
    winner = "AfroXLMR" if af > xl else "XLM-R"
    print(f"  {ds:<12} {af:>12.4f} {xl:>12.4f} {winner:>10}")