"""# Zero-shot Evaluation
### Both models are tested on Hausa and Amharic with no additional training
"""

EVAL_LANGS = {
    "hau": "Hausa",
    "amh": "Amharic",
}

LANG_FAMILY = {
    "kin": "Niger-Congo / Bantu (Central)",
    "hau": "Afro-Asiatic / Chadic",
    "amh": "Afro-Asiatic / Semitic",
}

def evaluate_zeroshot(model_key: str):
    print(f"\nZero-shot evaluation — {model_key.upper()}")
    save_path = f"{MODEL_DIR}/{model_key}_best"
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model     = SentenceSimilarityModel(save_path).to(DEVICE)
    model.eval()

    results = {}
    for lang, lang_name in EVAL_LANGS.items():
        df_all  = pd.read_csv(f"{DATA_DIR}/semrel_{lang}.csv")
        df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

        loader = DataLoader(
            SemRelDataset(df_test, tokenizer),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {lang}", leave=False):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                p = model(
                    batch["input_ids1"], batch["attention_mask1"],
                    batch["input_ids2"], batch["attention_mask2"],
                )
                preds.extend(p.cpu().numpy())
                labels.extend(batch["score"].cpu().numpy())

        rho, pval = spearmanr(preds, labels)
        if np.isnan(rho):
            rho = 0.0
        results[lang] = {"rho": rho, "pval": pval, "n": len(df_test)}
        print(f"  {lang_name:12s} ({lang})  ρ={rho:.4f}  p={pval:.4f}  n={len(df_test)}")

    return results

zeroshot_results = {}
for key in MODELS:
    zeroshot_results[key] = evaluate_zeroshot(key)

print("\n=== Zero-shot Spearman ρ Results ===")
print(f"{'Language':<14} {'Family':<30} {'AfroXLMR':>10} {'XLM-R':>10}")
print("-" * 66)
for lang, lang_name in EVAL_LANGS.items():
    afro_rho = zeroshot_results["afroxlmr"][lang]["rho"]
    xlmr_rho = zeroshot_results["xlmr"][lang]["rho"]
    print(f"{lang_name:<14} {LANG_FAMILY[lang]:<30} {afro_rho:>10.4f} {xlmr_rho:>10.4f}")
print()
print(f"Training language: Kinyarwanda  [{LANG_FAMILY['kin']}]")
print("Both eval languages are cross-family — lower ρ expected.")