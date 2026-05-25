"""# Error Analysis
### Sentence pairs where the model predictions differ most from human relatedness scores
"""

def error_analysis(model_key, lang, n=5):
    save_path = f"{MODEL_DIR}/{model_key}_best"
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model     = SentenceSimilarityModel(save_path).to(DEVICE)
    model.eval()

    df_all  = pd.read_csv(f"{DATA_DIR}/semrel_{lang}.csv")
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)
    loader  = DataLoader(SemRelDataset(df_test, tokenizer),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds.extend(model(
                batch["input_ids1"], batch["attention_mask1"],
                batch["input_ids2"], batch["attention_mask2"],
            ).cpu().numpy())

    df_test["pred"]  = preds
    df_test["error"] = (df_test["pred"] - df_test["score"]).abs()

    print(f"\n--- {model_key.upper()} | {EVAL_LANGS[lang]} — top {n} errors ---")
    for _, row in df_test.nlargest(n, "error").iterrows():
        print(f"  Gold={row['score']:.2f}  Pred={row['pred']:.2f}  Err={row['error']:.2f}")
        print(f"    S1: {row['sentence1'][:80]}")
        print(f"    S2: {row['sentence2'][:80]}")
        print()

for lang in EVAL_LANGS:
    error_analysis("afroxlmr", lang)

"""# Final Results Summary"""

print("=" * 65)
print("FINAL RESULTS SUMMARY")
print("=" * 65)

print("\n1. Fine-tuning on Kinyarwanda (dev Spearman ρ)")
for key, rho in ft_results.items():
    print(f"   {key:<15}  dev ρ = {rho:.4f}")

print("\n2. Zero-shot transfer (test Spearman ρ)")
print(f"   {'Language':<12} {'AfroXLMR':>10} {'XLM-R':>10}")
print("   " + "-" * 34)
for lang, lang_name in EVAL_LANGS.items():
    af = zeroshot_results["afroxlmr"][lang]["rho"]
    xl = zeroshot_results["xlmr"][lang]["rho"]
    print(f"   {lang_name:<12} {af:>10.4f} {xl:>10.4f}")

print("\n3. SA language clustering (silhouette, cosine distance)")
print(f"   {'Dataset':<12} {'AfroXLMR':>12} {'XLM-R':>12}")
print("   " + "-" * 38)
print(f"   {'FLORES':<12} {sil_flores_afro:>12.4f} {sil_flores_xlmr:>12.4f}")
print(f"   {'Mafand':<12} {sil_mafand_afro:>12.4f} {sil_mafand_xlmr:>12.4f}")
print("=" * 65)

"""# Visualisation
### PCA plots showing sentence embedding clusters coloured by language for both models and both datasets
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

def plot_pca(embs, labels, lang_codes, name_map, title, ax):
    pca    = PCA(n_components=2, random_state=42)
    proj   = pca.fit_transform(embs)
    colors = cm.tab10(np.linspace(0, 1, len(lang_codes)))
    for idx, (lc, col) in enumerate(zip(lang_codes, colors)):
        mask = labels == idx
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   label=name_map.get(lc, lc), color=col, s=8, alpha=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_pca(flores_embs_afro, flores_labels_afro, flores_codes, SA_FAMILY, "FLORES — AfroXLMR", axes[0,0])
plot_pca(flores_embs_xlmr, flores_labels_xlmr, flores_codes, SA_FAMILY, "FLORES — XLM-R",   axes[0,1])
plot_pca(mafand_embs_afro, mafand_labels_afro, mafand_codes, SA_FAMILY, "Mafand — AfroXLMR", axes[1,0])
plot_pca(mafand_embs_xlmr, mafand_labels_xlmr, mafand_codes, SA_FAMILY, "Mafand — XLM-R",   axes[1,1])
fig.suptitle("Sentence Embedding Clusters by Language (PCA)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("embedding_clusters_pca.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: embedding_clusters_pca.png")