# COS760 Group 20 - Cross-lingual Semantic Relatedness for African Languages

This project investigates whether NLP models can transfer knowledge about semantic relatedness across African languages, with a focus on Bantu language families and South African languages.

## Research Question

Do African languages from the same family share enough structure to transfer semantic relatedness knowledge between them?

We fine-tune AfroXLMR and XLM-R on Kinyarwanda relatedness data and test zero-shot transfer to Hausa and Amharic, then evaluate how well the models cluster South African Bantu languages they have never seen with relatedness labels.

## How to Run

The script `src/complete_notebook.py` runs the full pipeline and works on both Google Colab and a local machine — no changes needed either way. It auto-detects its environment and handles paths and login accordingly.

---

### Option A: Google Colab (recommended, free GPU)

**Step 1: Get the file**

Go to the GitHub repo, click on `src/complete_notebook.py`, then click the download icon (or right-click Raw → Save As) to download the file.

**Step 2: Open Google Colab and upload the file**

Go to [colab.research.google.com](https://colab.research.google.com) and open a new notebook. In the left sidebar, click the **folder icon** to open the Files panel, then click the **upload icon** and upload `complete_notebook.py`.

**Step 3: Enable GPU**

```
Runtime > Change runtime type > T4 GPU → Save
```

**Step 4: Set up your Hugging Face token**

Create a free account at https://huggingface.co and get a read token at https://huggingface.co/settings/tokens.

In Colab, click the **key icon** in the left sidebar, then:
- Click **Add new secret**
- Name: `HF_TOKEN`
- Value: paste your token
- Toggle **Notebook access ON**

**Step 5: Run**

In a code cell, run:

```python
!python complete_notebook.py
```

The script installs its own dependencies, logs in to Hugging Face, downloads all datasets, fine-tunes both models, and produces all results. Fine-tuning takes ~15 minutes on a T4 GPU.

> **Note:** Colab wipes local files when the session ends. To keep your model checkpoints across sessions, mount Google Drive before running:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> ```
> Then change `DATA_DIR` and `MODEL_DIR` at the top of the script to point to your Drive.

---

### Option B: Local Machine

**Step 1: Clone the repo**

```bash
git clone https://github.com/Mulisa-Musehane/COS760-Project.git
cd COS760-Project
```

**Step 2: Install dependencies**

```bash
pip install -r src/requirements.txt
```

**Step 3: Set up your Hugging Face token**

Create a free account at https://huggingface.co and get a read token at https://huggingface.co/settings/tokens.

Create a `.env` file in the project root:

```
HF_TOKEN=your_token_here
```

**Step 4: Run**

```bash
python src/complete_notebook.py
```

Datasets and model checkpoints are saved to `data/` and `models/` in the project root. Fine-tuning takes ~1 hour on CPU or ~15 minutes with a GPU.

---

### What the pipeline does

| Step | What it does | Time (GPU) |
|---|---|---|
| 1 | Download and save all datasets to `data/` | ~1 min |
| 2 | Fine-tune AfroXLMR and XLM-R on Kinyarwanda | ~15 min |
| 3 | Zero-shot evaluation on Hausa and Amharic (Spearman ρ) | ~1 min |
| 4 | Extract SA language embeddings from FLORES and Mafand | ~1 min |
| 5 | Silhouette scores, PCA plots, error analysis | ~30s |

## Project Structure

```
COS760-Project/
├── README.md
├── src/
│   ├── complete_notebook.py             # Full pipeline in a single script
│   ├── prepare_data.py                  # Load and save SemRel2024, FLORES-200, Mafand-MT
│   ├── fine_tune.py                     # Fine-tune AfroXLMR and XLM-R on Kinyarwanda
│   ├── zero_shot.py                     # Zero-shot evaluation on Hausa and Amharic
│   ├── embedding_extraction.py          # Extract SA language embeddings
│   ├── error_analysis_silhoutte_results.py  # Silhouette scores, PCA plots, error analysis
│   └── requirements.txt
├── data/                                # Created when the pipeline runs
│   ├── semrel_kin.csv                   # Kinyarwanda relatedness pairs
│   ├── semrel_hau.csv                   # Hausa relatedness pairs
│   ├── semrel_amh.csv                   # Amharic relatedness pairs
│   ├── flores_zul_Latn.csv             # isiZulu sentences (FLORES-200)
│   ├── flores_xho_Latn.csv             # isiXhosa sentences (FLORES-200)
│   ├── flores_nso_Latn.csv             # Sepedi sentences (FLORES-200)
│   ├── flores_tsn_Latn.csv             # Setswana sentences (FLORES-200)
│   ├── flores_tso_Latn.csv             # Xitsonga sentences (FLORES-200)
│   ├── mafand_zul.csv                  # isiZulu sentences (Mafand-MT)
│   ├── mafand_xho.csv                  # isiXhosa sentences (Mafand-MT)
│   └── mafand_tsn.csv                  # Setswana sentences (Mafand-MT)
└── models/                              # Created after fine-tuning runs
    ├── afroxlmr_best/                   # Best AfroXLMR checkpoint
    └── xlmr_best/                       # Best XLM-R checkpoint
```

## Datasets

| Dataset | Languages | Use | Source |
|---|---|---|---|
| SemRel2024 | Kinyarwanda, Hausa, Amharic | Training and zero-shot eval | [HuggingFace](https://huggingface.co/datasets/SemRel/SemRel2024) |
| FLORES-200 | isiZulu, isiXhosa, Sepedi, Setswana, Xitsonga | SA embedding extraction | [HuggingFace](https://huggingface.co/datasets/facebook/flores) |
| Mafand-MT | isiZulu, isiXhosa, Setswana | SA embedding extraction | [HuggingFace](https://huggingface.co/datasets/masakhane/mafand) |

## Models

| Model | HuggingFace ID | Description |
|---|---|---|
| AfroXLMR | Davlan/afro-xlmr-base | XLM-R continued pre-training on African languages |
| XLM-R | FacebookAI/xlm-roberta-base | General multilingual baseline |

## Language Family Reference

| Language | Family | Sub-family | Dataset |
|---|---|---|---|
| Kinyarwanda | Niger-Congo | Bantu (Central) | SemRel (train) |
| Hausa | Afro-Asiatic | Chadic | SemRel (zero-shot eval) |
| Amharic | Afro-Asiatic | Semitic | SemRel (zero-shot eval) |
| isiZulu | Niger-Congo | Bantu (Nguni) | FLORES and Mafand |
| isiXhosa | Niger-Congo | Bantu (Nguni) | FLORES and Mafand |
| Sepedi | Niger-Congo | Bantu (Sotho) | FLORES only |
| Setswana | Niger-Congo | Bantu (Sotho) | FLORES and Mafand |
| Xitsonga | Niger-Congo | Bantu | FLORES only |

## Requirements

```
datasets<3.0.0
huggingface_hub
matplotlib
numpy
pandas
protobuf>=3.20.0
python-dotenv
scikit-learn>=1.3.0
scipy>=1.10.0
sentencepiece>=0.1.99
torch>=2.0.0
tqdm>=4.65.0
transformers>=4.35.0
```

Note: `datasets` must be pinned below 3.0.0 for compatibility with the Mafand-MT loader.

## References

- Ousidhoum et al. (2024). SemRel2024: A collection of semantic textual relatedness datasets for 14 languages. NAACL 2024.
- Alabi et al. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. COLING 2022.
- Conneau et al. (2020). Unsupervised cross-lingual representation learning at scale. ACL 2020.
- NLLB Team et al. (2022). No language left behind: Scaling human-centered machine translation. arXiv:2207.04672.
