"""# Fine-tuning
### AfroXLMR and XLM-R trained on Kinyarwanda SemRel data using mean-pooled sentence embeddings and cosine similarity regression
"""

# Fine-tune AfroXLMR (and XLM-R as baseline) on Kinyarwanda SemRel data
# Use mean-pooled sentence embeddings+cosine similarity regression
# Saves best checkpoint to models/

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from tqdm import tqdm

MODELS = {
    "afroxlmr": "Davlan/afro-xlmr-base",
    "xlmr": "FacebookAI/xlm-roberta-base",
}

DATA_DIR   = "data"
MODEL_DIR  = "models"
TRAIN_LANG = "kin"   #Kinyarwanda
BATCH_SIZE = 8       # reduced from 16 — large model needs smaller batches to stay stable
EPOCHS     = 6
LR = {"afroxlmr": 2e-5, "xlmr": 2e-5}
MAX_LEN    = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

class SemRelDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc1 = self.tokenizer(
            row["sentence1"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            row["sentence2"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids1":      enc1["input_ids"].squeeze(0),
            "attention_mask1": enc1["attention_mask"].squeeze(0),
            "input_ids2":      enc2["input_ids"].squeeze(0),
            "attention_mask2": enc2["attention_mask"].squeeze(0),
            "score":           torch.tensor(row["score"], dtype=torch.float),
        }

class SentenceSimilarityModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        )

    @staticmethod
    def mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pool(out.last_hidden_state, attention_mask)
        emb = torch.nan_to_num(emb, nan=0.0)
        norms = emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        emb = emb / norms
        return emb

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        e1  = self.encode(input_ids1, attention_mask1)
        e2  = self.encode(input_ids2, attention_mask2)
        cos = nn.functional.cosine_similarity(e1, e2)
        return (cos + 1.0) / 2.0

def train_model(model_key: str, model_name: str):
    print(f"Training {model_key.upper()}  ({model_name})")

    # Load data
    df_all   = pd.read_csv(f"{DATA_DIR}/semrel_{TRAIN_LANG}.csv")
    df_train = df_all[df_all["split"] == "train"]
    df_dev   = df_all[df_all["split"] == "dev"]
    print(f"Train: {len(df_train)}  Dev: {len(df_dev)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceSimilarityModel(model_name).to(DEVICE)
    if model_key == "afroxlmr":
        model.encoder.gradient_checkpointing_enable()

    train_loader = DataLoader(
        SemRelDataset(df_train, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0   # 2→0: prevents DataLoader hangs
    )
    dev_loader = DataLoader(
        SemRelDataset(df_dev, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0  # 2→0: prevents DataLoader hangs
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR[model_key], weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    loss_fn = nn.MSELoss()

    best_rho = -2.0
    best_epoch = -1
    patience = 3
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model(
                batch["input_ids1"], batch["attention_mask1"],
                batch["input_ids2"], batch["attention_mask2"],
            )
            loss = loss_fn(preds, batch["score"])
            if torch.isnan(loss):
                print(f"NaN batch skipped — pred range: {preds.min():.3f} to {preds.max():.3f}")
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate on dev
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch} [dev]", leave=False):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                preds = model(
                    batch["input_ids1"], batch["attention_mask1"],
                    batch["input_ids2"], batch["attention_mask2"],
                )
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["score"].cpu().numpy())

        rho, _ = spearmanr(all_preds, all_labels)
        if np.isnan(rho):   # constant predictions produce NaN — treat as worst case
            rho = -1.0
        print(f"  Epoch {epoch}  loss={avg_loss:.4f}  dev ρ={rho:.4f}")

        if rho > best_rho:
            best_rho   = rho
            best_epoch = epoch
            no_improve = 0
            save_path  = f"{MODEL_DIR}/{model_key}_best"
            model.encoder.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"New best - saved to {save_path}")
        else:
           no_improve += 1
           if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\n{model_key.upper()} best dev ρ = {best_rho:.4f} (epoch {best_epoch})")
    return best_rho

# Run fine-tuning — trains AfroXLMR then XLM-R on Kinyarwanda
# Checkpoints saved to models/afroxlmr_best and models/xlmr_best
ft_results = {}
for key, name in MODELS.items():
    rho = train_model(key, name)
    ft_results[key] = rho

print("\nFine-tuning complete")
for key, rho in ft_results.items():
    print(f"  {key:12s}  best dev ρ = {rho:.4f}")