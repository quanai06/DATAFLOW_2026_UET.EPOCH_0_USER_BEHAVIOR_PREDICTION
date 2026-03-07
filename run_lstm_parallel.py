"""
Parallel BiLSTM+Attention training while v3 transformer trains.
Different architecture → different inductive biases → complementary errors.
Uses same 96k pseudo-labeled dataset.
"""
import sys, pickle, os, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, '.')

from src.models.losses import ExactMatchFocalLoss
from src.metrics.metrics import evaluate_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer"
MODEL_DIR = Path("models/bilstm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f)
    VOCAB_SIZE = remap["vocab_size"]
MASK_TOKEN = VOCAB_SIZE  # same mask token as v3

with open(f"{FEATURE_PATH}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

NUM_CLASSES = [12, 31, 99, 12, 31, 99]
SEEDS = [100, 200, 300, 400, 500]

CFG = dict(
    embed_dim=256,
    hidden=256,
    num_layers=2,
    dropout=0.3,
    epochs=20,
    patience=4,
    batch_size=128,
    n_splits=5,
)


# ─────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, num_classes_list, embed_dim=256,
                 hidden=256, num_layers=2, dropout=0.3, num_stat_features=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)

        self.use_stats = num_stat_features > 0
        if self.use_stats:
            self.stats_branch = nn.Sequential(
                nn.Linear(num_stat_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            combined_dim = hidden * 2 * 3 + 64  # mean + max + attn pooling + stats
        else:
            combined_dim = hidden * 2 * 3

        self.heads = nn.ModuleList([nn.Linear(combined_dim, n) for n in num_classes_list])

    def forward(self, x, mask=None, stats=None):
        emb = self.dropout(self.embedding(x))          # (B, T, E)
        lstm_out, _ = self.lstm(emb)                   # (B, T, 2H)

        # Attention pooling
        attn_w = self.attn(lstm_out).squeeze(-1)       # (B, T)
        if mask is not None:
            attn_w = attn_w.masked_fill(mask, -1e9)
        attn_w = torch.softmax(attn_w, dim=1).unsqueeze(-1)
        attn_pool = (lstm_out * attn_w).sum(dim=1)     # (B, 2H)

        # Mean pooling (ignore padding)
        if mask is not None:
            mask_inv = (~mask).float().unsqueeze(-1)
            mean_pool = (lstm_out * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)
            x_max = lstm_out.masked_fill(mask.unsqueeze(-1), -1e9)
            max_pool = x_max.max(dim=1)[0]
        else:
            mean_pool = lstm_out.mean(dim=1)
            max_pool  = lstm_out.max(dim=1)[0]

        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=1)

        if self.use_stats and stats is not None:
            combined = torch.cat([combined, self.stats_branch(stats)], dim=1)

        return [head(combined) for head in self.heads]


# ─────────────────────────────────────────────────────────
# Dataset with augmentation
# ─────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, seqs, masks, stats, labels, augment=True,
                 crop_prob=0.35, mask_prob=0.15):
        self.seqs  = seqs;  self.masks  = masks
        self.stats = stats; self.labels = labels
        self.augment   = augment
        self.crop_prob = crop_prob
        self.mask_prob = mask_prob
        self.max_len   = seqs.shape[1]

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq   = self.seqs[idx].copy()
        mask  = self.masks[idx].copy()
        stats = self.stats[idx]
        label = self.labels[idx]

        if self.augment:
            actual_len = int((~mask).sum())
            if actual_len > 3 and np.random.random() < self.crop_prob:
                crop_len = np.random.randint(max(2, actual_len // 2), actual_len)
                start    = np.random.randint(0, actual_len - crop_len + 1)
                new_seq  = np.zeros(self.max_len, dtype=seq.dtype)
                new_mask = np.ones(self.max_len,  dtype=bool)
                new_seq[:crop_len]  = seq[start:start + crop_len]
                new_mask[:crop_len] = False
                seq, mask = new_seq, new_mask
                actual_len = crop_len
            if actual_len > 1:
                n_mask = max(1, int(actual_len * self.mask_prob))
                positions = np.random.choice(actual_len, size=n_mask, replace=False)
                seq[positions] = MASK_TOKEN

        return (torch.LongTensor(seq), torch.BoolTensor(mask),
                torch.FloatTensor(stats), torch.LongTensor(label))


def get_class_weights(y):
    weights_list = []
    for i in range(6):
        labels = y[:, i]
        classes, counts = np.unique(labels, return_counts=True)
        exponents = np.where(counts < 50, 0.4, np.where(counts < 200, 0.6, 0.75))
        w = (len(labels) / (len(classes) * counts)) ** exponents
        weights_list.append(torch.FloatTensor(w).to(DEVICE))
    return weights_list


def build_model(num_stat_features):
    return BiLSTMAttention(
        vocab_size=VOCAB_SIZE,
        num_classes_list=NUM_CLASSES,
        embed_dim=CFG["embed_dim"],
        hidden=CFG["hidden"],
        num_layers=CFG["num_layers"],
        dropout=CFG["dropout"],
        num_stat_features=num_stat_features,
    )


def train_fold(model, train_loader, val_loader, weights_list):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    criterion = ExactMatchFocalLoss(weights_list=weights_list, alpha=0.01, gamma=2.0)

    best_val, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(CFG["epochs"]):
        model.train()
        total = 0
        for x, mask, stats, y in train_loader:
            x, mask, stats, y = x.to(DEVICE), mask.to(DEVICE), stats.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x, mask, stats), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        scheduler.step()
        avg = total / len(train_loader)

        model.eval()
        vtotal = 0
        with torch.no_grad():
            for x, mask, stats, y in val_loader:
                x, mask, stats, y = x.to(DEVICE), mask.to(DEVICE), stats.to(DEVICE), y.to(DEVICE)
                vtotal += criterion(model(x, mask, stats), y).item()
        vloss = vtotal / len(val_loader)

        lr = optimizer.param_groups[0]["lr"]
        print(f"    Epoch {epoch+1:2d}/{CFG['epochs']} LR:{lr:.6f} Loss:{avg:.4f} Val:{vloss:.4f}", flush=True)

        if vloss < best_val:
            best_val, best_state, patience_cnt = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"    Early stop epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model


def main():
    # Load data — same pseudo-labeled set as v3
    print("Loading data...", flush=True)
    X_tr  = np.load(f"{FEATURE_PATH}/X_train_seq.npy")
    M_tr  = np.load(f"{FEATURE_PATH}/X_train_mask.npy")
    S_tr  = np.load(f"{FEATURE_PATH}/X_train_stats.npy").astype(np.float32)
    y_tr  = np.load(f"{FEATURE_PATH}/y_train.npy")

    X_va  = np.load(f"{FEATURE_PATH}/X_val_seq.npy")
    M_va  = np.load(f"{FEATURE_PATH}/X_val_mask.npy")
    S_va  = np.load(f"{FEATURE_PATH}/X_val_stats.npy").astype(np.float32)
    y_va  = np.load(f"{FEATURE_PATH}/y_val.npy")

    # Re-use pseudo-labels from v3 run (already generated)
    X_te  = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
    M_te  = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
    S_te  = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)

    # Generate pseudo-labels using v2 models (geometric mean of logits)
    from run_v3 import load_v2_models, generate_soft_pseudo_labels
    print("Generating pseudo-labels with v2 models...", flush=True)
    v2_models = load_v2_models()
    pl_seq, pl_mask, pl_stats, pl_hard, _, _ = generate_soft_pseudo_labels(
        v2_models, X_te, M_te, S_te, conf_threshold=0.99
    )
    print(f"Pseudo-labels: {len(pl_seq)}", flush=True)
    del v2_models

    X_all = np.concatenate([X_tr, X_va, pl_seq])
    M_all = np.concatenate([M_tr, M_va, pl_mask])
    S_all = np.concatenate([S_tr, S_va, pl_stats])
    y_all = np.concatenate([y_tr, y_va, pl_hard])

    num_stat_features = S_tr.shape[1]
    print(f"Total dataset: {len(X_all)} samples", flush=True)

    kfold = StratifiedKFold(n_splits=CFG["n_splits"], shuffle=True, random_state=42)
    strat_key = y_all[:, 2]
    saved = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'='*50}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*50}", flush=True)

        for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_all, strat_key)):
            print(f"\n  FOLD {fold+1}/{CFG['n_splits']}", flush=True)

            weights_list = get_class_weights(y_all[tr_idx])
            train_ds = SeqDataset(X_all[tr_idx], M_all[tr_idx], S_all[tr_idx], y_all[tr_idx], augment=True)
            val_ds   = SeqDataset(X_all[va_idx], M_all[va_idx], S_all[va_idx], y_all[va_idx], augment=False)

            train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                      num_workers=4, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                                      num_workers=4, pin_memory=True)

            model = build_model(num_stat_features).to(DEVICE)
            model = train_fold(model, train_loader, val_loader, weights_list)

            val_res = evaluate_model(model, val_loader, DEVICE)
            val_f1  = val_res["macro_f1_score"]
            val_em  = val_res["exact_match_accuracy"]
            print(f"  Fold {fold+1} F1={val_f1:.4f} EM={val_em:.4f}", flush=True)

            path = MODEL_DIR / f"bilstm_seed{seed}_fold{fold}.pt"
            torch.save({
                "model_state_dict":  model.state_dict(),
                "val_macro_f1":      val_f1,
                "val_metrics":       val_res,
                "num_stat_features": num_stat_features,
            }, path)
            saved.append((path, val_f1))
            print(f"  ✅ Saved {path}", flush=True)

        # Full model
        print(f"\n  FULL (seed={seed})", flush=True)
        weights_full = get_class_weights(y_all)
        full_ds = SeqDataset(X_all, M_all, S_all, y_all, augment=True)
        full_loader = DataLoader(full_ds, batch_size=CFG["batch_size"], shuffle=True,
                                 num_workers=4, pin_memory=True)
        _, va0 = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_all, strat_key))
        val_ds0 = SeqDataset(X_all[va0], M_all[va0], S_all[va0], y_all[va0], augment=False)
        val_loader0 = DataLoader(val_ds0, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model(num_stat_features).to(DEVICE)
        model = train_fold(model, full_loader, val_loader0, weights_full)

        val_m = evaluate_model(model, val_loader0, DEVICE)
        path  = MODEL_DIR / f"bilstm_seed{seed}_full.pt"
        torch.save({
            "model_state_dict":  model.state_dict(),
            "val_macro_f1":      val_m["macro_f1_score"],
            "num_stat_features": num_stat_features,
        }, path)
        saved.append((path, val_m["macro_f1_score"]))
        print(f"  ✅ Full seed={seed} F1={val_m['macro_f1_score']:.4f}", flush=True)

    print(f"\n{'='*50}", flush=True)
    print(f"BiLSTM training done. {len(saved)} models saved.", flush=True)
    f1s = [f for _, f in saved]
    print(f"Val F1 range: [{min(f1s):.4f}, {max(f1s):.4f}]  mean={np.mean(f1s):.4f}", flush=True)


if __name__ == "__main__":
    main()
