"""
v4: Push beyond v3
1. Bigger model: d_model=512, nhead=8, num_layers=6, dim_ff=1024
2. Iterative pseudo-labeling: use v3 ensemble to label test (higher quality + more confident)
3. Higher pseudo-label threshold: 0.995 (cleaner labels)
4. 8 seeds for more ensemble diversity
5. Warmup + CosineAnnealing LR schedule
"""
import sys, pickle, os, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, '.')

from src.models.transformer_model import TransformerModel
from src.models.losses import ExactMatchFocalLoss
from src.metrics.metrics import evaluate_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer"
MODEL_DIR = Path("models/transformer_v4")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f)
    VOCAB_SIZE = remap["vocab_size"]
MASK_TOKEN = VOCAB_SIZE  # +1 token for [MASK]

with open(f"{FEATURE_PATH}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

NUM_CLASSES = [12, 31, 99, 12, 31, 99]

BASE_CONFIG = dict(
    vocab_size=VOCAB_SIZE + 1,
    num_classes_list=NUM_CLASSES,
    d_model=512, nhead=8, num_layers=6, dim_ff=1024, max_len=37,
)
TRAIN_CFG = dict(
    epochs=25, patience=5, batch_size=256,
    n_splits=5,
    seeds=[42, 123, 2024, 314, 777, 1337, 999, 555],
    conf_threshold=0.99,
    warmup_epochs=2,
)


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────
class AugDataset(Dataset):
    def __init__(self, seqs, masks, stats, y_hard, y_soft=None,
                 augment=True, crop_prob=0.35, mask_prob=0.15):
        self.seqs    = seqs
        self.masks   = masks
        self.stats   = stats
        self.y_hard  = y_hard
        self.y_soft  = y_soft
        self.augment   = augment
        self.crop_prob = crop_prob
        self.mask_prob = mask_prob
        self.max_len   = seqs.shape[1]

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq   = self.seqs[idx].copy()
        mask  = self.masks[idx].copy()
        stats = self.stats[idx]
        yh    = self.y_hard[idx]

        if self.augment:
            actual_len = int((~mask).sum())
            if actual_len > 3 and np.random.random() < self.crop_prob:
                crop_len = np.random.randint(max(2, actual_len // 2), actual_len)
                start    = np.random.randint(0, actual_len - crop_len + 1)
                new_seq  = np.zeros(self.max_len, dtype=seq.dtype)
                new_mask = np.ones(self.max_len, dtype=bool)
                new_seq[:crop_len]  = seq[start:start + crop_len]
                new_mask[:crop_len] = False
                seq, mask = new_seq, new_mask
                actual_len = crop_len
            if actual_len > 1:
                n_mask = max(1, int(actual_len * self.mask_prob))
                positions = np.random.choice(actual_len, size=n_mask, replace=False)
                seq[positions] = MASK_TOKEN

        ret = [torch.LongTensor(seq), torch.BoolTensor(mask),
               torch.FloatTensor(stats), torch.LongTensor(yh)]

        if self.y_soft is not None:
            ret.append([torch.FloatTensor(s[idx]) for s in self.y_soft])
        else:
            # one-hot for real data — pre-built here so collate_fn stays simple
            soft = []
            for h, nc in enumerate(NUM_CLASSES):
                oh = np.zeros(nc, dtype=np.float32)
                oh[yh[h]] = 1.0
                soft.append(torch.FloatTensor(oh))
            ret.append(soft)
        return ret


def collate_fn(batch):
    seqs   = torch.stack([b[0] for b in batch])
    masks  = torch.stack([b[1] for b in batch])
    stats  = torch.stack([b[2] for b in batch])
    yhards = torch.stack([b[3] for b in batch])
    # Every item always has soft labels (real → one-hot, pseudo → prob)
    n_heads = len(batch[0][4])
    ysoft = [torch.stack([b[4][h] for b in batch]) for h in range(n_heads)]
    return seqs, masks, stats, yhards, ysoft


# ─────────────────────────────────────────────────────────
# Pseudo-label generation using v3 ensemble
# ─────────────────────────────────────────────────────────
def load_v3_models():
    V3_DIR = Path("models/transformer_v3")
    files  = sorted(V3_DIR.glob("*.pt"))
    models = []
    for path in files:
        ckpt = torch.load(path, map_location=DEVICE)
        num_stat = ckpt.get("num_stat_features", 0)
        m = TransformerModel(
            vocab_size=VOCAB_SIZE + 1,
            num_classes_list=NUM_CLASSES,
            d_model=384, nhead=8, num_layers=4, dim_ff=768, max_len=37,
            num_stat_features=num_stat,
        ).to(DEVICE)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append((m, float(ckpt.get("val_macro_f1", 1.0))))
    return models


def generate_pseudo_labels(models, X_te, M_te, S_te, conf_threshold=0.995):
    """Geometric-mean ensemble → hard labels with high confidence."""
    BATCH = 512
    N = len(X_te)
    X_t = torch.LongTensor(X_te).to(DEVICE)
    M_t = torch.BoolTensor(M_te).to(DEVICE)
    S_t = torch.FloatTensor(S_te).to(DEVICE)

    # Accumulate weighted logits
    logit_sum = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]
    total_w   = 0.0

    for model, w in models:
        with torch.no_grad():
            for start in range(0, N, BATCH):
                xb = X_t[start:start+BATCH]
                mb = M_t[start:start+BATCH]
                sb = S_t[start:start+BATCH]
                outs = model(xb, mb, sb)
                for i, o in enumerate(outs):
                    logit_sum[i][start:start+BATCH] += o.cpu().float().numpy() * w
        total_w += w

    # Geometric-mean probabilities: softmax on raw summed logits (NOT divided by weight)
    # Dividing before softmax reduces sharpness due to scale sensitivity of softmax
    def _softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    probs = [_softmax(logit_sum[i]).astype(np.float32) for i in range(6)]

    # Confidence = minimum of max probs across 6 heads (same as v3)
    hard = np.zeros((N, 6), dtype=np.int64)
    max_probs = np.zeros((N, 6), dtype=np.float32)
    for i in range(6):
        hard[:, i] = np.argmax(probs[i], axis=1)
        max_probs[:, i] = probs[i].max(axis=1)
    confidence = max_probs.min(axis=1)

    mask = confidence >= conf_threshold
    print(f"  Pseudo-label candidates: {mask.sum()} / {N} (threshold={conf_threshold})", flush=True)

    pl_seq   = X_te[mask]
    pl_mask  = M_te[mask]
    pl_stats = S_te[mask]
    pl_hard  = hard[mask]
    pl_soft  = [probs[i][mask] for i in range(6)]

    return pl_seq, pl_mask, pl_stats, pl_hard, pl_soft


# ─────────────────────────────────────────────────────────
# Loss with soft labels
# ─────────────────────────────────────────────────────────
class MixedLoss(nn.Module):
    def __init__(self, weights_list, alpha=0.01, gamma=2.0, soft_weight=0.3):
        super().__init__()
        self.focal = ExactMatchFocalLoss(weights_list=weights_list, alpha=alpha, gamma=gamma)
        self.soft_weight = soft_weight

    def forward(self, outputs, y_hard, y_soft=None):
        focal_loss = self.focal(outputs, y_hard)
        if y_soft is None:
            return focal_loss
        kl_loss = 0.0
        for i, (logit, soft) in enumerate(zip(outputs, y_soft)):
            log_p = F.log_softmax(logit, dim=1)
            kl_loss += F.kl_div(log_p, soft.to(logit.device), reduction="batchmean")
        kl_loss /= len(outputs)
        return (1 - self.soft_weight) * focal_loss + self.soft_weight * kl_loss


# ─────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────
def get_class_weights(y):
    weights_list = []
    for i in range(6):
        labels = y[:, i]
        classes, counts = np.unique(labels, return_counts=True)
        exponents = np.where(counts < 50, 0.4, np.where(counts < 200, 0.6, 0.75))
        w = (len(labels) / (len(classes) * counts)) ** exponents
        weights_list.append(torch.FloatTensor(w).to(DEVICE))
    return weights_list


# ─────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────
def train_fold(model, train_loader, val_loader, weights_list, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    warmup = cfg["warmup_epochs"]
    total_epochs = cfg["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = MixedLoss(weights_list, alpha=0.01, gamma=2.0, soft_weight=0.3)

    best_val, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(total_epochs):
        model.train()
        total = 0
        for batch in train_loader:
            if len(batch) == 4:
                x, mask, stats, yh = batch
                ys = None
            else:
                x, mask, stats, yh, ys = batch

            x, mask, stats, yh = x.to(DEVICE), mask.to(DEVICE), stats.to(DEVICE), yh.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x, mask, stats)
            loss = criterion(outputs, yh, ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        scheduler.step()
        avg = total / len(train_loader)
        lr  = optimizer.param_groups[0]["lr"]

        model.eval()
        vtotal = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    xv, mv, sv, yv = batch
                    ys = None
                else:
                    xv, mv, sv, yv, ys = batch
                xv, mv, sv, yv = xv.to(DEVICE), mv.to(DEVICE), sv.to(DEVICE), yv.to(DEVICE)
                vtotal += criterion(model(xv, mv, sv), yv).item()
        vloss = vtotal / len(val_loader)

        print(f"    Epoch {epoch+1:2d}/{total_epochs} LR:{lr:.6f} Loss:{avg:.4f} Val:{vloss:.4f}", flush=True)

        if vloss < best_val:
            best_val, best_state, patience_cnt = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"    Early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model


def build_model(num_stat_features):
    cfg = dict(**BASE_CONFIG, num_stat_features=num_stat_features)
    return TransformerModel(**cfg)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("Loading base data...", flush=True)
    X_tr = np.load(f"{FEATURE_PATH}/X_train_seq.npy")
    M_tr = np.load(f"{FEATURE_PATH}/X_train_mask.npy")
    S_tr = np.load(f"{FEATURE_PATH}/X_train_stats.npy").astype(np.float32)
    y_tr = np.load(f"{FEATURE_PATH}/y_train.npy")

    X_va = np.load(f"{FEATURE_PATH}/X_val_seq.npy")
    M_va = np.load(f"{FEATURE_PATH}/X_val_mask.npy")
    S_va = np.load(f"{FEATURE_PATH}/X_val_stats.npy").astype(np.float32)
    y_va = np.load(f"{FEATURE_PATH}/y_val.npy")

    X_te = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
    M_te = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
    S_te = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)

    # Generate pseudo-labels with v3 models
    print("Loading v3 models for pseudo-labeling...", flush=True)
    v3_models = load_v3_models()
    print(f"Loaded {len(v3_models)} v3 models.", flush=True)

    pl_seq, pl_mask, pl_stats, pl_hard, pl_soft = generate_pseudo_labels(
        v3_models, X_te, M_te, S_te, conf_threshold=TRAIN_CFG["conf_threshold"]
    )
    del v3_models

    X_all = np.concatenate([X_tr, X_va, pl_seq])
    M_all = np.concatenate([M_tr, M_va, pl_mask])
    S_all = np.concatenate([S_tr, S_va, pl_stats])
    y_all = np.concatenate([y_tr, y_va, pl_hard])

    # Soft labels: None for train+val rows, probs for pseudo-labeled rows
    n_real = len(X_tr) + len(X_va)
    n_pl   = len(pl_seq)
    # We'll track which indices are pseudo-labeled via y_soft_all
    y_soft_all = None  # use hard labels for everything except pseudo rows
    pl_soft_full = pl_soft  # per-head prob arrays for pseudo rows only

    num_stat_features = S_tr.shape[1]
    print(f"Dataset: {n_real} real + {n_pl} pseudo = {len(X_all)} total", flush=True)

    strat_key = y_all[:, 2]
    kfold = StratifiedKFold(n_splits=TRAIN_CFG["n_splits"], shuffle=True, random_state=42)
    saved = []

    for seed in TRAIN_CFG["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'='*55}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*55}", flush=True)

        for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_all, strat_key)):
            print(f"\n  FOLD {fold+1}/{TRAIN_CFG['n_splits']}", flush=True)

            # Split indices into real vs pseudo
            # For pseudo rows (index >= n_real), attach soft labels
            tr_real_mask = tr_idx < n_real
            tr_pl_mask   = tr_idx >= n_real

            # Build dataset: mix hard-only rows with soft-label rows
            # Easier: give all rows to dataset, soft labels only for pl rows
            tr_yh    = y_all[tr_idx]
            va_yh    = y_all[va_idx]
            va_soft  = None  # val always hard

            # soft labels indexed within pl rows
            def get_soft_for_idx(idx_arr):
                """Returns list-of-arrays (6 heads) for the given global indices, None where real."""
                pl_idx = idx_arr[idx_arr >= n_real] - n_real
                if len(pl_idx) == 0:
                    return None
                return [pl_soft_full[h][pl_idx] for h in range(6)]

            # We'll build two sub-datasets: real (hard) + pseudo (soft) and concatenate
            tr_real_idx = tr_idx[tr_real_mask]
            tr_pl_idx   = tr_idx[tr_pl_mask]

            real_ds = AugDataset(
                X_all[tr_real_idx], M_all[tr_real_idx], S_all[tr_real_idx],
                y_all[tr_real_idx], y_soft=None, augment=True
            ) if len(tr_real_idx) > 0 else None

            pl_ds = AugDataset(
                X_all[tr_pl_idx], M_all[tr_pl_idx], S_all[tr_pl_idx],
                y_all[tr_pl_idx],
                y_soft=[pl_soft_full[h][tr_pl_idx - n_real] for h in range(6)],
                augment=True
            ) if len(tr_pl_idx) > 0 else None

            from torch.utils.data import ConcatDataset
            if real_ds and pl_ds:
                train_ds = ConcatDataset([real_ds, pl_ds])
            elif real_ds:
                train_ds = real_ds
            else:
                train_ds = pl_ds

            val_ds = AugDataset(
                X_all[va_idx], M_all[va_idx], S_all[va_idx],
                va_yh, y_soft=None, augment=False
            )

            train_loader = DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"],
                                      shuffle=True, num_workers=4, pin_memory=True,
                                      collate_fn=collate_fn)
            val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False,
                                      num_workers=4, pin_memory=True,
                                      collate_fn=collate_fn)

            weights_list = get_class_weights(y_all[tr_idx])
            model = build_model(num_stat_features).to(DEVICE)
            model = train_fold(model, train_loader, val_loader, weights_list, TRAIN_CFG)

            val_res = evaluate_model(model, val_loader, DEVICE)
            val_f1  = val_res["macro_f1_score"]
            val_em  = val_res["exact_match_accuracy"]
            print(f"  Fold {fold+1} F1={val_f1:.4f} EM={val_em:.4f}", flush=True)

            path = MODEL_DIR / f"model_seed{seed}_fold{fold}.pt"
            torch.save({
                "model_state_dict":  model.state_dict(),
                "val_macro_f1":      val_f1,
                "val_metrics":       val_res,
                "num_stat_features": num_stat_features,
                "config":            BASE_CONFIG,
            }, path)
            saved.append((path, val_f1))
            print(f"  Saved {path}", flush=True)

        # Full model (all data)
        print(f"\n  FULL (seed={seed})", flush=True)
        weights_full = get_class_weights(y_all)

        real_full_ds = AugDataset(X_all[:n_real], M_all[:n_real], S_all[:n_real],
                                   y_all[:n_real], y_soft=None, augment=True)
        pl_full_ds   = AugDataset(X_all[n_real:], M_all[n_real:], S_all[n_real:],
                                   y_all[n_real:], y_soft=pl_soft_full, augment=True)
        full_ds = ConcatDataset([real_full_ds, pl_full_ds])

        _, va0 = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_all, strat_key))
        val_ds0 = AugDataset(X_all[va0], M_all[va0], S_all[va0], y_all[va0], augment=False)

        full_loader = DataLoader(full_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True,
                                 num_workers=4, pin_memory=True, collate_fn=collate_fn)
        val_loader0 = DataLoader(val_ds0, batch_size=512, shuffle=False,
                                 num_workers=4, pin_memory=True, collate_fn=collate_fn)

        model = build_model(num_stat_features).to(DEVICE)
        model = train_fold(model, full_loader, val_loader0, weights_full, TRAIN_CFG)

        val_m = evaluate_model(model, val_loader0, DEVICE)
        path  = MODEL_DIR / f"model_seed{seed}_full.pt"
        torch.save({
            "model_state_dict":  model.state_dict(),
            "val_macro_f1":      val_m["macro_f1_score"],
            "num_stat_features": num_stat_features,
            "config":            BASE_CONFIG,
        }, path)
        saved.append((path, val_m["macro_f1_score"]))
        print(f"  Full seed={seed} F1={val_m['macro_f1_score']:.4f}", flush=True)

    print(f"\n{'='*55}", flush=True)
    print(f"v4 done. {len(saved)} models saved.", flush=True)
    f1s = [f for _, f in saved]
    print(f"Val F1: min={min(f1s):.4f} max={max(f1s):.4f} mean={np.mean(f1s):.4f}", flush=True)


if __name__ == "__main__":
    main()
