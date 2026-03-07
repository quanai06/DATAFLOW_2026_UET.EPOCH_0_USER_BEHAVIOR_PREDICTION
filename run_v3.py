"""
v3 improvements over v2:
1. Re-generate pseudo-labels using v2 models (higher quality, more samples)
2. Soft pseudo-labels — store full probability distributions, train with KL divergence
3. Token masking augmentation (BERT-style, 15% mask rate)
4. Bigger model: d_model=384
5. More seeds: 6 total (42, 123, 2024, 314, 777, 1337)
"""
import sys, pickle, os, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, '.')

from src.models.transformer_model import TransformerModel
from src.models.losses import ExactMatchFocalLoss
from src.metrics.metrics import evaluate_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer"
MODEL_DIR = Path("models/transformer_v3")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f)
    VOCAB_SIZE = remap["vocab_size"]
MASK_TOKEN = VOCAB_SIZE  # extra token index for [MASK]

with open(f"{FEATURE_PATH}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

NUM_CLASSES = [12, 31, 99, 12, 31, 99]

# d_model=384: bigger capacity for 95k+ dataset
BASE_CONFIG = dict(
    vocab_size=VOCAB_SIZE + 1,  # +1 for MASK token
    num_classes_list=NUM_CLASSES,
    d_model=384, nhead=8, num_layers=4, dim_ff=768, max_len=37,
)
TRAIN_CONFIG = dict(
    epochs=20, patience=4, batch_size=64,
    n_splits=5,
    seeds=[42, 123, 2024, 314, 777, 1337],
)


# ─────────────────────────────────────────────────────────
# Dataset with token masking + subsequence crop
# ─────────────────────────────────────────────────────────
class AugDatasetV3(Dataset):
    """
    Supports both hard labels (y_soft=None) and soft labels (y_soft = prob arrays).
    Augmentations: random token masking + subsequence cropping.
    """
    def __init__(self, seqs, masks, stats, y_hard, y_soft=None,
                 augment=True, crop_prob=0.35, mask_prob=0.15):
        self.seqs    = seqs
        self.masks   = masks
        self.stats   = stats
        self.y_hard  = y_hard
        self.y_soft  = y_soft       # list of 6 arrays (N, n_classes) or None
        self.augment = augment
        self.crop_prob = crop_prob
        self.mask_prob = mask_prob
        self.max_len = seqs.shape[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq   = self.seqs[idx].copy()
        mask  = self.masks[idx].copy()
        stats = self.stats[idx]
        label = self.y_hard[idx]

        soft = None
        if self.y_soft is not None:
            soft = [self.y_soft[i][idx] for i in range(6)]

        if self.augment:
            actual_len = int((~mask).sum())

            # Subsequence crop
            if actual_len > 3 and np.random.random() < self.crop_prob:
                crop_len = np.random.randint(max(2, actual_len // 2), actual_len)
                start    = np.random.randint(0, actual_len - crop_len + 1)
                new_seq  = np.zeros(self.max_len, dtype=seq.dtype)
                new_mask = np.ones(self.max_len, dtype=bool)
                new_seq[:crop_len]  = seq[start:start + crop_len]
                new_mask[:crop_len] = False
                seq, mask = new_seq, new_mask
                actual_len = crop_len

            # Token masking (BERT-style)
            if actual_len > 1:
                n_mask = max(1, int(actual_len * self.mask_prob))
                positions = np.random.choice(actual_len, size=n_mask, replace=False)
                seq[positions] = MASK_TOKEN

        out = [torch.LongTensor(seq), torch.BoolTensor(mask),
               torch.FloatTensor(stats), torch.LongTensor(label)]

        if soft is not None:
            out.append([torch.FloatTensor(s) for s in soft])

        return tuple(out)


def collate_fn(batch):
    has_soft = len(batch[0]) == 5
    seqs   = torch.stack([b[0] for b in batch])
    masks  = torch.stack([b[1] for b in batch])
    stats  = torch.stack([b[2] for b in batch])
    labels = torch.stack([b[3] for b in batch])
    if not has_soft:
        return seqs, masks, stats, labels
    # soft: list of 6 tensors each (B, n_classes)
    soft = [torch.stack([b[4][i] for b in batch]) for i in range(6)]
    return seqs, masks, stats, labels, soft


# ─────────────────────────────────────────────────────────
# Mixed loss: focal (real data) + KL divergence (pseudo)
# ─────────────────────────────────────────────────────────
class MixedLoss(nn.Module):
    def __init__(self, weights_list=None, alpha=0.01, gamma=2.0, kl_weight=1.0):
        super().__init__()
        self.focal = ExactMatchFocalLoss(weights_list=weights_list, alpha=alpha, gamma=gamma)
        self.kl_weight = kl_weight

    def forward(self, outputs, y_hard, y_soft=None):
        # Focal loss on hard labels
        hard_loss = self.focal(outputs, y_hard)

        if y_soft is None:
            return hard_loss

        # KL divergence loss on soft labels
        kl_loss = 0.0
        for i, out in enumerate(outputs):
            log_prob = F.log_softmax(out, dim=1)
            kl = F.kl_div(log_prob, y_soft[i].to(out.device), reduction='batchmean')
            kl_loss = kl_loss + kl

        return hard_loss + self.kl_weight * (kl_loss / 6.0)


# ─────────────────────────────────────────────────────────
# Class weights (adaptive)
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


def build_model(num_stat_features):
    return TransformerModel(num_stat_features=num_stat_features, **BASE_CONFIG)


# ─────────────────────────────────────────────────────────
# Load v2 models for pseudo-label generation
# ─────────────────────────────────────────────────────────
def load_v2_models():
    models = []
    for p in sorted(Path("models/transformer_v2").glob("*.pt")):
        ckpt = torch.load(p, map_location=DEVICE)
        m = TransformerModel(
            vocab_size=VOCAB_SIZE, num_classes_list=NUM_CLASSES,
            d_model=256, nhead=8, num_layers=4, dim_ff=512, max_len=37,
            num_stat_features=ckpt.get("num_stat_features", 0)
        )
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(DEVICE).eval()
        models.append(m)
    print(f"Loaded {len(models)} v2 models", flush=True)
    return models


# ─────────────────────────────────────────────────────────
# Generate SOFT pseudo-labels from ensemble
# ─────────────────────────────────────────────────────────
def generate_soft_pseudo_labels(models, X_seq_np, X_mask_np, X_stats_np,
                                 conf_threshold=0.99, batch_size=512):
    N = len(X_seq_np)
    # Accumulate arithmetic mean of per-model softmax probabilities
    prob_sums = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]

    ds = TensorDataset(
        torch.LongTensor(X_seq_np),
        torch.BoolTensor(X_mask_np),
        torch.FloatTensor(X_stats_np)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Sum logits across models, then softmax → geometric mean (sharper, higher confidence)
    logit_sums = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]

    for m in models:
        m.eval()
        with torch.no_grad():
            for start_idx, (x, mask, stats) in enumerate(loader):
                x, mask, stats = x.to(DEVICE), mask.to(DEVICE), stats.to(DEVICE)
                b_start = start_idx * batch_size
                b_end   = min(b_start + batch_size, N)
                outs = m(x, mask, stats)
                for i, o in enumerate(outs):
                    logit_sums[i][b_start:b_end] += o.cpu().numpy()

    # softmax of summed logits = geometric mean of individual softmax distributions
    def _softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    soft_probs = [_softmax(ls).astype(np.float32) for ls in logit_sums]

    # Hard predictions + confidence
    hard_preds = np.stack([p.argmax(axis=1) for p in soft_probs], axis=1)
    min_conf   = np.stack([p.max(axis=1) for p in soft_probs], axis=1).min(axis=1)

    keep = min_conf > conf_threshold
    print(f"Pseudo-labels (conf>{conf_threshold}): {keep.sum()} / {N} ({keep.mean()*100:.1f}%)", flush=True)

    soft_kept = [p[keep].astype(np.float32) for p in soft_probs]
    return X_seq_np[keep], X_mask_np[keep], X_stats_np[keep], hard_preds[keep], soft_kept, keep


# ─────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────
def train_fold(model, train_loader, val_loader, weights_list, has_soft):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
    criterion = MixedLoss(weights_list=weights_list, alpha=0.01, gamma=2.0, kl_weight=0.5)

    best_val, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        total = 0

        for batch in train_loader:
            if has_soft and len(batch) == 5:
                x, mask, stats, y, y_soft = batch
                y_soft = [s.to(DEVICE) for s in y_soft]
            else:
                x, mask, stats, y = batch
                y_soft = None

            x, mask, stats, y = x.to(DEVICE), mask.to(DEVICE), stats.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x, mask, stats)
            loss = criterion(outputs, y, y_soft)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        scheduler.step()
        avg = total / len(train_loader)

        model.eval()
        vtotal = 0
        with torch.no_grad():
            for batch in val_loader:
                xv, mv, sv, yv = batch[:4]
                xv, mv, sv, yv = xv.to(DEVICE), mv.to(DEVICE), sv.to(DEVICE), yv.to(DEVICE)
                outs = model(xv, mv, sv)
                vtotal += criterion(outs, yv).item()
        vloss = vtotal / len(val_loader)

        lr = optimizer.param_groups[0]["lr"]
        print(f"    Epoch {epoch+1:2d}/{TRAIN_CONFIG['epochs']} LR:{lr:.6f} Loss:{avg:.4f} Val:{vloss:.4f}", flush=True)

        if vloss < best_val:
            best_val, best_state, patience_cnt = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= TRAIN_CONFIG["patience"]:
                print(f"    Early stop epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────
# Multi-seed training
# ─────────────────────────────────────────────────────────
def run_training(X_seq, X_mask, X_stats, y_hard, y_soft, num_stat_features):
    kfold = StratifiedKFold(n_splits=TRAIN_CONFIG["n_splits"], shuffle=True, random_state=42)
    strat_key = y_hard[:, 2]
    saved = []

    for seed in TRAIN_CONFIG["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'='*55}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*55}", flush=True)

        for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_seq, strat_key)):
            print(f"\n  FOLD {fold+1}/{TRAIN_CONFIG['n_splits']}", flush=True)

            weights_list = get_class_weights(y_hard[tr_idx])

            # Build soft label arrays for train split
            y_soft_tr = [y_soft[i][tr_idx] for i in range(6)] if y_soft is not None else None

            train_ds = AugDatasetV3(
                X_seq[tr_idx], X_mask[tr_idx], X_stats[tr_idx], y_hard[tr_idx],
                y_soft=y_soft_tr, augment=True
            )
            val_ds = AugDatasetV3(
                X_seq[va_idx], X_mask[va_idx], X_stats[va_idx], y_hard[va_idx],
                y_soft=None, augment=False
            )

            train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"],
                                      shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                                      collate_fn=collate_fn, num_workers=2, pin_memory=True)

            model = build_model(num_stat_features).to(DEVICE)
            model = train_fold(model, train_loader, val_loader, weights_list, has_soft=(y_soft is not None))

            val_res  = evaluate_model(model, val_loader, DEVICE)
            val_f1   = val_res["macro_f1_score"]
            val_em   = val_res["exact_match_accuracy"]
            print(f"  Fold {fold+1} val F1={val_f1:.4f} EM={val_em:.4f}", flush=True)

            path = MODEL_DIR / f"model_seed{seed}_fold{fold}.pt"
            torch.save({
                "model_state_dict":  model.state_dict(),
                "val_macro_f1":      val_f1,
                "val_metrics":       val_res,
                "num_stat_features": num_stat_features,
                "d_model": 384, "seed": seed, "fold": fold,
            }, path)
            saved.append((path, val_f1))
            print(f"  ✅ Saved {path}", flush=True)

        # Full model per seed
        print(f"\n  FULL (seed={seed})", flush=True)
        weights_full = get_class_weights(y_hard)
        full_ds = AugDatasetV3(X_seq, X_mask, X_stats, y_hard,
                               y_soft=y_soft, augment=True)
        full_loader = DataLoader(full_ds, batch_size=TRAIN_CONFIG["batch_size"],
                                 shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        # Val loader for early stopping reference
        _, va_idx0 = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_seq, strat_key))
        val_ds0  = AugDatasetV3(X_seq[va_idx0], X_mask[va_idx0], X_stats[va_idx0], y_hard[va_idx0],
                                augment=False)
        val_loader0 = DataLoader(val_ds0, batch_size=256, shuffle=False, collate_fn=collate_fn,
                                 num_workers=2, pin_memory=True)

        model = build_model(num_stat_features).to(DEVICE)
        model = train_fold(model, full_loader, val_loader0, weights_full, has_soft=(y_soft is not None))

        val_m = evaluate_model(model, val_loader0, DEVICE)
        path  = MODEL_DIR / f"model_seed{seed}_full.pt"
        torch.save({
            "model_state_dict":  model.state_dict(),
            "val_macro_f1":      val_m["macro_f1_score"],
            "num_stat_features": num_stat_features,
            "d_model": 384, "seed": seed,
        }, path)
        saved.append((path, val_m["macro_f1_score"]))
        print(f"  ✅ Full seed={seed} F1={val_m['macro_f1_score']:.4f}", flush=True)

    return saved


# ─────────────────────────────────────────────────────────
# MC dropout + TTA prediction
# ─────────────────────────────────────────────────────────
def predict_with_mc_tta(models, weights, X_seq_np, X_mask_np, X_stats_np,
                        mc_passes=15, batch_size=256):
    N = len(X_seq_np)
    prob_sum = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]
    total_w  = sum(weights)

    def accumulate(m, x, mask, stats, w, start, end, train_mode=False):
        if train_mode: m.train()
        else:          m.eval()
        with torch.no_grad():
            outs = m(x, mask, stats)
        for i, o in enumerate(outs):
            prob_sum[i][start:end] += torch.softmax(o, dim=1).cpu().numpy() * w

    for m, w in zip(models, weights):
        mw = w / total_w
        for variant in ["orig", "drop_first", "drop_last"]:
            tw = mw / 3.0
            for start in range(0, N, batch_size):
                end   = min(start + batch_size, N)
                x     = torch.LongTensor(X_seq_np[start:end]).to(DEVICE)
                mask  = torch.BoolTensor(X_mask_np[start:end]).to(DEVICE)
                stats = torch.FloatTensor(X_stats_np[start:end]).to(DEVICE)

                if variant == "drop_first":
                    x2 = x.clone(); m2 = mask.clone()
                    for j in range(len(x2)):
                        l = int((~mask[j]).sum())
                        if l > 1:
                            x2[j, :l-1] = x[j, 1:l]; x2[j, l-1] = 0
                            m2[j, :l-1] = False;       m2[j, l-1] = True
                    x, mask = x2, m2
                elif variant == "drop_last":
                    x2 = x.clone(); m2 = mask.clone()
                    for j in range(len(x2)):
                        l = int((~mask[j]).sum())
                        if l > 1:
                            x2[j, l-1] = 0; m2[j, l-1] = True
                    x, mask = x2, m2

                for _ in range(mc_passes):
                    accumulate(m, x, mask, stats, tw / mc_passes, start, end, train_mode=True)

    return np.stack([p.argmax(axis=1) for p in prob_sum], axis=1)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("STEP 1: Load v2 models for pseudo-labeling", flush=True)
    print("=" * 60, flush=True)
    v2_models = load_v2_models()

    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Generate SOFT pseudo-labels on test set", flush=True)
    print("=" * 60, flush=True)
    X_test_seq   = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
    X_test_mask  = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
    X_test_stats = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)

    pl_seq, pl_mask, pl_stats, pl_hard, pl_soft, _ = generate_soft_pseudo_labels(
        v2_models, X_test_seq, X_test_mask, X_test_stats, conf_threshold=0.99
    )

    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Build combined dataset", flush=True)
    print("=" * 60, flush=True)
    X_tr   = np.load(f"{FEATURE_PATH}/X_train_seq.npy")
    M_tr   = np.load(f"{FEATURE_PATH}/X_train_mask.npy")
    S_tr   = np.load(f"{FEATURE_PATH}/X_train_stats.npy").astype(np.float32)
    y_tr   = np.load(f"{FEATURE_PATH}/y_train.npy")

    X_va   = np.load(f"{FEATURE_PATH}/X_val_seq.npy")
    M_va   = np.load(f"{FEATURE_PATH}/X_val_mask.npy")
    S_va   = np.load(f"{FEATURE_PATH}/X_val_stats.npy").astype(np.float32)
    y_va   = np.load(f"{FEATURE_PATH}/y_val.npy")

    # Hard labels for real data (one-hot-like) + real label soft version
    n_real = len(X_tr) + len(X_va)

    X_all = np.concatenate([X_tr, X_va, pl_seq], axis=0)
    M_all = np.concatenate([M_tr, M_va, pl_mask], axis=0)
    S_all = np.concatenate([S_tr, S_va, pl_stats], axis=0)
    y_all = np.concatenate([y_tr, y_va, pl_hard], axis=0)

    # Soft labels: real data gets near-one-hot (label smoothing 0.05), pseudo gets soft probs
    smooth = 0.05
    y_soft_all = []
    for i, nc in enumerate(NUM_CLASSES):
        s = np.full((len(X_all), nc), smooth / (nc - 1), dtype=np.float32)
        # Real data: put 1-smooth on correct class
        for j in range(n_real):
            s[j, y_all[j, i]] = 1.0 - smooth
        # Pseudo-labeled data: use actual soft probs
        s[n_real:] = pl_soft[i]
        y_soft_all.append(s)

    num_stat_features = S_tr.shape[1]
    print(f"Dataset: {len(X_all)} total (real={n_real}, pseudo={len(pl_seq)})", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Multi-seed training (6 seeds × 5 folds + 6 full = 36 models)", flush=True)
    print("=" * 60, flush=True)
    saved = run_training(X_all, M_all, S_all, y_all, y_soft_all, num_stat_features)

    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Assemble final ensemble (v3 + v2 + v1)", flush=True)
    print("=" * 60, flush=True)

    all_models, all_weights = [], []

    # v3 models
    for path, f1 in saved:
        ckpt = torch.load(path, map_location=DEVICE)
        m = build_model(ckpt.get("num_stat_features", num_stat_features))
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(DEVICE)
        all_models.append(m)
        all_weights.append(float(f1))

    # v2 models (reuse, lower weight since v3 is better)
    for m in v2_models:
        all_models.append(m)
        all_weights.append(0.995)

    print(f"Total models: {len(all_models)} | weight range: [{min(all_weights):.4f}, {max(all_weights):.4f}]", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 6: MC dropout + TTA final prediction", flush=True)
    print("=" * 60, flush=True)

    preds = predict_with_mc_tta(
        all_models, all_weights,
        X_test_seq, X_test_mask, X_test_stats,
        mc_passes=15, batch_size=256
    )

    # Decode
    decoded = np.zeros_like(preds)
    for col in range(6):
        inv_map = {v: k for k, v in ENCODERS[col].items()}
        decoded[:, col] = np.vectorize(inv_map.get)(preds[:, col])

    test_df = pd.read_csv("data/layer1_raw/X_test.csv")
    result  = pd.DataFrame(decoded, columns=["attr_1","attr_2","attr_3","attr_4","attr_5","attr_6"])
    result.insert(0, "id", test_df["id"])
    result.to_csv("submission_v3.csv", index=False)
    print(f"\n✅ Saved submission_v3.csv  shape={result.shape}", flush=True)
    print(result.head(5).to_string(index=False), flush=True)

    # Val eval on original val set using first 5 v3 fold models
    print("\n--- Val eval (v3 fold models, original val set) ---", flush=True)
    from run_improved import AugDataset
    val_ds = AugDataset(X_va, M_va, S_va, y_va, augment=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    v3_fold_models = [m for m, (p, _) in zip(all_models, zip(saved, all_weights))
                      if "fold" in str(p) and "seed42" in str(p)][:5]
    if len(v3_fold_models) >= 5:
        val_res = evaluate_model(v3_fold_models, val_loader, DEVICE)
        print(f"Exact Match: {val_res['exact_match_accuracy']:.4f}", flush=True)
        print(f"Macro F1:    {val_res['macro_f1_score']:.4f}", flush=True)
        for i, f1 in enumerate(val_res["f1_per_attribute"]):
            print(f"  attr_{i+1}: {f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
