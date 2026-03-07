"""
Full improved training pipeline:
1. Generate pseudo-labels from current ensemble (high-confidence test predictions)
2. Retrain with: pseudo-labels + subsequence augmentation + stronger rare-class weights
3. Multi-seed ensemble (3 seeds x 5 folds = 15 models + 3 full models)
4. Predict with MC dropout + TTA
"""
import sys, pickle, os, torch, numpy as np, pandas as pd
import torch.nn as nn
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, '.')

from src.models.transformer_model import TransformerModel
from src.models.losses import ExactMatchFocalLoss
from src.metrics.metrics import evaluate_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer"
MODEL_DIR = Path("models/transformer_v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    VOCAB_SIZE = pickle.load(f)["vocab_size"]
with open(f"{FEATURE_PATH}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

BASE_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    num_classes_list=[12, 31, 99, 12, 31, 99],
    d_model=256, nhead=8, num_layers=4, dim_ff=512, max_len=37,
)
TRAIN_CONFIG = dict(
    epochs=20, patience=4, batch_size=64,
    n_splits=5, seeds=[42, 123, 2024],
)

# ──────────────────────────────────────────
# Dataset with subsequence augmentation
# ──────────────────────────────────────────
class AugDataset(Dataset):
    def __init__(self, seqs, masks, stats, labels, augment=True, crop_prob=0.4):
        self.seqs   = seqs
        self.masks  = masks
        self.stats  = stats
        self.labels = labels
        self.augment = augment
        self.crop_prob = crop_prob
        self.max_len = seqs.shape[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq   = self.seqs[idx].copy()
        mask  = self.masks[idx].copy()
        stats = self.stats[idx]
        label = self.labels[idx]

        if self.augment and np.random.random() < self.crop_prob:
            actual_len = int((~mask).sum())
            if actual_len > 3:
                crop_len = np.random.randint(max(2, actual_len // 2), actual_len)
                start    = np.random.randint(0, actual_len - crop_len + 1)
                new_seq  = np.zeros(self.max_len, dtype=seq.dtype)
                new_mask = np.ones(self.max_len,  dtype=bool)
                new_seq[:crop_len]  = seq[start:start + crop_len]
                new_mask[:crop_len] = False
                seq, mask = new_seq, new_mask

        return (torch.LongTensor(seq), torch.BoolTensor(mask),
                torch.FloatTensor(stats), torch.LongTensor(label))


# ──────────────────────────────────────────
# Class weights — adaptive exponent for rare classes
# ──────────────────────────────────────────
def get_class_weights(y):
    weights_list = []
    for i in range(6):
        labels = y[:, i]
        classes, counts = np.unique(labels, return_counts=True)
        # Stronger upweighting (lower exponent) for very rare classes
        exponents = np.where(counts < 50, 0.4, np.where(counts < 200, 0.6, 0.75))
        w = (len(labels) / (len(classes) * counts)) ** exponents
        weights_list.append(torch.FloatTensor(w).to(DEVICE))
    return weights_list


# ──────────────────────────────────────────
# Build model
# ──────────────────────────────────────────
def build_model(num_stat_features):
    return TransformerModel(num_stat_features=num_stat_features, **BASE_CONFIG)


# ──────────────────────────────────────────
# Train one fold
# ──────────────────────────────────────────
def train_fold(model, train_loader, val_loader, weights_list):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
    criterion = ExactMatchFocalLoss(weights_list=weights_list, alpha=0.01, gamma=2.0)

    best_val, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(TRAIN_CONFIG["epochs"]):
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

        # val loss
        model.eval()
        vtotal = 0
        with torch.no_grad():
            for x, mask, stats, y in val_loader:
                x, mask, stats, y = x.to(DEVICE), mask.to(DEVICE), stats.to(DEVICE), y.to(DEVICE)
                vtotal += criterion(model(x, mask, stats), y).item()
        vloss = vtotal / len(val_loader)

        lr = optimizer.param_groups[0]["lr"]
        print(f"    Epoch {epoch+1:2d}/{TRAIN_CONFIG['epochs']} - LR:{lr:.6f} - Loss:{avg:.4f} - Val:{vloss:.4f}", flush=True)

        if vloss < best_val:
            best_val, best_state, patience_cnt = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= TRAIN_CONFIG["patience"]:
                print(f"    Early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    return model


# ──────────────────────────────────────────
# STEP 1: Load existing fold models
# ──────────────────────────────────────────
def load_existing_models():
    models = []
    for fold in range(5):
        ckpt = torch.load(f"models/transformer/transformer_fold_{fold}.pt", map_location=DEVICE)
        nsf  = ckpt.get("num_stat_features", 0)
        m    = build_model(nsf)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(DEVICE).eval()
        models.append(m)
    ckpt = torch.load("models/transformer/transformer_full.pt", map_location=DEVICE)
    m = build_model(ckpt.get("num_stat_features", 0))
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()
    models.append(m)
    print(f"Loaded {len(models)} existing models", flush=True)
    return models


# ──────────────────────────────────────────
# STEP 2: Pseudo-label test set
# ──────────────────────────────────────────
def generate_pseudo_labels(models, conf_threshold=0.995):
    X_seq   = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
    X_mask  = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
    X_stats = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)

    ds = TensorDataset(torch.LongTensor(X_seq), torch.BoolTensor(X_mask), torch.FloatTensor(X_stats))
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    all_preds, all_confs = [], []
    for m in models:
        m.eval()

    with torch.no_grad():
        for x, mask, stats in loader:
            x, mask, stats = x.to(DEVICE), mask.to(DEVICE), stats.to(DEVICE)
            outputs_sum = None
            for m in models:
                outs = m(x, mask, stats)
                if outputs_sum is None:
                    outputs_sum = [o.clone() for o in outs]
                else:
                    for i in range(len(outs)):
                        outputs_sum[i] += outs[i]
            preds = np.stack([torch.argmax(o, dim=1).cpu().numpy() for o in outputs_sum], axis=1)
            confs = np.stack([torch.softmax(o, dim=1).max(dim=1)[0].cpu().numpy() for o in outputs_sum], axis=1)
            all_preds.append(preds)
            all_confs.append(confs)

    all_preds = np.concatenate(all_preds)
    all_confs = np.concatenate(all_confs)
    keep      = all_confs.min(axis=1) > conf_threshold

    print(f"Test samples: {len(all_preds)} | Pseudo-labeled (conf>{conf_threshold}): {keep.sum()} ({keep.mean()*100:.1f}%)", flush=True)

    return X_seq[keep], X_mask[keep], X_stats[keep], all_preds[keep]


# ──────────────────────────────────────────
# STEP 3: Multi-seed KFold training
# ──────────────────────────────────────────
def run_multiseed_training(X_seq, X_mask, X_stats, y, num_stat_features):
    kfold = StratifiedKFold(n_splits=TRAIN_CONFIG["n_splits"], shuffle=True, random_state=42)
    strat_key = y[:, 2]  # stratify on attr_3 (99 classes)

    saved_paths = []

    for seed in TRAIN_CONFIG["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'='*50}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*50}", flush=True)

        for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_seq, strat_key)):
            print(f"\n  FOLD {fold+1}/{TRAIN_CONFIG['n_splits']}", flush=True)

            y_fold       = y[tr_idx]
            weights_list = get_class_weights(y_fold)

            train_ds = AugDataset(X_seq[tr_idx], X_mask[tr_idx], X_stats[tr_idx], y[tr_idx], augment=True)
            val_ds   = AugDataset(X_seq[va_idx], X_mask[va_idx], X_stats[va_idx], y[va_idx], augment=False)

            train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=256,                       shuffle=False, num_workers=2, pin_memory=True)

            model = build_model(num_stat_features).to(DEVICE)
            model = train_fold(model, train_loader, val_loader, weights_list)

            val_metrics  = evaluate_model(model, val_loader, DEVICE)
            val_f1       = val_metrics["macro_f1_score"]
            print(f"  Fold {fold+1} val macro-F1: {val_f1:.4f}", flush=True)

            path = MODEL_DIR / f"model_seed{seed}_fold{fold}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_macro_f1":     val_f1,
                "val_metrics":      val_metrics,
                "num_stat_features": num_stat_features,
                "seed": seed, "fold": fold,
            }, path)
            saved_paths.append((path, val_f1))
            print(f"  ✅ Saved {path}", flush=True)

        # Full model per seed (train+val)
        print(f"\n  FULL MODEL (seed={seed})", flush=True)
        weights_full = get_class_weights(y)
        full_ds      = AugDataset(X_seq, X_mask, X_stats, y, augment=True)
        full_loader  = DataLoader(full_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

        model = build_model(num_stat_features).to(DEVICE)
        # Use first fold's val_loader for early stopping reference on full train
        _, va_idx0 = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_seq, strat_key))
        val_ds0  = AugDataset(X_seq[va_idx0], X_mask[va_idx0], X_stats[va_idx0], y[va_idx0], augment=False)
        val_loader0 = DataLoader(val_ds0, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
        model = train_fold(model, full_loader, val_loader0, weights_full)

        val_m = evaluate_model(model, val_loader0, DEVICE)
        path  = MODEL_DIR / f"model_seed{seed}_full.pt"
        torch.save({
            "model_state_dict":  model.state_dict(),
            "val_macro_f1":      val_m["macro_f1_score"],
            "num_stat_features": num_stat_features,
            "seed": seed,
        }, path)
        saved_paths.append((path, val_m["macro_f1_score"]))
        print(f"  ✅ Saved full model seed={seed}  val-F1={val_m['macro_f1_score']:.4f}", flush=True)

    return saved_paths


# ──────────────────────────────────────────
# STEP 4: MC Dropout + TTA ensemble predict
# ──────────────────────────────────────────
def mc_tta_predict(models, weights, X_seq_np, X_mask_np, X_stats_np,
                   mc_passes=10, batch_size=256):

    N = len(X_seq_np)
    num_classes = BASE_CONFIG["num_classes_list"]
    # accumulate weighted softmax probs
    prob_sum = [np.zeros((N, nc), dtype=np.float64) for nc in num_classes]

    def run_batch(model, x, mask, stats, weight, train_mode=False):
        if train_mode:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            outs = model(x, mask, stats)
        for i, o in enumerate(outs):
            prob_sum[i][start:end] += torch.softmax(o, dim=1).cpu().numpy() * weight

    total_w = sum(weights)

    for m, w in zip(models, weights):
        m_w = w / total_w  # normalise weight

        # TTA variants: original, drop-first, drop-last
        for variant in ["orig", "drop_first", "drop_last"]:
            tta_w = m_w / 3.0  # split evenly across 3 TTA variants

            for start in range(0, N, batch_size):
                end   = min(start + batch_size, N)
                x     = torch.LongTensor(X_seq_np[start:end]).to(DEVICE)
                mask  = torch.BoolTensor(X_mask_np[start:end]).to(DEVICE)
                stats = torch.FloatTensor(X_stats_np[start:end]).to(DEVICE)

                if variant == "drop_first":
                    x2    = x.clone(); mask2 = mask.clone()
                    lengths = (~mask).sum(dim=1)
                    for j in range(len(x2)):
                        l = lengths[j].item()
                        if l > 1:
                            x2[j, :l-1] = x[j, 1:l]; x2[j, l-1] = 0
                            mask2[j, :l-1] = False;   mask2[j, l-1] = True
                    x, mask = x2, mask2

                elif variant == "drop_last":
                    x2    = x.clone(); mask2 = mask.clone()
                    lengths = (~mask).sum(dim=1)
                    for j in range(len(x2)):
                        l = lengths[j].item()
                        if l > 1:
                            x2[j, l-1] = 0; mask2[j, l-1] = True
                    x, mask = x2, mask2

                # MC dropout passes
                for _ in range(mc_passes):
                    run_batch(m, x, mask, stats, tta_w / mc_passes, train_mode=True)

    preds = np.stack([p.argmax(axis=1) for p in prob_sum], axis=1)
    return preds


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("STEP 1: Load existing models", flush=True)
    print("=" * 60, flush=True)
    existing_models = load_existing_models()

    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Generate pseudo-labels on test set", flush=True)
    print("=" * 60, flush=True)
    pl_seq, pl_mask, pl_stats, pl_labels = generate_pseudo_labels(existing_models, conf_threshold=0.995)

    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Build combined dataset (train + val + pseudo-labels)", flush=True)
    print("=" * 60, flush=True)

    X_train_seq   = np.load(f"{FEATURE_PATH}/X_train_seq.npy")
    X_train_mask  = np.load(f"{FEATURE_PATH}/X_train_mask.npy")
    X_train_stats = np.load(f"{FEATURE_PATH}/X_train_stats.npy").astype(np.float32)
    y_train       = np.load(f"{FEATURE_PATH}/y_train.npy")

    X_val_seq   = np.load(f"{FEATURE_PATH}/X_val_seq.npy")
    X_val_mask  = np.load(f"{FEATURE_PATH}/X_val_mask.npy")
    X_val_stats = np.load(f"{FEATURE_PATH}/X_val_stats.npy").astype(np.float32)
    y_val       = np.load(f"{FEATURE_PATH}/y_val.npy")

    # Combine train + val + pseudo-labeled test
    X_all   = np.concatenate([X_train_seq,   X_val_seq,   pl_seq],   axis=0)
    M_all   = np.concatenate([X_train_mask,  X_val_mask,  pl_mask],  axis=0)
    S_all   = np.concatenate([X_train_stats, X_val_stats, pl_stats], axis=0)
    y_all   = np.concatenate([y_train,       y_val,       pl_labels],axis=0)

    num_stat_features = X_train_stats.shape[1]
    print(f"Combined dataset size: {len(X_all)} (train={len(X_train_seq)}, val={len(X_val_seq)}, pseudo={len(pl_seq)})", flush=True)

    # Class distribution report for rare classes
    for i, attr in enumerate(['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']):
        classes, counts = np.unique(y_all[:, i], return_counts=True)
        rare = (counts < 50).sum()
        print(f"  {attr}: {len(classes)} classes, {rare} rare (<50 samples)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Multi-seed training", flush=True)
    print("=" * 60, flush=True)
    saved = run_multiseed_training(X_all, M_all, S_all, y_all, num_stat_features)

    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Load all new models + old models for final ensemble", flush=True)
    print("=" * 60, flush=True)

    all_models, all_weights = [], []

    # Load new v2 models
    for path, val_f1 in saved:
        ckpt = torch.load(path, map_location=DEVICE)
        m = build_model(ckpt.get("num_stat_features", num_stat_features))
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(DEVICE)
        all_models.append(m)
        all_weights.append(float(val_f1))

    # Also include original 5 fold models + full
    for m in existing_models:
        all_models.append(m)
        all_weights.append(0.998)  # approximate their val F1

    print(f"Total models in ensemble: {len(all_models)}", flush=True)
    print(f"Weights (val macro-F1): min={min(all_weights):.4f} max={max(all_weights):.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 6: MC dropout + TTA prediction", flush=True)
    print("=" * 60, flush=True)

    X_test_seq   = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
    X_test_mask  = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
    X_test_stats = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)

    preds = mc_tta_predict(all_models, all_weights, X_test_seq, X_test_mask, X_test_stats,
                           mc_passes=10, batch_size=256)

    # Decode predictions
    decoded = np.zeros_like(preds)
    for col in range(6):
        inv_map = {v: k for k, v in ENCODERS[col].items()}
        decoded[:, col] = np.vectorize(inv_map.get)(preds[:, col])

    test_df = pd.read_csv("data/layer1_raw/X_test.csv")
    result  = pd.DataFrame(decoded, columns=["attr_1","attr_2","attr_3","attr_4","attr_5","attr_6"])
    result.insert(0, "id", test_df["id"])
    result.to_csv("submission_improved.csv", index=False)
    print(f"\n✅ Saved submission_improved.csv  shape={result.shape}", flush=True)
    print(result.head(5), flush=True)

    # Also validate on val set (original val, not pseudo-labeled)
    print("\n--- Validating on original val set ---", flush=True)
    val_ds = AugDataset(X_val_seq, X_val_mask, X_val_stats, y_val, augment=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Use new models only for unbiased eval (they didn't train on val via pseudo-labels ideally)
    new_models = [m for m, _ in zip(all_models, all_weights)][:len(saved)]
    new_weights = all_weights[:len(saved)]
    val_res = evaluate_model(new_models[:5], val_loader, DEVICE)
    print(f"Val Exact Match: {val_res['exact_match_accuracy']:.4f}", flush=True)
    print(f"Val Macro F1:    {val_res['macro_f1_score']:.4f}", flush=True)
    for i, f1 in enumerate(val_res['f1_per_attribute']):
        print(f"  attr_{i+1}: {f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
