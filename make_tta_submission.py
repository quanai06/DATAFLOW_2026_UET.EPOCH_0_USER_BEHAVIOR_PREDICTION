"""
TTA (Test-Time Augmentation) on v3 transformer models.
Each model runs N_TTA inference passes with random token masking on test data.
Logits averaged across passes and models.
"""
import sys, pickle, torch, numpy as np, pandas as pd
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, '.')

from src.models.transformer_model import TransformerModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer"
NUM_CLASSES = [12, 31, 99, 12, 31, 99]
N_TTA = 8          # augmented passes per model (+ 1 clean pass = 9 total)
MASK_PROB = 0.10   # lighter mask for TTA (don't destroy too much signal)
BATCH = 512

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f)
    VOCAB_SIZE = remap["vocab_size"]
MASK_TOKEN = VOCAB_SIZE  # same as training

with open(f"{FEATURE_PATH}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

print("Loading test data...", flush=True)
X_te = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
M_te = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
S_te = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)
N = len(X_te)
print(f"Test samples: {N}", flush=True)


def augment_seq(seq_np, mask_np, mask_prob):
    """Apply random token masking to a batch. Returns masked LongTensor."""
    seq = seq_np.copy()
    for i in range(len(seq)):
        actual_len = int((~mask_np[i]).sum())
        if actual_len > 1:
            n_mask = max(1, int(actual_len * mask_prob))
            pos = np.random.choice(actual_len, size=n_mask, replace=False)
            seq[i, pos] = MASK_TOKEN
    return torch.LongTensor(seq).to(DEVICE)


def infer_tta(model, x_np, mask_np, stats_t, n_tta, mask_prob):
    """Run clean pass + n_tta augmented passes, return averaged logits."""
    model.eval()
    mask_t  = torch.BoolTensor(mask_np).to(DEVICE)
    acc = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]

    # Clean pass (no augmentation)
    with torch.no_grad():
        for start in range(0, N, BATCH):
            xb = torch.LongTensor(x_np[start:start+BATCH]).to(DEVICE)
            mb = mask_t[start:start+BATCH]
            sb = stats_t[start:start+BATCH]
            res = model(xb, mb, sb)
            for i, r in enumerate(res):
                acc[i][start:start+BATCH] += r.cpu().float().numpy()

    # Augmented passes
    for _ in range(n_tta):
        x_aug = augment_seq(x_np, mask_np, mask_prob)
        with torch.no_grad():
            for start in range(0, N, BATCH):
                xb = x_aug[start:start+BATCH]
                mb = mask_t[start:start+BATCH]
                sb = stats_t[start:start+BATCH]
                res = model(xb, mb, sb)
                for i, r in enumerate(res):
                    acc[i][start:start+BATCH] += r.cpu().float().numpy()

    passes = 1 + n_tta
    return [acc[i] / passes for i in range(6)]


stats_t = torch.FloatTensor(S_te).to(DEVICE)
logits_sum = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]
total_weight = 0.0

V3_DIR = Path("models/transformer_v3")
v3_files = sorted(V3_DIR.glob("*.pt"))
print(f"\nRunning TTA ({N_TTA} aug + 1 clean) on {len(v3_files)} v3 models...", flush=True)

for i, path in enumerate(v3_files):
    ckpt = torch.load(path, map_location=DEVICE)
    w = float(ckpt.get("val_macro_f1", 1.0))
    num_stat = ckpt.get("num_stat_features", 0)

    model = TransformerModel(
        vocab_size=VOCAB_SIZE + 1,
        num_classes_list=NUM_CLASSES,
        d_model=384, nhead=8, num_layers=4, dim_ff=768, max_len=37,
        num_stat_features=num_stat,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    outs = infer_tta(model, X_te, M_te, stats_t, N_TTA, MASK_PROB)
    for j in range(6):
        logits_sum[j] += outs[j] * w
    total_weight += w
    del model
    print(f"  [{i+1}/{len(v3_files)}] {path.name}  w={w:.4f}", flush=True)

print(f"\nTotal weight: {total_weight:.4f}", flush=True)
preds = np.stack([np.argmax(logits_sum[j] / total_weight, axis=1) for j in range(6)], axis=1)

decoded = np.zeros_like(preds)
for col in range(6):
    mapping = ENCODERS[col]
    inv_map = {v: k for k, v in mapping.items()}
    decoded[:, col] = np.vectorize(inv_map.get)(preds[:, col])

test_df = pd.read_csv("data/layer1_raw/X_test.csv")
result = pd.DataFrame(decoded, columns=["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"])
result.insert(0, "id", test_df["id"])
result.to_csv("submission_tta.csv", index=False)
print("Saved submission_tta.csv", flush=True)
print(result.head())
