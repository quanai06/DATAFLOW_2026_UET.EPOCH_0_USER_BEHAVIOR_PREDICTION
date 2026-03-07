"""
Final submission: v3 transformers (36) + BiLSTM (30) = 66 models
Best val result: EM=0.9985, F1=0.9989
"""
import sys, pickle, os, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, '.')

from src.models.transformer_model import TransformerModel
from run_lstm_parallel import BiLSTMAttention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer"
NUM_CLASSES = [12, 31, 99, 12, 31, 99]

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f)
    VOCAB_SIZE = remap["vocab_size"]

with open(f"{FEATURE_PATH}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

print("Loading test data...", flush=True)
X_te  = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
M_te  = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
S_te  = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)

X_seq_t   = torch.LongTensor(X_te).to(DEVICE)
X_mask_t  = torch.BoolTensor(M_te).to(DEVICE)
X_stats_t = torch.FloatTensor(S_te).to(DEVICE)
N = len(X_te)
print(f"Test samples: {N}", flush=True)

# ── Accumulate weighted logits ────────────────────────────────────────────────
logits_sum = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]
total_weight = 0.0

BATCH = 1024

def infer(model, x, mask, stats):
    model.eval()
    outs = [np.zeros((N, nc), dtype=np.float32) for nc in NUM_CLASSES]
    with torch.no_grad():
        for start in range(0, N, BATCH):
            xb  = x[start:start+BATCH]
            mb  = mask[start:start+BATCH]
            sb  = stats[start:start+BATCH]
            res = model(xb, mb, sb)
            for i, r in enumerate(res):
                outs[i][start:start+BATCH] = r.cpu().float().numpy()
    return outs

# ── Load v3 transformer models ────────────────────────────────────────────────
V3_DIR = Path("models/transformer_v3")
v3_files = sorted(V3_DIR.glob("*.pt"))
print(f"\nLoading {len(v3_files)} v3 transformer models...", flush=True)

for i, path in enumerate(v3_files):
    ckpt = torch.load(path, map_location=DEVICE)
    w = float(ckpt.get("val_macro_f1", 1.0))
    num_stat = ckpt.get("num_stat_features", 0)

    model = TransformerModel(
        vocab_size=VOCAB_SIZE + 1,  # +1 for MASK token
        num_classes_list=NUM_CLASSES,
        d_model=384, nhead=8, num_layers=4, dim_ff=768, max_len=37,
        num_stat_features=num_stat,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    outs = infer(model, X_seq_t, X_mask_t, X_stats_t)
    for j in range(6):
        logits_sum[j] += outs[j] * w
    total_weight += w
    del model
    print(f"  [{i+1}/{len(v3_files)}] {path.name}  w={w:.4f}", flush=True)

# ── Load BiLSTM models ────────────────────────────────────────────────────────
BILSTM_DIR = Path("models/bilstm")
bilstm_files = sorted(BILSTM_DIR.glob("*.pt"))
print(f"\nLoading {len(bilstm_files)} BiLSTM models...", flush=True)

for i, path in enumerate(bilstm_files):
    ckpt = torch.load(path, map_location=DEVICE)
    w = float(ckpt.get("val_macro_f1", 1.0))
    num_stat = ckpt.get("num_stat_features", 0)

    model = BiLSTMAttention(
        vocab_size=VOCAB_SIZE,
        num_classes_list=NUM_CLASSES,
        embed_dim=256, hidden=256, num_layers=2, dropout=0.3,
        num_stat_features=num_stat,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    outs = infer(model, X_seq_t, X_mask_t, X_stats_t)
    for j in range(6):
        logits_sum[j] += outs[j] * w
    total_weight += w
    del model
    print(f"  [{i+1}/{len(bilstm_files)}] {path.name}  w={w:.4f}", flush=True)

# ── Decode predictions ────────────────────────────────────────────────────────
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
result.to_csv("submission_best.csv", index=False)
print("Saved submission_best.csv", flush=True)
print(result.head())
