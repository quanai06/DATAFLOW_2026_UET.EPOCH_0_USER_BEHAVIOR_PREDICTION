"""Generate submission using v4 ensemble (48 models)."""
import sys, pickle, torch, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '.')
from src.models.transformer_model import TransformerModel

DEVICE = "cuda"
FP = "data/layer3_features/transformer"
NUM_CLASSES = [12, 31, 99, 12, 31, 99]
BATCH = 1024

with open(f"{FP}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f); VOCAB_SIZE = remap["vocab_size"]
with open(f"{FP}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

X_te = np.load(f"{FP}/X_test_seq.npy")
M_te = np.load(f"{FP}/X_test_mask.npy")
S_te = np.load(f"{FP}/X_test_stats.npy").astype(np.float32)
N = len(X_te)
X_t = torch.LongTensor(X_te).to(DEVICE)
M_t = torch.BoolTensor(M_te).to(DEVICE)
S_t = torch.FloatTensor(S_te).to(DEVICE)
print(f"Test: {N} samples", flush=True)

logits_sum = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]
total_w = 0.0

files = sorted(Path("models/transformer_v4").glob("*.pt"))
print(f"Loading {len(files)} v4 models...", flush=True)
for i, path in enumerate(files):
    ck = torch.load(path, map_location=DEVICE)
    w = float(ck.get("val_macro_f1", 1.0))
    m = TransformerModel(vocab_size=VOCAB_SIZE+1, num_classes_list=NUM_CLASSES,
        d_model=512, nhead=8, num_layers=6, dim_ff=1024, max_len=37,
        num_stat_features=ck.get("num_stat_features", 0)).to(DEVICE)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    with torch.no_grad():
        for s in range(0, N, BATCH):
            outs = m(X_t[s:s+BATCH], M_t[s:s+BATCH], S_t[s:s+BATCH])
            for j, o in enumerate(outs):
                logits_sum[j][s:s+BATCH] += o.cpu().float().numpy() * w
    total_w += w
    del m
    print(f"  [{i+1}/{len(files)}] {path.name} w={w:.4f}", flush=True)

preds = np.stack([np.argmax(logits_sum[j]/total_w, axis=1) for j in range(6)], axis=1)
decoded = np.zeros_like(preds)
for col in range(6):
    inv = {v: k for k, v in ENCODERS[col].items()}
    decoded[:, col] = np.vectorize(inv.get)(preds[:, col])

test_df = pd.read_csv("data/layer1_raw/X_test.csv")
result = pd.DataFrame(decoded, columns=["attr_1","attr_2","attr_3","attr_4","attr_5","attr_6"])
result.insert(0, "id", test_df["id"])
result.to_csv("submission_v4.csv", index=False)
print("Saved submission_v4.csv", flush=True)
print(result.head())
