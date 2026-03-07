"""
Inference-only script for v3 transformer models.
Loads saved checkpoints from models/transformer_v3/, runs MC dropout + TTA,
and saves submission_v3.csv.
"""
import sys, pickle, torch, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '.')

from src.models.transformer_model import TransformerModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer"
NUM_CLASSES = [12, 31, 99, 12, 31, 99]
MC_PASSES = 15
BATCH = 256

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f)
    VOCAB_SIZE = remap["vocab_size"]

with open(f"{FEATURE_PATH}/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)

BASE_CONFIG = dict(
    vocab_size=VOCAB_SIZE + 1,  # +1 for MASK token
    num_classes_list=NUM_CLASSES,
    d_model=384, nhead=8, num_layers=4, dim_ff=768, max_len=37,
)

# Load test data
print("Loading test data...", flush=True)
X_test_seq   = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
X_test_mask  = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
X_test_stats = np.load(f"{FEATURE_PATH}/X_test_stats.npy").astype(np.float32)
N = len(X_test_seq)
print(f"Test samples: {N}", flush=True)

# Load v3 models
V3_DIR = Path("models/transformer_v3")
v3_files = sorted(V3_DIR.glob("*.pt"))
print(f"Found {len(v3_files)} v3 model checkpoints", flush=True)

models = []
weights = []
for path in v3_files:
    ckpt = torch.load(path, map_location=DEVICE)
    f1 = float(ckpt.get("val_macro_f1", 1.0))
    m = TransformerModel(
        num_stat_features=ckpt.get("num_stat_features", 0),
        **BASE_CONFIG
    ).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    models.append(m)
    weights.append(f1)
    print(f"  {path.name}  f1={f1:.4f}", flush=True)

# MC dropout + TTA prediction
print(f"\nRunning MC dropout ({MC_PASSES} passes) + TTA (orig/drop_first/drop_last)...", flush=True)
prob_sum = [np.zeros((N, nc), dtype=np.float64) for nc in NUM_CLASSES]
total_w = sum(weights)

for m, w in zip(models, weights):
    mw = w / total_w
    for variant in ["orig", "drop_first", "drop_last"]:
        tw = mw / 3.0
        for start in range(0, N, BATCH):
            end   = min(start + BATCH, N)
            x     = torch.LongTensor(X_test_seq[start:end]).to(DEVICE)
            mask  = torch.BoolTensor(X_test_mask[start:end]).to(DEVICE)
            stats = torch.FloatTensor(X_test_stats[start:end]).to(DEVICE)

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

            for _ in range(MC_PASSES):
                m.train()  # enable dropout for MC
                with torch.no_grad():
                    outs = m(x, mask, stats)
                for i, o in enumerate(outs):
                    prob_sum[i][start:end] += torch.softmax(o, dim=1).cpu().numpy() * (tw / MC_PASSES)

preds = np.stack([p.argmax(axis=1) for p in prob_sum], axis=1)

# Decode predictions
decoded = np.zeros_like(preds)
for col in range(6):
    inv_map = {v: k for k, v in ENCODERS[col].items()}
    decoded[:, col] = np.vectorize(inv_map.get)(preds[:, col])

test_df = pd.read_csv("data/layer1_raw/X_test.csv")
result = pd.DataFrame(decoded, columns=["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"])
result.insert(0, "id", test_df["id"])
result.to_csv("submission_v3.csv", index=False)
print(f"\nSaved submission_v3.csv  shape={result.shape}", flush=True)
print(result.head(5).to_string(index=False))
