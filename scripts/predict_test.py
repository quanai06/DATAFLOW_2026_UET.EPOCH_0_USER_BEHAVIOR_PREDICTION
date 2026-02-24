import torch
import numpy as np
import pandas as pd
import pickle

from src.models.transformer_model import TransformerModel


# =========================
# Load test features
# =========================
def load_test_data(feature_path):

    X_seq = np.load(f"{feature_path}/X_test_seq.npy")
    X_mask = np.load(f"{feature_path}/X_test_mask.npy")

    return torch.tensor(X_seq), torch.tensor(X_mask)


# =========================
# Decode predictions về label gốc
# =========================
def decode_predictions(preds, encoders):

    decoded_cols = []

    for i, mapping in enumerate(encoders):

        # mapping gốc: value -> index
        inverse_map = {v: k for k, v in mapping.items()}

        col = [inverse_map[idx] for idx in preds[:, i]]
        decoded_cols.append(col)

    decoded = np.stack(decoded_cols, axis=1)

    return decoded


# =========================
# Predict
# =========================
def predict(model, X_seq, X_mask, device):

    model.eval()

    preds = []

    with torch.no_grad():

        X_seq = X_seq.to(device)
        X_mask = X_mask.to(device)

        outputs = model(X_seq, X_mask)

        for out in outputs:
            preds.append(torch.argmax(out, dim=1).cpu().numpy())

    preds = np.stack(preds, axis=1)

    return preds


# =========================
# Main
# =========================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_path = "data/layer3_features/transformer"

    # ======================
    # Load encoders
    # ======================
    with open(f"{feature_path}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    num_classes_list = [len(e) for e in encoders]

    # ======================
    # Load model config
    # ======================
    checkpoint = torch.load("models/transformer_model.pt", map_location=device)
    config = checkpoint["config"]

    # ======================
    # Build model
    # ======================
    model = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=num_classes_list,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("✅ Model loaded")

    # ======================
    # Load test data
    # ======================
    X_seq, X_mask = load_test_data(feature_path)

    # ======================
    # Predict
    # ======================
    preds = predict(model, X_seq, X_mask, device)

    # ======================
    # Decode về label gốc
    # ======================
    preds_decoded = decode_predictions(preds, encoders)

    # ======================
    # Load ID
    # ======================
    test_df = pd.read_csv("data/layer1_raw/X_test.csv")

    result = pd.DataFrame(
        preds_decoded,
        columns=[
            "attr_1",
            "attr_2",
            "attr_3",
            "attr_4",
            "attr_5",
            "attr_6",
        ],
    )

    result.insert(0, "id", test_df["id"])

    result.to_csv("submission.csv", index=False)

    print("🎯 Saved submission.csv")


if __name__ == "__main__":
    main()