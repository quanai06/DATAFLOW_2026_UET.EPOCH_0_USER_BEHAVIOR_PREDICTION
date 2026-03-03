import torch
import numpy as np
import pandas as pd
import pickle

from src.models.transformer_model import TransformerModel


def load_test_data(feature_path):

    X_seq = np.load(f"{feature_path}/X_test_seq.npy")
    X_mask = np.load(f"{feature_path}/X_test_mask.npy")

    return torch.LongTensor(X_seq), torch.BoolTensor(X_mask)


def ensemble_predict(models, X_seq, X_mask, device):

    for m in models:
        m.eval()

    with torch.no_grad():

        X_seq = X_seq.to(device)
        X_mask = X_mask.to(device)

        outputs_sum = None
        
        weights = [1, 1, 1, 1, 1, 1]
        
        for m, w in zip(models, weights):

            outputs = m(X_seq, X_mask)

            if outputs_sum is None:
                outputs_sum = outputs
            else:
                for i in range(len(outputs)):
                    outputs_sum[i] += outputs[i] * w;

        preds = []

        for out in outputs_sum:
            preds.append(torch.argmax(out, dim=1).cpu().numpy())

        preds = np.stack(preds, axis=1)

    return preds


def decode_predictions(preds, encoders):

    decoded = np.zeros_like(preds)

    for col in range(preds.shape[1]):

        mapping = encoders[col]
        inv_map = {v: k for k, v in mapping.items()}

        decoded[:, col] = np.vectorize(inv_map.get)(preds[:, col])

    return decoded


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_path = "data/layer3_features/transformer"

    # ======================
    # Load encoders
    # ======================
    with open(f"{feature_path}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        
    # ======================
    # Load config from fold0
    # ======================
    ckpt = torch.load("models/transformer/transformer_fold_0.pt", map_location=device)
    config = ckpt["config"]

    num_classes_list = ckpt["num_classes_list"]

    # ======================
    # Load all fold models
    # ======================
    models = []

    n_splits = config.get("n_splits", 5)

    for fold in range(n_splits):

        ckpt = torch.load(
            f"models/transformer/transformer_fold_{fold}.pt",
            map_location=device
        )

        model = TransformerModel(
            vocab_size=config["vocab_size"],
            num_classes_list=num_classes_list,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_ff=config["dim_ff"],
            max_len=config["max_len"]
        )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        models.append(model)


    # ======================
    # Load FULL model
    # ======================
    ckpt = torch.load(
        "models/transformer/transformer_full.pt",
        map_location=device
    )
    
    model_full = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=num_classes_list,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"]
    )
    
    model_full.load_state_dict(ckpt["model_state_dict"])
    model_full.to(device)
    
    models.append(model_full) 
    # models = [model_full]
    
    # ======================
    # Load test features
    # ======================
    X_seq, X_mask = load_test_data(feature_path)

    preds = ensemble_predict(models, X_seq, X_mask, device)

    preds = decode_predictions(preds, encoders)

    # ======================
    # Create submission
    # ======================
    test_df = pd.read_csv("data/layer1_raw/X_test.csv")

    result = pd.DataFrame(
        preds,
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

    print("✅ Saved submission.csv")


if __name__ == "__main__":
    main()