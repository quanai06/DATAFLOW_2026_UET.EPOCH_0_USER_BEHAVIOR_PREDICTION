import argparse
import torch

from src.training.train_transformer import train_transformer
from src.metrics.metrics import evaluate_model
from src.data.loaders import get_transformer_loaders
from src.models.transformer_model import TransformerModel
from src.training.train_transformer import train_full_model

def transformer():

    config = {
        "feature_path": "data/layer3_features/transformer",
        "batch_size": 64,
        "epochs": 10,
        "vocab_size": 25000,
        "num_classes_list": [12, 31, 99, 12, 31, 99],

        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_ff": 128,
        "max_len": 37,

        "n_splits": 5
    }

    # ======================
    # Train KFold
    # ======================
    train_transformer(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================
    # Load data
    # ======================
    train_loader, val_loader = get_transformer_loaders(config)

    # ======================
    # Load all fold models
    # ======================
    models = []

    for fold in range(config["n_splits"]):

        ckpt = torch.load(
            f"models/transformer/transformer_fold_{fold}.pt",
            map_location=device
        )

        model = TransformerModel(
            vocab_size=config["vocab_size"],
            num_classes_list=config["num_classes_list"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_ff=config["dim_ff"],
            max_len=config["max_len"]
        )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        models.append(model)

    # ======================
    # Evaluate ensemble
    # ======================
    train_results = evaluate_model(models, train_loader, device)
    val_results = evaluate_model(models, val_loader, device)

    print("\n===== TRAIN METRICS =====")
    print(train_results)

    print("\n===== VAL METRICS =====")
    print(val_results)
    
    # ======================
    # Train FULL model
    # ======================
    train_full_model(config)
    
    # ======================
    # Load FULL model
    # ======================
    ckpt = torch.load(
        "models/transformer/transformer_full.pt",
        map_location=device
    )
    
    model_full = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=config["num_classes_list"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"]
    )
    
    model_full.load_state_dict(ckpt["model_state_dict"])
    model_full.to(device)
    model_full.eval()
    
    # ======================
    # Evaluate FULL
    # ======================
    train_results_full = evaluate_model(model_full, train_loader, device)
    val_results_full = evaluate_model(model_full, val_loader, device)
    
    print("\n===== FULL MODEL TRAIN METRICS =====")
    print(train_results_full)
    
    print("\n===== FULL MODEL VAL METRICS =====")
    print(val_results_full)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.model == "transformer":
        transformer()
    else:
        raise ValueError("Unknown model")


if __name__ == "__main__":
    main()