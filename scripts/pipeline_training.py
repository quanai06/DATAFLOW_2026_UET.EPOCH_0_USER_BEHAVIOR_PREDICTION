import argparse
import torch

from src.training.train_transformer import train_transformer

from src.metrics.metrics import evaluate_model
from src.data.loaders import get_transformer_loaders


def transformer():

    config = {
        "feature_path": "data/layer3_features/transformer",
        "batch_size": 64,
        "epochs": 10,
        "vocab_size": 25000,
        "num_classes_list": [12, 31, 99, 12, 31, 99],

        "d_model": 64,
        "nhead": 2,
        "num_layers": 2,
        "dim_ff": 128,
        "max_len": 37
    }

    # ======================
    # Train
    # ======================
    model = train_transformer(config)

    # ======================
    # Device
    # ======================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ======================
    # Load data for eval
    # ======================
    train_loader, val_loader = get_transformer_loaders(config)

    # ======================
    # Evaluate
    # ======================
    train_results = evaluate_model(model, train_loader, device)
    val_results = evaluate_model(model, val_loader, device)

    print("\n===== TRAIN METRICS =====")
    print(train_results)

    print("\n===== VAL METRICS =====")
    print(val_results)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="transformer | lstm | mlp",
    )

    args = parser.parse_args()

    if args.model == "transformer":
        transformer()

    elif args.model == "lstm":
        print("LSTM pipeline not implemented")

    elif args.model == "mlp":
        print("MLP pipeline not implemented")

    else:
        raise ValueError("Unknown model")


if __name__ == "__main__":
    main()