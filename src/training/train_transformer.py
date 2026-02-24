import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

from src.models.transformer_model import TransformerModel

def train_transformer(config):

    feature_path = config["feature_path"]

    # ========================
    # Load features
    # ========================
    X_train = np.load(f"{feature_path}/X_train_seq.npy")
    mask_train = np.load(f"{feature_path}/X_train_mask.npy")
    y_train = np.load(f"{feature_path}/y_train.npy")
    # print("X max:", X_train.max())
    # print("Vocab size:", config["vocab_size"])

    dataset = TensorDataset(
        torch.LongTensor(X_train),
        torch.BoolTensor(mask_train),
        torch.LongTensor(y_train),   
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,   # CPU safe
    )

    # ========================
    # Model
    # ========================
    model = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=config["num_classes_list"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterions = [
        nn.CrossEntropyLoss()
        for _ in range(6)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ========================
    # Training
    # ========================

    print("y shape:", y_train.shape)
    print("max per column:", y_train.max(axis=0))
    print("min per column:", y_train.min(axis=0))
    for epoch in range(config["epochs"]):

        model.train()
        total_loss = 0

        for x_batch, mask_batch, y_batch in loader:

            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch, mask_batch)

            loss = 0

            for i in range(6):
                loss += criterions[i](outputs[i], y_batch[:, i])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{config['epochs']}  Loss: {avg_loss:.4f}")

    # ========================
    # Save model
    # ========================
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "transformer_model.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "num_classes_list": config["num_classes_list"],
    }, model_path)

    print(f"✅ Model saved to {model_path}")

    return model