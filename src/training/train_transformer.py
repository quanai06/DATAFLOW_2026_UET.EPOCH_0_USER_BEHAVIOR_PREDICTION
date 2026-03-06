import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

from src.models.transformer_model import TransformerModel

from src.models.losses import ExactMatchFocalLoss


def get_class_weights(y_train):
    weights_list = []
    for i in range(6):
        labels = y_train[:, i]
        classes, counts = np.unique(labels, return_counts=True)
        # Thay vì tính tuyến tính, ta dùng lũy thừa 0.75 hoặc 0.8
        # Điều này giúp trọng số cao nhưng không quá cực đoan gây nổ loss
        weights = (len(labels) / (len(classes) * counts)) ** 0.75
        weights_list.append(torch.FloatTensor(weights))
    return weights_list

def train_one_fold(model, loader, device, config, weights_list=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # criterions = [
    #     nn.CrossEntropyLoss()
    #     for _ in range(6)
    # ]
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = ExactMatchFocalLoss(weights_list=weights_list, alpha=0.01, gamma=2.0)

    for epoch in range(config["epochs"]):

        model.train()
        total_loss = 0

        for x_batch, mask_batch, y_batch in loader:

            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch, mask_batch)
            loss = criterion(outputs, y_batch)

            # loss = 0
            # for i in range(6):
            #     loss += criterions[i](outputs[i], y_batch[:, i])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        # print(f"   Epoch {epoch+1}/{config['epochs']}  Loss: {avg_loss:.4f}")
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch+1}/{config['epochs']} - LR: {current_lr:.6f} - Loss: {avg_loss:.4f}")


def train_transformer(config):

    feature_path = config["feature_path"]
    n_splits = config.get("n_splits", 5)

    # ========================
    # Load data
    # ========================
    X = np.load(f"{feature_path}/X_train_seq.npy")
    mask = np.load(f"{feature_path}/X_train_mask.npy")
    y = np.load(f"{feature_path}/y_train.npy")

    dataset = TensorDataset(
        torch.LongTensor(X),
        torch.BoolTensor(mask),
        torch.LongTensor(y),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========================
    # KFold
    # ========================
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    model_dir = Path("models/transformer")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Starting KFold training...")
    print("Total samples:", len(dataset))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):

        print(f"\n========== FOLD {fold+1}/{n_splits} ==========")

        y_train_fold = y[train_idx]
        weights_list = get_class_weights(y_train_fold)
        weights_list = [w.to(device) for w in weights_list] # Đưa lên GPU
        
        train_subset = Subset(dataset, train_idx)

        loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
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

        model.to(device)

        # ========================
        # Train
        # ========================
        train_one_fold(model, loader, device, config, weights_list=weights_list)

        # ========================
        # Save
        # ========================
        model_path = model_dir / f"transformer_fold_{fold}.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "num_classes_list": config["num_classes_list"],
        }, model_path)

        print(f"✅ Saved {model_path}")

    print("\n🎉 KFold training completed!")


    return None
    
def train_full_model(config):

    feature_path = config["feature_path"]

    X = np.load(f"{feature_path}/X_train_seq.npy")
    mask = np.load(f"{feature_path}/X_train_mask.npy")
    y = np.load(f"{feature_path}/y_train.npy")
    
    weights_list = get_class_weights(y)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_list = [w.to(device) for w in weights_list]

    dataset = TensorDataset(
        torch.LongTensor(X),
        torch.BoolTensor(mask),
        torch.LongTensor(y),
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=config["num_classes_list"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"]
    )

    model.to(device)

    print("\n========== TRAIN FULL DATA ==========")

    train_one_fold(model, loader, device, config, weights_list=weights_list)

    Path("models").mkdir(exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "num_classes_list": config["num_classes_list"],
    }, "models/transformer/transformer_full.pt")

    print("✅ Saved models/transformer/transformer_full.pt")