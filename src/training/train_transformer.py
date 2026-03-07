import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold  # Step 6: StratifiedKFold

from src.models.transformer_model import TransformerModel
from src.models.losses import ExactMatchFocalLoss
from src.metrics.metrics import evaluate_model


def get_class_weights(y_train):
    weights_list = []
    for i in range(6):
        labels = y_train[:, i]
        classes, counts = np.unique(labels, return_counts=True)
        weights = (len(labels) / (len(classes) * counts)) ** 0.75
        weights_list.append(torch.FloatTensor(weights))
    return weights_list


def train_one_fold(model, loader, val_loader, device, config, weights_list=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = ExactMatchFocalLoss(weights_list=weights_list, alpha=0.01, gamma=2.0)

    best_val_loss = float("inf")
    patience = config.get("patience", 3)
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):

        model.train()
        total_loss = 0

        for batch in loader:
            if len(batch) == 3:
                x_batch, mask_batch, y_batch = batch
                stats_batch = None
            else:
                x_batch, mask_batch, stats_batch, y_batch = batch
                stats_batch = stats_batch.to(device)

            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch, mask_batch, stats_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Step 11: val monitoring per fold for early stopping
        val_loss = avg_loss  # fallback if no val_loader
        if val_loader is not None:
            model.eval()
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        xv, mv, yv = batch
                        sv = None
                    else:
                        xv, mv, sv, yv = batch
                        sv = sv.to(device)
                    xv, mv, yv = xv.to(device), mv.to(device), yv.to(device)
                    outs = model(xv, mv, sv)
                    val_total += criterion(outs, yv).item()
            val_loss = val_total / len(val_loader)

        print(f"    Epoch {epoch+1}/{config['epochs']} - LR: {current_lr:.6f} - Loss: {avg_loss:.4f} - Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def _make_stratify_key(y):
    """Use attr_3 (col index 2, 99 classes) as stratification key — most imbalanced output."""
    return y[:, 2]


def train_transformer(config):

    feature_path = config["feature_path"]
    n_splits = config.get("n_splits", 5)

    X = np.load(f"{feature_path}/X_train_seq.npy")
    mask = np.load(f"{feature_path}/X_train_mask.npy")
    y = np.load(f"{feature_path}/y_train.npy")

    # Step 5: load stats if available
    stats_path = f"{feature_path}/X_train_stats.npy"
    has_stats = Path(stats_path).exists()
    if has_stats:
        stats = np.load(stats_path)
        dataset = TensorDataset(
            torch.LongTensor(X),
            torch.BoolTensor(mask),
            torch.FloatTensor(stats),
            torch.LongTensor(y),
        )
        num_stat_features = stats.shape[1]
    else:
        dataset = TensorDataset(
            torch.LongTensor(X),
            torch.BoolTensor(mask),
            torch.LongTensor(y),
        )
        num_stat_features = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 6: StratifiedKFold on combined label key
    strat_key = _make_stratify_key(y)
    # StratifiedKFold requires integer labels — encode the combined key
    unique_keys, encoded_keys = np.unique(strat_key, return_inverse=True)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_dir = Path("models/transformer")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Starting KFold training...")
    print("Total samples:", len(dataset))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, encoded_keys)):

        print(f"\n========== FOLD {fold+1}/{n_splits} ==========")

        y_train_fold = y[train_idx]
        weights_list = get_class_weights(y_train_fold)
        weights_list = [w.to(device) for w in weights_list]

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

        model = TransformerModel(
            vocab_size=config["vocab_size"],
            num_classes_list=config["num_classes_list"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_ff=config["dim_ff"],
            max_len=config["max_len"],
            num_stat_features=num_stat_features,
        )
        model.to(device)

        train_one_fold(model, train_loader, val_loader, device, config, weights_list=weights_list)

        # Compute validation metrics for weighting ensemble later
        val_metrics = evaluate_model(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1_score"]
        print(f"    Fold {fold+1} val macro-F1: {val_macro_f1:.4f}")

        model_path = model_dir / f"transformer_fold_{fold}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "num_classes_list": config["num_classes_list"],
            "num_stat_features": num_stat_features,
            "val_macro_f1": val_macro_f1,
            "val_metrics": val_metrics,
        }, model_path)

        print(f"✅ Saved {model_path}")

    print("\n🎉 KFold training completed!")
    return None


def train_full_model(config):

    feature_path = config["feature_path"]

    # Step 12: train full model on train+val combined
    X_train = np.load(f"{feature_path}/X_train_seq.npy")
    mask_train = np.load(f"{feature_path}/X_train_mask.npy")
    y_train = np.load(f"{feature_path}/y_train.npy")

    X_val = np.load(f"{feature_path}/X_val_seq.npy")
    mask_val = np.load(f"{feature_path}/X_val_mask.npy")
    y_val = np.load(f"{feature_path}/y_val.npy")

    X = np.concatenate([X_train, X_val], axis=0)
    mask = np.concatenate([mask_train, mask_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)

    stats_path = f"{feature_path}/X_train_stats.npy"
    stats_val_path = f"{feature_path}/X_val_stats.npy"
    has_stats = Path(stats_path).exists() and Path(stats_val_path).exists()
    if has_stats:
        stats_train = np.load(stats_path)
        stats_val = np.load(stats_val_path)
        stats = np.concatenate([stats_train, stats_val], axis=0)
        dataset = TensorDataset(
            torch.LongTensor(X),
            torch.BoolTensor(mask),
            torch.FloatTensor(stats),
            torch.LongTensor(y),
        )
        num_stat_features = stats.shape[1]
    else:
        dataset = TensorDataset(
            torch.LongTensor(X),
            torch.BoolTensor(mask),
            torch.LongTensor(y),
        )
        num_stat_features = 0

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_list = get_class_weights(y)
    weights_list = [w.to(device) for w in weights_list]

    model = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=config["num_classes_list"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"],
        num_stat_features=num_stat_features,
    )
    model.to(device)

    print("\n========== TRAIN FULL DATA (train+val) ==========")
    # No early stopping for full model — train all epochs
    train_one_fold(model, loader, None, device, config, weights_list=weights_list)

    Path("models").mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "num_classes_list": config["num_classes_list"],
        "num_stat_features": num_stat_features,
    }, "models/transformer/transformer_full.pt")

    print("✅ Saved models/transformer/transformer_full.pt")
