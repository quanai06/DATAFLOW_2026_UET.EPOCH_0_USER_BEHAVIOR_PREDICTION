import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_transformer_loaders(config):

    path = config["feature_path"]

    X_train = np.load(f"{path}/X_train_seq.npy")
    mask_train = np.load(f"{path}/X_train_mask.npy")
    y_train = np.load(f"{path}/y_train.npy")

    X_val = np.load(f"{path}/X_val_seq.npy")
    mask_val = np.load(f"{path}/X_val_mask.npy")
    y_val = np.load(f"{path}/y_val.npy")

    train_ds = TensorDataset(
        torch.LongTensor(X_train),
        torch.BoolTensor(mask_train),
        torch.LongTensor(y_train),
    )

    val_ds = TensorDataset(
        torch.LongTensor(X_val),
        torch.BoolTensor(mask_val),
        torch.LongTensor(y_val),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader