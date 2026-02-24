import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, mask, y in loader:

            x = x.to(device)
            mask = mask.to(device)

            outputs = model(x, mask)

            preds = []

            for out in outputs:
                p = torch.argmax(out, dim=1)
                preds.append(p)

            preds = torch.stack(preds, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # ======================
    # Exact Match Accuracy
    # ======================
    exact_match = np.mean(np.all(y_pred == y_true, axis=1))

    # ======================
    # Per-task accuracy
    # ======================
    task_acc = []

    for i in range(y_true.shape[1]):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        task_acc.append(acc)

    # ======================
    # Per-task F1
    # ======================
    task_f1 = []

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, i], y_pred[:, i], average="macro")
        task_f1.append(f1)

    results = {
        "exact_match": exact_match,
        "task_accuracy": task_acc,
        "task_f1": task_f1,
        "mean_task_acc": np.mean(task_acc),
        "mean_task_f1": np.mean(task_f1),
    }

    return results