import torch
import numpy as np


def exact_match(y_true, y_pred):
    return np.mean(np.all(y_true == y_pred, axis=1))


def evaluate_model(model, loader, device):

    # =====================
    # Allow ensemble
    # =====================
    is_ensemble = isinstance(model, list)

    if is_ensemble:
        for m in model:
            m.eval()
    else:
        model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, mask, y in loader:

            x = x.to(device)
            mask = mask.to(device)

            if is_ensemble:

                outputs_sum = None

                for m in model:
                    outputs = m(x, mask)

                    if outputs_sum is None:
                        outputs_sum = outputs
                    else:
                        for i in range(len(outputs)):
                            outputs_sum[i] += outputs[i]

                outputs = outputs_sum

            else:
                outputs = model(x, mask)

            preds = []

            for out in outputs:
                preds.append(torch.argmax(out, dim=1).cpu().numpy())

            preds = np.stack(preds, axis=1)

            all_preds.append(preds)
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    score = exact_match(all_labels, all_preds)

    return {
        "exact_match_accuracy": float(score)
    }