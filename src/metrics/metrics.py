import torch
import numpy as np
from sklearn.metrics import f1_score # Thêm thư viện này

def exact_match(y_true, y_pred):
    return np.mean(np.all(y_true == y_pred, axis=1))

def evaluate_model(model, loader, device):
    is_ensemble = isinstance(model, list)

    if is_ensemble:
        for m in model: m.eval()
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
                        outputs_sum = [o.clone() for o in outputs]
                    else:
                        for i in range(len(outputs)):
                            outputs_sum[i] += outputs[i]
                outputs = outputs_sum
            else:
                outputs = model(x, mask)

            # Lấy argmax cho từng head trong 6 heads
            batch_preds = []
            for out in outputs:
                batch_preds.append(torch.argmax(out, dim=1).cpu().numpy())
            
            # Chuyển về shape (batch_size, 6)
            batch_preds = np.stack(batch_preds, axis=1)

            all_preds.append(batch_preds)
            all_labels.append(y.numpy() if isinstance(y, torch.Tensor) else y)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 1. Tính Overall Exact Match Accuracy
    em_score = exact_match(all_labels, all_preds)

    # 2. Tính F1-Score và Accuracy cho từng nhãn (6 heads)
    f1_per_col = []
    acc_per_col = [] # Thêm list này
    
    for i in range(6):
        # Tính F1
        col_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='macro')
        f1_per_col.append(col_f1)
        
        # Tính Accuracy cho từng nhãn (Exact Match của riêng nhãn đó)
        col_acc = np.mean(all_labels[:, i] == all_preds[:, i])
        acc_per_col.append(col_acc)
    
    avg_f1 = np.mean(f1_per_col)

    return {
        "exact_match_accuracy": float(em_score),
        "macro_f1_score": float(avg_f1),
        "f1_per_attribute": [round(f, 4) for f in f1_per_col],
        "acc_per_attribute": [round(a, 4) for a in acc_per_col] # Thêm dòng này
    }