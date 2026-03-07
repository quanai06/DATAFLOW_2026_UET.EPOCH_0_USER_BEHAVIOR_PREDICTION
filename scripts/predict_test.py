import torch
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from src.training import train_combine as tc
import joblib
import os
from src.training import train_lstm as tl

from src.models.transformer_model import TransformerModel


def load_test_data(feature_path):
    X_seq  = np.load(f"{feature_path}/X_test_seq.npy")
    X_mask = np.load(f"{feature_path}/X_test_mask.npy")

    stats_path = f"{feature_path}/X_test_stats.npy"
    X_stats = np.load(stats_path).astype(np.float32) if os.path.exists(stats_path) else None

    return torch.LongTensor(X_seq), torch.BoolTensor(X_mask), X_stats


def ensemble_predict(models, X_seq, X_mask, X_stats, device, fold_weights=None):
    for m in models:
        m.eval()

    X_seq  = X_seq.to(device)
    X_mask = X_mask.to(device)
    stats_tensor = torch.FloatTensor(X_stats).to(device) if X_stats is not None else None

    if fold_weights is None:
        fold_weights = [1.0] * len(models)

    weight_sum = sum(fold_weights)

    with torch.no_grad():
        outputs_sum = None

        for m, w in zip(models, fold_weights):
            outputs = m(X_seq, X_mask, stats_tensor)

            if outputs_sum is None:
                outputs_sum = [o * (w / weight_sum) for o in outputs]
            else:
                for i in range(len(outputs)):
                    outputs_sum[i] += outputs[i] * (w / weight_sum)

        preds = [torch.argmax(o, dim=1).cpu().numpy() for o in outputs_sum]
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
    """Transformer ensemble prediction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_path = "data/layer3_features/transformer"

    with open(f"{feature_path}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    # Auto-detect vocab_size
    remapper_path = f"{feature_path}/action_remapper.pkl"
    if os.path.exists(remapper_path):
        with open(remapper_path, "rb") as f:
            remap_data = pickle.load(f)
        vocab_size = remap_data["vocab_size"]
    else:
        vocab_size = 25000

    ckpt0 = torch.load("models/transformer/transformer_fold_0.pt", map_location=device)
    config = ckpt0["config"]
    config["vocab_size"] = vocab_size  # use correct vocab_size
    num_classes_list = ckpt0["num_classes_list"]

    # Load all fold models
    models = []
    fold_weights = []
    n_splits = config.get("n_splits", 5)

    for fold in range(n_splits):
        ckpt = torch.load(f"models/transformer/transformer_fold_{fold}.pt", map_location=device)
        num_stat_features = ckpt.get("num_stat_features", 0)
        val_macro_f1 = float(ckpt.get("val_macro_f1", 1.0))

        model = TransformerModel(
            vocab_size=vocab_size,
            num_classes_list=num_classes_list,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_ff=config["dim_ff"],
            max_len=config["max_len"],
            num_stat_features=num_stat_features,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        models.append(model)
        fold_weights.append(val_macro_f1)

    # Load FULL model
    ckpt_full = torch.load("models/transformer/transformer_full.pt", map_location=device)
    num_stat_features = ckpt_full.get("num_stat_features", 0)
    full_weight = float(ckpt_full.get("val_macro_f1", np.mean(fold_weights) if len(fold_weights) > 0 else 1.0))

    model_full = TransformerModel(
        vocab_size=vocab_size,
        num_classes_list=num_classes_list,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"],
        num_stat_features=num_stat_features,
    )
    model_full.load_state_dict(ckpt_full["model_state_dict"])
    model_full.to(device)
    models.append(model_full)

    # Use validation macro-F1 weights for folds; default full model weight to their mean
    fold_weights.append(full_weight)
    print("Ensemble weights (folds + full):", fold_weights)

    X_seq, X_mask, X_stats = load_test_data(feature_path)
    preds = ensemble_predict(models, X_seq, X_mask, X_stats, device, fold_weights=fold_weights)
    preds = decode_predictions(preds, encoders)

    test_df = pd.read_csv("data/layer1_raw/X_test.csv")
    result = pd.DataFrame(preds, columns=["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"])
    result.insert(0, "id", test_df["id"])
    result.to_csv("submission.csv", index=False)
    print("✅ Saved submission.csv")


def predict_test_combine():
    CFG = {
        'SEED': 42,
        'MAX_LEN': 37,
        'N_ENSEMBLE': 12,
        'TARGET_COLS': ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6'],
        'MODEL_DIR': 'models/combine',
        'DATA_PATH': 'data',
    }

    print("⏳ [PREDICT] Loading Test Data...")
    df_test = pd.read_csv(f"{CFG['DATA_PATH']}/layer2/X_test.csv")

    print("⚙️ Processing Test Data...")
    X_test_seq   = tc.process_sequences_val(tc.get_sequences(df_test), CFG['MAX_LEN'])
    X_test_stats = tc.get_stats(df_test)

    print("   Loading Scaler...")
    scaler_full = joblib.load(f"{CFG['MODEL_DIR']}/scaler_full.pkl")
    X_test_stats_sc = scaler_full.transform(X_test_stats)

    print("\n🚀 Starting Ensemble Prediction...")
    all_models_preds = []
    model_files = sorted([f for f in os.listdir(CFG['MODEL_DIR']) if f.endswith('.keras')])

    if len(model_files) == 0:
        print(f"❌ No models found in {CFG['MODEL_DIR']}")
        return

    for i, m_file in enumerate(model_files):
        print(f"   Using model {i+1}/{len(model_files)}: {m_file}...", end="\r")
        model_path = os.path.join(CFG['MODEL_DIR'], m_file)
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        preds = model.predict([X_test_seq, X_test_stats_sc], verbose=0)
        all_models_preds.append(preds)
        tf.keras.backend.clear_session()

    print("\n   ✅ All models executed.")

    print("⏳ Voting & Decoding...")
    submission = {'id': df_test['id']}

    for i, col in enumerate(CFG['TARGET_COLS']):
        le = joblib.load(f"{CFG['MODEL_DIR']}/encoder_{col}.pkl")
        col_preds = [m[i] for m in all_models_preds]
        avg_probs = np.mean(col_preds, axis=0)
        pred_labels = np.argmax(avg_probs, axis=1)
        submission[col] = le.inverse_transform(pred_labels)

    sub_df = pd.DataFrame(submission)
    sub_df.to_csv('submission_combine_seed42_final.csv', index=False)
    print("🏆 DONE! Saved: submission_combine_seed42_final.csv")


def predict_test_lstm():
    MODEL_DIR = "models/lstm"
    N_ENSEMBLE = 10
    target_cols = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6']

    print("⏳ Loading Pre-trained objects & Test data...")
    df_test  = pd.read_csv('data/layer2/X_test.csv')
    encoders = joblib.load(f'{MODEL_DIR}/encoders.pkl')
    scaler   = joblib.load(f'{MODEL_DIR}/scaler.pkl')

    X_test_seq   = tl.process_sequences_val(tl.get_sequences(df_test), max_len=37)
    X_test_stats = scaler.transform(tl.get_stats(df_test))

    all_preds = []
    print(f"🚀 Predicting with {N_ENSEMBLE} models...")

    for i in range(N_ENSEMBLE):
        model_path = f'models/lstm/lstm_model_{i}.keras'
        if not os.path.exists(model_path):
            print(f"❌ Not found: {model_path}")
            continue
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        preds = model.predict([X_test_seq, X_test_stats], verbose=0)
        all_preds.append(preds)
        print(f"✅ Model {i+1} done.")

    submission = {'id': df_test['id']}
    for idx, col in enumerate(target_cols):
        col_probs = np.mean([m[idx] for m in all_preds], axis=0)
        final_labels = np.argmax(col_probs, axis=1)
        submission[col] = encoders[col].inverse_transform(final_labels)

    sub_df = pd.DataFrame(submission)
    sub_df.to_csv('submission_final_lstm.csv', index=False)
    print("\n🔥 Saved: submission_final_lstm.csv")
    print(sub_df.head())


if __name__ == "__main__":
    main()
