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

    X_seq = np.load(f"{feature_path}/X_test_seq.npy")
    X_mask = np.load(f"{feature_path}/X_test_mask.npy")

    return torch.LongTensor(X_seq), torch.BoolTensor(X_mask)


def ensemble_predict(models, X_seq, X_mask, device):

    for m in models:
        m.eval()

    with torch.no_grad():

        X_seq = X_seq.to(device)
        X_mask = X_mask.to(device)

        outputs_sum = None
        
        weights = [1, 1, 1, 1, 1, 0.5]
        
        for m, w in zip(models, weights):

            outputs = m(X_seq, X_mask)

            if outputs_sum is None:
                outputs_sum = outputs
            else:
                for i in range(len(outputs)):
                    outputs_sum[i] += outputs[i] * w;

        preds = []

        for out in outputs_sum:
            preds.append(torch.argmax(out, dim=1).cpu().numpy())

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_path = "data/layer3_features/transformer"

    # ======================
    # Load encoders
    # ======================
    with open(f"{feature_path}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        
    # ======================
    # Load config from fold0
    # ======================
    ckpt = torch.load("models/transformer/transformer_fold_0.pt", map_location=device)
    config = ckpt["config"]

    num_classes_list = ckpt["num_classes_list"]

    # ======================
    # Load all fold models
    # ======================
    models = []

    n_splits = config.get("n_splits", 5)

    for fold in range(n_splits):

        ckpt = torch.load(
            f"models/transformer/transformer_fold_{fold}.pt",
            map_location=device
        )

        model = TransformerModel(
            vocab_size=config["vocab_size"],
            num_classes_list=num_classes_list,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_ff=config["dim_ff"],
            max_len=config["max_len"]
        )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        models.append(model)


    # ======================
    # Load FULL model
    # ======================
    ckpt = torch.load(
        "models/transformer/transformer_full.pt",
        map_location=device
    )
    
    model_full = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=num_classes_list,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"]
    )
    
    model_full.load_state_dict(ckpt["model_state_dict"])
    model_full.to(device)
    
    models.append(model_full) 
    # models = [model_full]
    
    # ======================
    # Load test features
    # ======================
    X_seq, X_mask = load_test_data(feature_path)

    preds = ensemble_predict(models, X_seq, X_mask, device)

    preds = decode_predictions(preds, encoders)

    # ======================
    # Create submission
    # ======================
    test_df = pd.read_csv("data/layer1_raw/X_test.csv")

    result = pd.DataFrame(
        preds,
        columns=[
            "attr_1",
            "attr_2",
            "attr_3",
            "attr_4",
            "attr_5",
            "attr_6",
        ],
    )

    result.insert(0, "id", test_df["id"])

    result.to_csv("submission.csv", index=False)

    print("✅ Saved submission.csv")

def predict_test_combine():
    # 1. CONFIGURATION (Phải khớp với Train)
    # ======================================================
    CFG = {
        'SEED': 42,
        'MAX_LEN': 37,
        'N_ENSEMBLE': 12,
        'TARGET_COLS': ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6'],
        'MODEL_DIR': 'models/combine',
        'DATA_PATH': 'data',
        'SUBMISSION_FILE': 'submission_combine_reproduced.csv'
    }
    
    # 2. LOAD DATA
    # ======================================================
    print("⏳ [PREDICT] Loading Test Data...")
    # Chỉ cần load file X_test (Layer 2 đã có feature)
    df_test = pd.read_csv(f"{CFG['DATA_PATH']}/layer2/X_test.csv")
    
    # 3. PREPROCESS TEST DATA
    # ======================================================
    print("⚙️ Processing Test Data...")
    
    # A. Sequence
    X_test_seq = tc.process_sequences_val(tc.get_sequences(df_test), CFG['MAX_LEN'])
    
    # B. Manual Stats
    X_test_stats = tc.get_stats(df_test)
    
    # C. Scale (Load Scaler Full từ file)
    print("   Loading Scaler...")
    try:
        scaler_full = joblib.load(f"{CFG['MODEL_DIR']}/scaler_full.pkl")
        X_test_stats_sc = scaler_full.transform(X_test_stats)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy 'scaler_full.pkl'. Hãy chạy train_main.py trước!")
        return

    # 4. LOAD MODELS & PREDICT
    # ======================================================
    print("\n🚀 Starting Ensemble Prediction...")
    
    all_models_preds = []
    model_files = sorted([f for f in os.listdir(CFG['MODEL_DIR']) if f.endswith('.keras')])
    
    if len(model_files) == 0:
        print(f"❌ Lỗi: Không tìm thấy model nào trong {CFG['MODEL_DIR']}")
        return

    for i, m_file in enumerate(model_files):
        print(f"   Using model {i+1}/{len(model_files)}: {m_file}...", end="\r")
        
        # Load model
        model_path = os.path.join(CFG['MODEL_DIR'], m_file)
        # Đổi đuôi h5 thành keras và thêm safe_mode=False
        model_path = f'{CFG["MODEL_DIR"]}/lstm_model_{i}.keras' 
        model = tf.keras.models.load_model(model_path, safe_mode=False)
                
        # Predict
        preds = model.predict([X_test_seq, X_test_stats_sc], verbose=0)
        all_models_preds.append(preds)
        
        # Clear RAM
        tf.keras.backend.clear_session()
        
    print("\n   ✅ All models executed.")

    # 5. VOTING & SUBMISSION
    # ======================================================
    print("⏳ Voting & Decoding...")
    submission = {'id': df_test['id']}

    for i, col in enumerate(CFG['TARGET_COLS']):
        # Load Encoder
        le = joblib.load(f"{CFG['MODEL_DIR']}/encoder_{col}.pkl")
        
        # Average Probabilities (Soft Voting)
        col_preds = [m[i] for m in all_models_preds]
        avg_probs = np.mean(col_preds, axis=0)
        
        # Get Label
        pred_labels = np.argmax(avg_probs, axis=1)
        
        # Inverse Transform
        submission[col] = le.inverse_transform(pred_labels)

    # Save
    sub_df = pd.DataFrame(submission)
    sub_df.to_csv('submission_combine_seed42_final.csv', index=False)
    print(f"\n🏆 DONE! Submission saved to: submission_combine_seed42_final.csv")

def predict_test_lstm():
    MODEL_DIR = "models/lstm"
    N_ENSEMBLE = 10
    target_cols = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6']

    print("⏳ Loading Pre-trained objects & Test data...")
    df_test = pd.read_csv('data/layer2/X_test.csv')
    encoders = joblib.load(f'{MODEL_DIR}/encoders.pkl')
    scaler = joblib.load(f'{MODEL_DIR}/scaler.pkl')

    # 1. Preprocess Test Data
    X_test_seq = tl.process_sequences_val(tl.get_sequences(df_test), max_len=37)
    X_test_stats = scaler.transform(tl.get_stats(df_test))

    # 2. Ensemble Inference
    all_preds = [] # Để lưu kết quả của từng model
    print(f"🚀 Đang dự đoán với {N_ENSEMBLE} models...")

    for i in range(N_ENSEMBLE):
        model_path = f'models/lstm/lstm_model_{i}.keras' 
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, safe_mode=False)
        else:
            print(f"❌ Không tìm thấy file: {model_path}")
            # Nếu vẫn dùng đuôi cũ .h5 thì thử:
            # model_path = f'models/lstm/lstm_model_{i}.h5'
            # model = tf.keras.models.load_model(model_path, safe_mode=False)
        preds = model.predict([X_test_seq, X_test_stats], verbose=0)
        all_preds.append(preds)
        print(f"✅ Model {i+1} xong.")

    # 3. Soft Voting (Trung bình xác suất)
    submission = {'id': df_test['id']}

    for idx, col in enumerate(target_cols):
        # all_preds[model_index][output_index]
        col_probs = np.mean([model_pred[idx] for model_pred in all_preds], axis=0)
        final_labels = np.argmax(col_probs, axis=1)
        submission[col] = encoders[col].inverse_transform(final_labels)

    # 4. Xuất File
    sub_df = pd.DataFrame(submission)
    sub_df.to_csv('submission_final_lstm.csv', index=False)
    print("\n🔥 ĐÃ XUẤT FILE: submission_final_lstm.csv")
    print(sub_df.head())

if __name__ == "__main__":
    # main()
    # predict_test_combine()
    predict_test_lstm()
