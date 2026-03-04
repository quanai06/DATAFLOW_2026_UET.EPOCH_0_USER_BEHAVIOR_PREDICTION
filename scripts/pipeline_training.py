import argparse
import torch
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.training import train_combine as tc

from src.training.train_transformer import train_transformer
from src.metrics.metrics import evaluate_model
from src.data.loaders import get_transformer_loaders
from src.models.transformer_model import TransformerModel
from src.training.train_transformer import train_full_model

def transformer():

    config = {
        "feature_path": "data/layer3_features/transformer",
        "batch_size": 64,
        "epochs": 10,
        "vocab_size": 25000,
        "num_classes_list": [12, 31, 99, 12, 31, 99],

        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_ff": 128,
        "max_len": 37,

        "n_splits": 5
    }

    # ======================
    # Train KFold
    # ======================
    train_transformer(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================
    # Load data
    # ======================
    train_loader, val_loader = get_transformer_loaders(config)

    # ======================
    # Load all fold models
    # ======================
    models = []

    for fold in range(config["n_splits"]):

        ckpt = torch.load(
            f"models/transformer/transformer_fold_{fold}.pt",
            map_location=device
        )

        model = TransformerModel(
            vocab_size=config["vocab_size"],
            num_classes_list=config["num_classes_list"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_ff=config["dim_ff"],
            max_len=config["max_len"]
        )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        models.append(model)

    # ======================
    # Evaluate ensemble
    # ======================
    train_results = evaluate_model(models, train_loader, device)
    val_results = evaluate_model(models, val_loader, device)

    print("\n===== TRAIN METRICS =====")
    print(train_results)

    print("\n===== VAL METRICS =====")
    print(val_results)
    
    # ======================
    # Train FULL model
    # ======================
    train_full_model(config)
    
    # ======================
    # Load FULL model
    # ======================
    ckpt = torch.load(
        "models/transformer/transformer_full.pt",
        map_location=device
    )
    
    model_full = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=config["num_classes_list"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"]
    )
    
    model_full.load_state_dict(ckpt["model_state_dict"])
    model_full.to(device)
    model_full.eval()
    
    # ======================
    # Evaluate FULL
    # ======================
    train_results_full = evaluate_model(model_full, train_loader, device)
    val_results_full = evaluate_model(model_full, val_loader, device)
    
    print("\n===== FULL MODEL TRAIN METRICS =====")
    print(train_results_full)
    
    print("\n===== FULL MODEL VAL METRICS =====")
    print(val_results_full)

def combine_lstm_gru_cnn():
    # 1. CONFIGURATION
    # ======================================================
    CFG = {
        'SEED': 42,
        'MAX_LEN': 37,
        'EMBEDDING_DIM': 128,
        'LSTM_UNITS': 256,
        'GRU_UNITS': 256,
        'CNN_FILTERS': 256,
        'CNN_KERNEL_SIZE': 3,
        'BATCH_SIZE': 128,
        'EPOCHS': 18,
        'N_ENSEMBLE': 12, # 4 LSTM + 4 GRU + 4 CNN
        'TARGET_COLS': ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6'],
        'MODEL_DIR': 'models/combine',
        'DATA_PATH': 'data'
    }
    os.makedirs(CFG['MODEL_DIR'], exist_ok=True)
    tc.seed_everything(CFG['SEED'])
    # 2. PREPARE DATA
    # ======================================================
    # Load Data
    df_train, df_val, df_test, X_full, df_y_train, df_y_val, Y_full = tc.load_data(CFG['DATA_PATH'])
    # --- Process Sequences ---
    print("⚙️ Processing Sequences...")
    X_train_seq = tc.process_sequences_val(tc.get_sequences(df_train), CFG['MAX_LEN'])
    X_val_seq   = tc.process_sequences_val(tc.get_sequences(df_val), CFG['MAX_LEN'])
    X_test_seq  = tc.process_sequences_val(tc.get_sequences(df_test), CFG['MAX_LEN'])
    X_full_seq  = tc.process_sequences_val(tc.get_sequences(X_full), CFG['MAX_LEN'])

    VOCAB_SIZE = max(np.max(X_train_seq), np.max(X_val_seq), np.max(X_test_seq)) + 1
    print(f"   Vocab Size: {VOCAB_SIZE}")

    # --- Process Manual Stats ---
    print("⚙️ Processing Manual Stats...")
    X_train_stats = tc.get_stats(df_train)
    X_val_stats   = tc.get_stats(df_val)
    X_test_stats  = tc.get_stats(df_test)
    X_full_stats  = tc.get_stats(X_full)

    # --- Scale Stats (Phase 1: Fit on Train) ---
    scaler_p1 = StandardScaler()
    scaler_p1.fit(X_train_stats)
    joblib.dump(scaler_p1, f"{CFG['MODEL_DIR']}/scaler_phase1.pkl")

    X_train_stats_sc = scaler_p1.transform(X_train_stats)
    X_val_stats_sc   = scaler_p1.transform(X_val_stats)
    NUM_WIDE_FEATURES = X_train_stats.shape[1]
    print(f"   Num features: {NUM_WIDE_FEATURES}")

    # --- Encode Labels ---
    print("⚙️ Encoding Labels...")
    encoders = tc.fit_encode_labels(Y_full, CFG['TARGET_COLS'])
    # Save encoders
    for col, le in encoders.items():
        joblib.dump(le, f"{CFG['MODEL_DIR']}/encoder_{col}.pkl")

    y_train_list = tc.transform_labels(df_y_train, CFG['TARGET_COLS'], encoders)
    y_val_list   = tc.transform_labels(df_y_val, CFG['TARGET_COLS'], encoders)
    y_full_encoded_list = tc.transform_labels(Y_full, CFG['TARGET_COLS'], encoders)

    # ======================================================
    # 3. PHASE 1: CHECK F1 (VALIDATION)
    # ======================================================
    print("\n" + "="*40)
    print("🔍 PHASE 1: CHECK VALIDATION F1")
    print("="*40)

    check_model = tc.build_model('lstm', CFG, VOCAB_SIZE, NUM_WIDE_FEATURES, encoders, CFG['TARGET_COLS'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    check_model.fit(
        x=[X_train_seq, X_train_stats_sc],
        y=y_train_list,
        validation_data=([X_val_seq, X_val_stats_sc], y_val_list),
        epochs=12, # Ít epoch để check nhanh
        batch_size=CFG['BATCH_SIZE'],
        callbacks=[early_stop],
        verbose=1
    )

    # Eval F1
    val_preds = check_model.predict([X_val_seq, X_val_stats_sc], verbose=0)
    f1_scores = []
    print("\n📊 Validation Scores:")
    for i, col in enumerate(CFG['TARGET_COLS']):
        p = np.argmax(val_preds[i], axis=1)
        t = y_val_list[i]
        score = f1_score(t, p, average='macro')
        f1_scores.append(score)
        print(f"   {col}: {score:.4f}")
    print(f"🔥 Macro F1 Avg: {np.mean(f1_scores):.5f}")

    # ======================================================
    # 4. PHASE 2: FULL TRAIN & ENSEMBLE
    # ======================================================
    print("\n" + "="*40)
    print("🚀 PHASE 2: FULL TRAINING & ENSEMBLE")
    print("="*40)

    # Re-fit scaler on Full Data
    scaler_full = StandardScaler()
    scaler_full.fit(X_full_stats)
    joblib.dump(scaler_full, f"{CFG['MODEL_DIR']}/scaler_full.pkl")

    X_full_stats_sc = scaler_full.transform(X_full_stats)
    X_test_stats_sc = scaler_full.transform(X_test_stats)

    all_models_preds = []

    for i in range(CFG['N_ENSEMBLE']):
        # Reseed per model
        current_seed = CFG['SEED'] + i
        tc.seed_everything(current_seed)
        tf.keras.backend.clear_session()
        
        # Select Model Architecture (4 LSTM - 4 GRU - 4 CNN)
        if i < 4: m_type = 'lstm'
        elif i < 8: m_type = 'gru'
        else: m_type = 'cnn'
        
        print(f"\n🔄 Training Model {i+1}/{CFG['N_ENSEMBLE']} [{m_type.upper()}] (Seed {current_seed})...")
        
        model = tc.build_model(m_type, CFG, VOCAB_SIZE, NUM_WIDE_FEATURES, encoders, CFG['TARGET_COLS'])
        
        model.fit(
            x=[X_full_seq, X_full_stats_sc],
            y=y_full_encoded_list,
            epochs=CFG['EPOCHS'],
            batch_size=CFG['BATCH_SIZE'],
            verbose=0, # Silent mode
            shuffle=True
        )
        
        # Save Model
        model_path = f"{CFG['MODEL_DIR']}/model_{i}_{m_type}.keras"
        model.save(model_path)
        
        # Predict Test
        preds = model.predict([X_test_seq, X_test_stats_sc], verbose=0)
        all_models_preds.append(preds)

    # ======================================================
    # 5. SUBMISSION
    # ======================================================
    print("\n⏳ Voting & Creating Submission...")
    submission = {'id': df_test['id']}

    for i, col in enumerate(CFG['TARGET_COLS']):
        # Average Probabilities
        col_preds = [m[i] for m in all_models_preds]
        avg_probs = np.mean(col_preds, axis=0)
        pred_labels = np.argmax(avg_probs, axis=1)
        
        submission[col] = encoders[col].inverse_transform(pred_labels)

    sub_df = pd.DataFrame(submission)
    filename = f"submission_combine_seed{CFG['SEED']}_final.csv"
    sub_df.to_csv(filename, index=False)

    print(f"\n🏆 DONE! File saved to: {filename}")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.model == "transformer":
        transformer()
    else:
        raise ValueError("Unknown model")


if __name__ == "__main__":
    # main()
    combine_lstm_gru_cnn()