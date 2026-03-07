import argparse
import torch
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.training import train_combine as tc
from src.training import train_lstm as tl

from src.training.train_transformer import train_transformer
from src.metrics.metrics import evaluate_model
from src.data.loaders import get_transformer_loaders
from src.models.transformer_model import TransformerModel
from src.training.train_transformer import train_full_model


def print_results(results, title):
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"Overall Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"Overall Macro F1-Score:       {results['macro_f1_score']:.4f}")
    print("-" * 50)
    print("F1-Score per Attribute (6 heads):")
    for i, f1 in enumerate(results['f1_per_attribute']):
        print(f"  Attr_{i+1}: {f1:.4f}")
    print(f"{'='*50}")


def transformer():

    config = {
        "feature_path": "data/layer3_features/transformer",
        "batch_size": 64,
        "epochs": 20,
        "patience": 4,
        "vocab_size": None,  # auto-detected from action_remapper
        "num_classes_list": [12, 31, 99, 12, 31, 99],
        "drop_out": 0.3,

        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_ff": 512,
        "max_len": 37,

        "n_splits": 5
    }

    # Auto-detect vocab_size from saved remapper
    import pickle
    remapper_path = f"{config['feature_path']}/action_remapper.pkl"
    if os.path.exists(remapper_path):
        with open(remapper_path, "rb") as f:
            remap_data = pickle.load(f)
        config["vocab_size"] = remap_data["vocab_size"]
        print(f"✅ vocab_size auto-detected: {config['vocab_size']}")
    else:
        config["vocab_size"] = 25000
        print("⚠️  action_remapper.pkl not found, using vocab_size=25000. Run build_transformer_features.py first.")

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

        num_stat_features = ckpt.get("num_stat_features", 0)

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

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        models.append(model)

    # ======================
    # Evaluate ensemble
    # ======================
    train_results = evaluate_model(models, train_loader, device)
    val_results = evaluate_model(models, val_loader, device)

    print_results(train_results, "TRAIN METRICS (ENSEMBLE)")
    print_results(val_results, "VAL METRICS (ENSEMBLE)")

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

    num_stat_features = ckpt.get("num_stat_features", 0)

    model_full = TransformerModel(
        vocab_size=config["vocab_size"],
        num_classes_list=config["num_classes_list"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"],
        num_stat_features=num_stat_features,
    )

    model_full.load_state_dict(ckpt["model_state_dict"])
    model_full.to(device)
    model_full.eval()

    train_results_full = evaluate_model(model_full, train_loader, device)
    val_results_full = evaluate_model(model_full, val_loader, device)

    print_results(train_results_full, "FULL MODEL TRAIN METRICS")
    print_results(val_results_full, "FULL MODEL VAL METRICS")


def train_combine():
    CFG = {
        'SEED': 42,
        'MAX_LEN': 37,
        'EMBEDDING_DIM': 128,
        'LSTM_UNITS': 256,
        'GRU_UNITS': 256,
        'CNN_FILTERS': 256,
        'CNN_KERNEL_SIZE': 3,
        'BATCH_SIZE': 128,
        'EPOCHS': 20,
        'N_ENSEMBLE': 12,
        'TARGET_COLS': ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6'],
        'MODEL_DIR': 'models/combine',
        'DATA_PATH': 'data'
    }

    os.makedirs(CFG['MODEL_DIR'], exist_ok=True)
    tc.seed_everything(CFG['SEED'])

    # ─── Data ───
    print("⏳ [TRAIN] Loading Data...")
    df_train, df_val, df_test, X_full, df_y_train, df_y_val, Y_full = tc.load_data(CFG['DATA_PATH'])

    print("⚙️ Processing Sequences...")
    X_train_seq = tc.process_sequences_val(tc.get_sequences(df_train), CFG['MAX_LEN'])
    X_val_seq   = tc.process_sequences_val(tc.get_sequences(df_val),   CFG['MAX_LEN'])
    X_test_seq  = tc.process_sequences_val(tc.get_sequences(df_test),  CFG['MAX_LEN'])
    X_full_seq  = tc.process_sequences_val(tc.get_sequences(X_full),   CFG['MAX_LEN'])

    VOCAB_SIZE = max(np.max(X_train_seq), np.max(X_val_seq), np.max(X_test_seq)) + 1
    print(f"   Vocab Size: {VOCAB_SIZE}")

    print("⚙️ Processing Manual Stats...")
    X_train_stats = tc.get_stats(df_train)
    X_val_stats   = tc.get_stats(df_val)
    X_full_stats  = tc.get_stats(X_full)

    scaler_p1 = StandardScaler()
    scaler_p1.fit(X_train_stats)
    joblib.dump(scaler_p1, f"{CFG['MODEL_DIR']}/scaler_phase1.pkl")

    X_train_stats_sc = scaler_p1.transform(X_train_stats)
    X_val_stats_sc   = scaler_p1.transform(X_val_stats)
    NUM_WIDE_FEATURES = X_train_stats.shape[1]

    print("⚙️ Encoding Labels...")
    encoders = tc.fit_encode_labels(Y_full, CFG['TARGET_COLS'])

    for col, le in encoders.items():
        joblib.dump(le, f"{CFG['MODEL_DIR']}/encoder_{col}.pkl")

    y_train_list = tc.transform_labels(df_y_train, CFG['TARGET_COLS'], encoders)
    y_val_list   = tc.transform_labels(df_y_val,   CFG['TARGET_COLS'], encoders)
    y_full_encoded_list = tc.transform_labels(Y_full, CFG['TARGET_COLS'], encoders)

    # ─── Phase 1: Validation check ───
    print("\n" + "=" * 40)
    print("🔍 PHASE 1: CHECK VALIDATION F1")
    print("=" * 40)

    check_model = tc.build_model('lstm', CFG, VOCAB_SIZE, NUM_WIDE_FEATURES, encoders, CFG['TARGET_COLS'])

    callbacks_p1 = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
    ]

    check_model.fit(
        x=[X_train_seq, X_train_stats_sc],
        y=y_train_list,
        validation_data=([X_val_seq, X_val_stats_sc], y_val_list),
        epochs=12,
        batch_size=CFG['BATCH_SIZE'],
        callbacks=callbacks_p1,
        verbose=1
    )

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

    # ─── Phase 2: Full train + ensemble ───
    print("\n" + "=" * 40)
    print("🚀 PHASE 2: FULL TRAINING & ENSEMBLE")
    print("=" * 40)

    scaler_full = StandardScaler()
    scaler_full.fit(X_full_stats)
    joblib.dump(scaler_full, f"{CFG['MODEL_DIR']}/scaler_full.pkl")
    X_full_stats_sc = scaler_full.transform(X_full_stats)

    # Step 8: compute class weights for full training
    sw_full = tc.compute_sample_weights(y_full_encoded_list)

    # Step 7: ReduceLROnPlateau callback for ensemble training
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)

    for i in range(CFG['N_ENSEMBLE']):
        current_seed = CFG['SEED'] + i
        tc.seed_everything(current_seed)
        tf.keras.backend.clear_session()

        if i < 4:
            m_type = 'lstm'
        elif i < 8:
            m_type = 'gru'
        else:
            m_type = 'cnn'

        print(f"\n🔄 Training Model {i+1}/{CFG['N_ENSEMBLE']} [{m_type.upper()}] (Seed {current_seed})...")

        model = tc.build_model(m_type, CFG, VOCAB_SIZE, NUM_WIDE_FEATURES, encoders, CFG['TARGET_COLS'])

        model.fit(
            x=[X_full_seq, X_full_stats_sc],
            y=y_full_encoded_list,
            epochs=CFG['EPOCHS'],
            batch_size=CFG['BATCH_SIZE'],
            sample_weight=sw_full,   # Step 8: class weights in Phase 2
            callbacks=[reduce_lr],   # Step 7: LR decay
            verbose=0,
            shuffle=True
        )

        model_path = f"{CFG['MODEL_DIR']}/model_{i}_{m_type}.keras"
        model.save(model_path)
        print(f"   ✅ Saved: {model_path}")

    print("\n🏁 TRAINING COMPLETE. Models saved.")


def train_lstm():
    SEED = 42
    N_ENSEMBLE = 10
    MODEL_DIR = "models/lstm"
    os.makedirs(MODEL_DIR, exist_ok=True)
    target_cols = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6']

    tl.seed_everything(SEED)
    df_train, df_val, df_test, X_full, df_y_train, df_y_val, Y_full = tl.load_data()

    encoders = {}
    for col in target_cols:
        le = LabelEncoder()
        le.fit(Y_full[col])
        encoders[col] = le
    joblib.dump(encoders, f'{MODEL_DIR}/encoders.pkl')

    scaler = StandardScaler()
    X_full_stats_raw = tl.get_stats(X_full)
    X_full_stats_scaled = scaler.fit_transform(X_full_stats_raw)
    joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')

    X_train_seq = tl.process_sequences_val(tl.get_sequences(df_train), max_len=37)
    X_val_seq   = tl.process_sequences_val(tl.get_sequences(df_val),   max_len=37)
    X_full_seq  = tl.process_sequences_val(tl.get_sequences(X_full),   max_len=37)

    y_train_list = [encoders[col].transform(df_y_train[col]) for col in target_cols]
    y_val_list   = [encoders[col].transform(df_y_val[col])   for col in target_cols]
    y_full_list  = [encoders[col].transform(Y_full[col])     for col in target_cols]

    vocab_size = np.max(X_full_seq) + 1
    n_stats = X_full_stats_scaled.shape[1]

    # ─── Phase 1: validation check ───
    print("\n🔍 PHASE 1: CHECK F1 ON VAL SET...")
    scaler_p1 = StandardScaler()
    X_train_stats_sc = scaler_p1.fit_transform(tl.get_stats(df_train))
    X_val_stats_sc   = scaler_p1.transform(tl.get_stats(df_val))

    sw_train = tl.calculate_sample_weights(y_train_list, target_cols)

    check_model = tl.build_model_hybrid(vocab_size, n_stats, target_cols, encoders)
    check_model.fit(
        x=[X_train_seq, X_train_stats_sc],
        y=y_train_list,
        validation_data=([X_val_seq, X_val_stats_sc], y_val_list),
        epochs=12,
        batch_size=128,
        sample_weight=sw_train,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
        ],
        verbose=1
    )
    tl.evaluate_f1(check_model, [X_val_seq, X_val_stats_sc], y_val_list, target_cols)

    # ─── Phase 2: full ensemble ───
    print("\n🚀 PHASE 2: FULL ENSEMBLE TRAINING...")
    sw_full = tl.calculate_sample_weights(y_full_list, target_cols)

    for i in range(N_ENSEMBLE):
        tl.seed_everything(SEED + i)
        print(f"--- Training Model {i+1}/{N_ENSEMBLE} ---")
        model = tl.build_model_hybrid(vocab_size, n_stats, target_cols, encoders)
        model.fit(
            x=[X_full_seq, X_full_stats_scaled],
            y=y_full_list,
            epochs=15,
            batch_size=128,
            sample_weight=sw_full,
            callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-5)],
            verbose=0
        )
        model.save(f'{MODEL_DIR}/lstm_model_{i}.keras')

    print("🏆 ALL MODELS SAVED!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    if args.model == "transformer":
        transformer()
    elif args.model == "combine":
        train_combine()
    elif args.model == "lstm":
        train_lstm()
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from: transformer, combine, lstm")


if __name__ == "__main__":
    main()
