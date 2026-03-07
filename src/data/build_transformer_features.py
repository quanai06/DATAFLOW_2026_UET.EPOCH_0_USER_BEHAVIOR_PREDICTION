import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/layer2"
SAVE_PATH = "data/layer3_features/transformer"

MAX_LEN = 37
PAD = 0


def row_to_seq(row, feat_cols):
    """Extract only feature_* columns to avoid polluting sequence with stat values."""
    seq = row[feat_cols].dropna().values.astype(int)
    return seq


def pad_sequence(seq, action_remapper=None):
    if action_remapper is not None:
        seq = np.array([action_remapper.get(int(v), 0) for v in seq], dtype=np.int64)

    padded = np.full(MAX_LEN, PAD, dtype=np.int64)
    mask = np.ones(MAX_LEN, dtype=bool)  # True = padding position

    length = min(len(seq), MAX_LEN)
    padded[:length] = seq[:length]
    mask[:length] = False  # False = real token

    return padded, mask


def encode_multilabel(y_df):
    y = y_df.iloc[:, 1:].values
    encoders = []
    y_encoded_cols = []

    for i in range(y.shape[1]):
        vals = np.unique(y[:, i])
        mapping = {v: idx for idx, v in enumerate(vals)}
        encoders.append(mapping)
        col = np.vectorize(mapping.get)(y[:, i])
        y_encoded_cols.append(col)

    y_encoded = np.stack(y_encoded_cols, axis=1).astype(np.int64)
    return y_encoded, encoders


def apply_encoders(y_df, encoders):
    y = y_df.iloc[:, 1:].values
    y_encoded_cols = []

    for i in range(y.shape[1]):
        mapping = encoders[i]
        col = np.vectorize(mapping.get)(y[:, i])
        y_encoded_cols.append(col)

    y_encoded = np.stack(y_encoded_cols, axis=1).astype(np.int64)
    return y_encoded


def get_stat_cols(df):
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    return [c for c in df.columns if c not in feat_cols and c != 'id']


def build_features(df, feat_cols, action_remapper=None):
    sequences = []
    masks = []

    for _, row in df.iterrows():
        seq = row_to_seq(row, feat_cols)
        pad, mask = pad_sequence(seq, action_remapper)
        sequences.append(pad)
        masks.append(mask)

    return np.array(sequences), np.array(masks)


def build_stats(df):
    """Extract and return scaled manual stat features."""
    stat_cols = get_stat_cols(df)
    return df[stat_cols].fillna(0).values.astype(np.float32), stat_cols


def main():

    os.makedirs(SAVE_PATH, exist_ok=True)

    # =====================
    # Load data
    # =====================
    X_train = pd.read_csv(f"{RAW_PATH}/X_train.csv")
    X_val = pd.read_csv(f"{RAW_PATH}/X_val.csv")
    X_test = pd.read_csv(f"{RAW_PATH}/X_test.csv")

    y_train = pd.read_csv("data/layer1_raw/Y_train.csv")
    y_val = pd.read_csv("data/layer1_raw/Y_val.csv")

    feat_cols = [c for c in X_train.columns if c.startswith('feature_')]

    # =====================
    # Step 3: Build action remapper (re-index to contiguous 1..N, 0=PAD)
    # =====================
    all_actions = []
    for df in [X_train, X_val, X_test]:
        vals = df[feat_cols].values.ravel()
        vals = vals[~pd.isna(vals)].astype(int)
        all_actions.extend(vals[vals != 0].tolist())

    unique_actions = sorted(set(all_actions))
    # 0 reserved for PAD; remap 1..N
    action_remapper = {a: idx + 1 for idx, a in enumerate(unique_actions)}
    vocab_size = len(unique_actions) + 1  # +1 for PAD token

    print(f"Unique actions: {len(unique_actions)}, vocab_size: {vocab_size}")

    with open(f"{SAVE_PATH}/action_remapper.pkl", "wb") as f:
        pickle.dump({"remapper": action_remapper, "vocab_size": vocab_size}, f)

    # =====================
    # Build sequence features
    # =====================
    train_seq, train_mask = build_features(X_train, feat_cols, action_remapper)
    val_seq, val_mask = build_features(X_val, feat_cols, action_remapper)
    test_seq, test_mask = build_features(X_test, feat_cols, action_remapper)

    # =====================
    # Step 5: Build & scale stat features for transformer
    # =====================
    train_stats_raw, stat_cols = build_stats(X_train)
    val_stats_raw, _ = build_stats(X_val)
    test_stats_raw, _ = build_stats(X_test)

    scaler = StandardScaler()
    train_stats = scaler.fit_transform(train_stats_raw).astype(np.float32)
    val_stats = scaler.transform(val_stats_raw).astype(np.float32)
    test_stats = scaler.transform(test_stats_raw).astype(np.float32)

    with open(f"{SAVE_PATH}/stat_scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "stat_cols": stat_cols}, f)

    # =====================
    # Save features
    # =====================
    np.save(f"{SAVE_PATH}/X_train_seq.npy", train_seq)
    np.save(f"{SAVE_PATH}/X_train_mask.npy", train_mask)
    np.save(f"{SAVE_PATH}/X_train_stats.npy", train_stats)

    np.save(f"{SAVE_PATH}/X_val_seq.npy", val_seq)
    np.save(f"{SAVE_PATH}/X_val_mask.npy", val_mask)
    np.save(f"{SAVE_PATH}/X_val_stats.npy", val_stats)

    np.save(f"{SAVE_PATH}/X_test_seq.npy", test_seq)
    np.save(f"{SAVE_PATH}/X_test_mask.npy", test_mask)
    np.save(f"{SAVE_PATH}/X_test_stats.npy", test_stats)

    # =====================
    # Encode labels
    # =====================
    y_train_arr, encoders = encode_multilabel(y_train)
    y_val_arr = apply_encoders(y_val, encoders)

    np.save(f"{SAVE_PATH}/y_train.npy", y_train_arr)
    np.save(f"{SAVE_PATH}/y_val.npy", y_val_arr)

    with open(f"{SAVE_PATH}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print(f"✅ Transformer features saved. vocab_size={vocab_size}, stat_features={train_stats.shape[1]}")


if __name__ == "__main__":
    main()
