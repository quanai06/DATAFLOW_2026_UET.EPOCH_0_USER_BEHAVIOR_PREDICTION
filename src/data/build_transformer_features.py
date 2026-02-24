import os
import numpy as np
import pandas as pd
import pickle

RAW_PATH = "data/layer1_raw"
SAVE_PATH = "data/layer3_features/transformer"

MAX_LEN = 37
PAD = 0


def row_to_seq(row):
    seq = row[1:].dropna().values.astype(int)
    return seq


def pad_sequence(seq):

    padded = np.full(MAX_LEN, PAD, dtype=np.int64)
    mask = np.ones(MAX_LEN, dtype=bool)  # True = padding

    length = min(len(seq), MAX_LEN)

    padded[:length] = seq[:length]
    mask[:length] = False   # False = real token

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


def build_features(df):

    sequences = []
    masks = []

    for _, row in df.iterrows():

        seq = row_to_seq(row)
        pad, mask = pad_sequence(seq)

        sequences.append(pad)
        masks.append(mask)

    return np.array(sequences), np.array(masks)


def main():

    os.makedirs(SAVE_PATH, exist_ok=True)

    # =====================
    # Load raw data
    # =====================
    X_train = pd.read_csv(f"{RAW_PATH}/X_train.csv")
    X_val = pd.read_csv(f"{RAW_PATH}/X_val.csv")
    X_test = pd.read_csv(f"{RAW_PATH}/X_test.csv")

    y_train = pd.read_csv(f"{RAW_PATH}/Y_train.csv")
    y_val = pd.read_csv(f"{RAW_PATH}/Y_val.csv")
    test_seq, test_mask = build_features(X_test)

    # =====================
    # Build features
    # =====================
    train_seq, train_mask = build_features(X_train)
    val_seq, val_mask = build_features(X_val)

    # =====================
    # Save features
    # =====================
    np.save(f"{SAVE_PATH}/X_train_seq.npy", train_seq)
    np.save(f"{SAVE_PATH}/X_train_mask.npy", train_mask)

    np.save(f"{SAVE_PATH}/X_val_seq.npy", val_seq)
    np.save(f"{SAVE_PATH}/X_val_mask.npy", val_mask)

    np.save(f"{SAVE_PATH}/X_test_seq.npy", test_seq)     
    np.save(f"{SAVE_PATH}/X_test_mask.npy", test_mask)

    y_train_arr, encoders = encode_multilabel(y_train)
    y_val_arr = apply_encoders(y_val, encoders)

    np.save(f"{SAVE_PATH}/y_train.npy", y_train_arr)
    np.save(f"{SAVE_PATH}/y_val.npy", y_val_arr)

    with open(f"{SAVE_PATH}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print("✅ Transformer features saved.")


if __name__ == "__main__":
    main()