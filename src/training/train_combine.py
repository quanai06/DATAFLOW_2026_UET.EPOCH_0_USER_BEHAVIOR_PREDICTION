import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ======================================================
# 1. SETUP & UTILS
# ======================================================
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass
    print(f"🔒 Global Seed Locked: {seed}")

def load_data(data_path="data"):
    print(f"⏳ Loading data from {data_path}...")
    # Load Processed Features (Layer 2)
    df_train = pd.read_csv(f'{data_path}/layer2/X_train.csv')
    df_val   = pd.read_csv(f'{data_path}/layer2/X_val.csv')
    df_test  = pd.read_csv(f'{data_path}/layer2/X_test.csv')
    X_full   = pd.read_csv(f'{data_path}/layer2/full_with_manual_features.csv')

    # Load Labels (Layer 1 - Raw)
    df_y_train = pd.read_csv(f'{data_path}/layer1_raw/Y_train.csv')
    df_y_val   = pd.read_csv(f'{data_path}/layer1_raw/Y_val.csv')
    
    # Gộp Y để encoder học hết các class
    Y_full = pd.concat([df_y_train, df_y_val], ignore_index=True)
    
    return df_train, df_val, df_test, X_full, df_y_train, df_y_val, Y_full

# ======================================================
# 2. DATA PROCESSING
# ======================================================
def get_sequences(df):
    """Lấy các cột sequence (feature_...)"""
    seq_cols = [c for c in df.columns if c.startswith('feature_')]
    return df[seq_cols].values

def process_sequences_val(values, max_len, padding='post', truncating='post'):
    """Padding sequences"""
    sequences = [row[~pd.isna(row)].astype(int).tolist() for row in values]
    return pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating, value=0)

def get_stats(df):
    """Lấy các cột manual features (trừ id và sequence)"""
    all_cols = df.columns
    seq_cols = [c for c in all_cols if c.startswith('feature_')]
    exclude_cols = set(seq_cols + ['id'])
    stat_cols = [c for c in all_cols if c not in exclude_cols]
    
    stats_data = df[stat_cols].values
    if stats_data.shape[1] == 0:
        raise ValueError("❌ Error: No statistic features found! Check feature engineering step.")
    return stats_data

# ======================================================
# 3. ENCODING
# ======================================================
def fit_encode_labels(y_full_df, target_cols):
    """Fit LabelEncoder cho từng cột target"""
    encoders = {}
    for col in target_cols:
        le = LabelEncoder()
        le.fit(y_full_df[col])
        encoders[col] = le
    return encoders

def transform_labels(df, target_cols, encoders):
    """Chuyển đổi label sang số"""
    y_list = []
    for col in target_cols:
        y_list.append(encoders[col].transform(df[col]))
    return y_list

# ======================================================
# 4. MODEL BUILDING
# ======================================================
def build_model(model_type, config, vocab_size, num_wide_features, encoders, target_cols):
    """
    Hàm dựng model tổng quát (LSTM/GRU/CNN)
    config: Dictionary chứa tham số (DIM, UNITS...)
    """
    input_seq = Input(shape=(config['MAX_LEN'],), name='input_ids')
    emb = layers.Embedding(vocab_size, config['EMBEDDING_DIM'], mask_zero=(model_type != 'cnn'))(input_seq)
    
    input_stats = Input(shape=(num_wide_features,), name='input_stats')
    dense_stats = layers.Dense(64, activation='relu')(input_stats)

    outputs = []
    for col in target_cols:
        num_class = len(encoders[col].classes_)
        
        if model_type == 'lstm':
            x = layers.Bidirectional(layers.LSTM(config['LSTM_UNITS'], dropout=0.3))(emb)
        elif model_type == 'gru':
            x = layers.Bidirectional(layers.GRU(config['GRU_UNITS'], dropout=0.3))(emb)
        else: # CNN
            x = layers.Conv1D(config['CNN_FILTERS'], config['CNN_KERNEL_SIZE'], activation='relu', padding='same')(emb)
            x = layers.Conv1D(config['CNN_FILTERS'], 5, activation='relu', padding='same')(x)
            x = layers.GlobalMaxPooling1D()(x)
            x = layers.Dropout(0.3)(x)

        merged = layers.Concatenate()([x, dense_stats])
        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(num_class, activation='softmax', name=f'out_{col}')(x)
        outputs.append(out)

    model = models.Model(inputs=[input_seq, input_stats], outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']*len(target_cols))
    return model