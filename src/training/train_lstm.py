import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data(data_path="data"):
    df_train = pd.read_csv(f'{data_path}/layer2/X_train.csv')
    df_val   = pd.read_csv(f'{data_path}/layer2/X_val.csv')
    df_test  = pd.read_csv(f'{data_path}/layer2/X_test.csv')
    X_full   = pd.read_csv(f'{data_path}/layer2/full_with_manual_features.csv')
    df_y_train = pd.read_csv(f'{data_path}/layer1_raw/Y_train.csv')
    df_y_val   = pd.read_csv(f'{data_path}/layer1_raw/Y_val.csv')
    Y_full = pd.concat([df_y_train, df_y_val], ignore_index=True)
    return df_train, df_val, df_test, X_full, df_y_train, df_y_val, Y_full

def get_sequences(df):
    seq_cols = [c for c in df.columns if c.startswith('feature_')]
    return df[seq_cols].values

def process_sequences_val(values, max_len, padding='post', truncating='post'):
    sequences = [row[~pd.isna(row)].astype(int).tolist() for row in values]
    return pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating, value=0)

def get_stats(df):
    all_cols = df.columns
    seq_cols = [c for c in all_cols if c.startswith('feature_')]
    exclude_cols = set(seq_cols + ['id'])
    stat_cols = [c for c in all_cols if c not in exclude_cols]
    return df[stat_cols].values

def get_multi_output_class_weights(y_df, target_cols):
    cw_dict = {}
    for col in target_cols:
        classes = np.unique(y_df[col])
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_df[col])
        cw_dict[f'out_{col}'] = {i: w for i, w in enumerate(weights)}
    return cw_dict

def build_model_hybrid(vocab_size, n_stats, target_cols, encoders, embedding_dim=128, lstm_units=128):
    input_seq = Input(shape=(37,), name='input_ids')
    emb = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(input_seq)

    input_stats = Input(shape=(n_stats,), name='input_stats')
    dense_stats = layers.Dense(64, activation='relu')(input_stats) 

    outputs = []
    for col in target_cols:
        num_class = len(encoders[col].classes_)
        lstm_out = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=0.3))(emb)
        merged = layers.Concatenate()([lstm_out, dense_stats])
        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(num_class, activation='softmax', name=f'out_{col}')(x)
        outputs.append(out)

    model = models.Model(inputs=[input_seq, input_stats], outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']*6)
    return model