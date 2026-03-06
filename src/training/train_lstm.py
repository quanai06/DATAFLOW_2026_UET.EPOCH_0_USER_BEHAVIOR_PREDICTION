import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
from sklearn.metrics import f1_score


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

def calculate_sample_weights(y_list, target_cols):
    """
    Tính sample weights cho model multi-output dựa trên nhãn đã mã hóa (integer).
    y_list: List các array nhãn đã encode [y_attr1, y_attr2, ...]
    """
    sample_weights_list = []
    for i, col in enumerate(target_cols):
        current_y_encoded = y_list[i] # Đây là mảng các số nguyên (0, 1, 2...)
        
        # Lấy danh sách các class index duy nhất có trong dữ liệu hiện tại
        classes = np.unique(current_y_encoded)
        
        # Tính trọng số dựa trên mảng số nguyên
        weights = class_weight.compute_class_weight(
            'balanced', 
            classes=classes, 
            y=current_y_encoded
        )
        
        # Tạo map: {class_index: weight} -> ví dụ: {0: 1.5, 1: 0.5}
        cw_map = {cls: w for cls, w in zip(classes, weights)}
        
        # Ánh xạ trọng số cho từng dòng dữ liệu
        sw = np.array([cw_map[label] for label in current_y_encoded])
        sample_weights_list.append(sw)
        
    return sample_weights_list

def build_model_hybrid(vocab_size, n_stats, target_cols, encoders):
    input_seq = Input(shape=(37,), name='input_ids')
    emb = layers.Embedding(vocab_size, 128, mask_zero=True)(input_seq)
    input_stats = Input(shape=(n_stats,), name='input_stats')
    dense_stats = layers.Dense(64, activation='relu')(input_stats) 

    outputs = []
    for col in target_cols:
        num_class = len(encoders[col].classes_)
        lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.3))(emb)
        merged = layers.Concatenate()([lstm_out, dense_stats])
        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(num_class, activation='softmax', name=f'out_{col}')(x)
        outputs.append(out)

    model = models.Model(inputs=[input_seq, input_stats], outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']*6)
    return model

def evaluate_f1(model, x_val, y_val_list, target_cols):
    preds = model.predict(x_val, verbose=0)
    f1_scores = []
    for i, col in enumerate(target_cols):
        p = np.argmax(preds[i], axis=1)
        t = y_val_list[i]
        score = f1_score(t, p, average='macro')
        f1_scores.append(score)
        print(f"   - {col}: {score:.4f}")
    avg_f1 = np.mean(f1_scores)
    print(f"🔥 Average F1-Macro: {avg_f1:.5f}")
    return avg_f1