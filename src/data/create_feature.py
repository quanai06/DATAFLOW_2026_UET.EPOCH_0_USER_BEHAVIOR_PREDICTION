import pandas as pd
import numpy as np
from collections import Counter

# ======================================================
# 1. LOAD DATA
# ======================================================
print("⏳ Loading data...")

df_train = pd.read_csv('data/layer1_raw/X_train.csv')
df_val   = pd.read_csv('data/layer1_raw/X_val.csv')
df_test  = pd.read_csv('data/layer1_raw/X_test.csv')

df_y_train = pd.read_csv('data/layer1_raw/Y_train.csv')
df_y_val   = pd.read_csv('data/layer1_raw/Y_val.csv')

X_full = pd.concat([df_train, df_val], ignore_index=True)
Y_full = pd.concat([df_y_train, df_y_val], ignore_index=True)

# ======================================================
# 3B. CREATE + MERGE + EXPORT MANUAL FEATURES (ALL-IN-ONE)
# ======================================================

print("🔍 Finding Top 10 hubs...")

feature_cols = X_full.filter(like='feature_').columns

# Flatten toàn bộ action trong X_full
all_actions = (
    X_full[feature_cols]
    .values
    .ravel()
)

all_actions = all_actions[~pd.isna(all_actions)].astype(int)

cnt = Counter(all_actions)
TOP_10_HUBS = [k for k, v in cnt.most_common(10)]

print("Top hubs:", TOP_10_HUBS)

# ======================================================
# Function tạo manual features
# ======================================================
def create_manual_features(df, top_hubs):
    feature_df = df.filter(like='feature_')

    stats_list = []

    for row in feature_df.values:
        seq = row[~pd.isna(row)].astype(int)

        if len(seq) == 0:
            stats_list.append([0]*(5 + len(top_hubs)))
            continue

        length = len(seq)
        nunique = len(np.unique(seq))

        try:
            mode_val = stats.mode(seq, keepdims=True).mode[0]
        except:
            mode_val = 0

        first_item = seq[0]
        last_item  = seq[-1]

        seq_list = seq.tolist()
        hub_counts = [seq_list.count(hub) for hub in top_hubs]

        row_stats = [length, nunique, first_item, last_item, mode_val] + hub_counts
        stats_list.append(row_stats)

    columns = (
        ["length", "nunique", "first_item", "last_item", "mode"]
        + [f"hub_{hub}" for hub in top_hubs]
    )

    return pd.DataFrame(stats_list, columns=columns)


print("🔧 Creating manual features...")

# ======================================================
# Tạo manual features
# ======================================================
train_stats = create_manual_features(df_train, TOP_10_HUBS)
val_stats   = create_manual_features(df_val, TOP_10_HUBS)
test_stats  = create_manual_features(df_test, TOP_10_HUBS)
full_stats  = create_manual_features(X_full, TOP_10_HUBS)

# ======================================================
# Merge với dataframe gốc
# ======================================================
df_train_final = pd.concat(
    [df_train.reset_index(drop=True), train_stats.reset_index(drop=True)],
    axis=1
)

df_val_final = pd.concat(
    [df_val.reset_index(drop=True), val_stats.reset_index(drop=True)],
    axis=1
)

df_test_final = pd.concat(
    [df_test.reset_index(drop=True), test_stats.reset_index(drop=True)],
    axis=1
)

X_full_final = pd.concat(
    [X_full.reset_index(drop=True), full_stats.reset_index(drop=True)],
    axis=1
)

print("Done.")

# ======================================================
# Export CSV
# ======================================================
df_train_final.to_csv("data/layer2/X_train.csv", index=False)
df_val_final.to_csv("data/layer2/X_val.csv", index=False)
df_test_final.to_csv("data/layer2/X_test.csv", index=False)
X_full_final.to_csv("data/layer2/full_with_manual_features.csv", index=False)

print("CSV files exported successfully!")
