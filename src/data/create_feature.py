import pandas as pd
import numpy as np
from collections import Counter
import math

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
# 2. TOP-10 HUBS
# ======================================================
print("🔍 Finding Top 10 hubs...")

feature_cols = X_full.filter(like='feature_').columns

all_actions = X_full[feature_cols].values.ravel()
all_actions = all_actions[~pd.isna(all_actions)].astype(int)
all_actions = all_actions[all_actions != 0]

cnt = Counter(all_actions)
TOP_10_HUBS = [k for k, v in cnt.most_common(10)]
print("Top hubs:", TOP_10_HUBS)

# Rare actions: appear in fewer than 0.1% of sequences
total_seqs = len(X_full)
rare_threshold = max(2, int(total_seqs * 0.001))
rare_actions = set(a for a, c in cnt.items() if c < rare_threshold)
print(f"Rare actions (count < {rare_threshold}): {len(rare_actions)}")


# ======================================================
# 3. FEATURE ENGINEERING
# ======================================================
def _entropy(seq):
    if len(seq) == 0:
        return 0.0
    n = len(seq)
    counts = Counter(seq)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p + 1e-12)
    return float(ent)


def _rb3_count(seq):
    """A → B → A patterns"""
    n = len(seq)
    if n < 3:
        return 0
    return sum(1 for i in range(n - 2) if seq[i] == seq[i + 2] and seq[i] != seq[i + 1])


def _rb4_count(seq):
    """A → B → C → A patterns"""
    n = len(seq)
    if n < 4:
        return 0
    return sum(1 for i in range(n - 3) if seq[i] == seq[i + 3] and seq[i] != seq[i + 1])


def create_manual_features(df, top_hubs, rare_actions_set):
    feature_df = df.filter(like='feature_')
    stats_list = []

    for row in feature_df.values:
        seq = row[~pd.isna(row)].astype(int)
        seq = seq[seq != 0].tolist()

        if len(seq) == 0:
            stats_list.append([0] * (17 + len(top_hubs)))
            continue

        length = len(seq)
        nunique = len(set(seq))
        first_item = seq[0]
        last_item = seq[-1]

        cnt_seq = Counter(seq)
        mode_val = cnt_seq.most_common(1)[0][0]
        action_dominance = cnt_seq.most_common(1)[0][1] / length

        entropy = _entropy(seq)

        # Transition features
        n_transitions = sum(1 for i in range(length - 1) if seq[i] != seq[i + 1])
        transition_ratio = n_transitions / max(length - 1, 1)

        # Rollback features
        rb_3_steps = _rb3_count(seq)
        rb_4_steps = _rb4_count(seq)
        first_action_rb = 1 if (length >= 3 and seq[0] == seq[2]) else 0

        # Rare action features
        n_rare = sum(1 for a in seq if a in rare_actions_set)
        rare_ratio = n_rare / length

        # Hub counts
        hub_counts = [seq.count(hub) for hub in top_hubs]

        row_stats = [
            length, nunique, first_item, last_item, mode_val,
            action_dominance, entropy, n_transitions, transition_ratio,
            rb_3_steps, rb_4_steps, first_action_rb,
            n_rare, rare_ratio,
            nunique / length,  # diversity ratio
            last_item == first_item,  # loop flag
            length / 37.0,  # normalized length
        ] + hub_counts

        stats_list.append(row_stats)

    columns = [
        "length", "nunique", "first_item", "last_item", "mode",
        "action_dominance", "entropy", "n_transitions", "transition_ratio",
        "rb_3_steps", "rb_4_steps", "first_action_rb",
        "n_rare_actions", "rare_action_ratio",
        "diversity_ratio", "loop_flag", "norm_length",
    ] + [f"hub_{hub}" for hub in top_hubs]

    return pd.DataFrame(stats_list, columns=columns)


print("🔧 Creating manual features...")

train_stats = create_manual_features(df_train, TOP_10_HUBS, rare_actions)
val_stats   = create_manual_features(df_val,   TOP_10_HUBS, rare_actions)
test_stats  = create_manual_features(df_test,  TOP_10_HUBS, rare_actions)
full_stats  = create_manual_features(X_full,   TOP_10_HUBS, rare_actions)

# ======================================================
# 4. MERGE AND EXPORT
# ======================================================
df_train_final = pd.concat([df_train.reset_index(drop=True), train_stats.reset_index(drop=True)], axis=1)
df_val_final   = pd.concat([df_val.reset_index(drop=True),   val_stats.reset_index(drop=True)],   axis=1)
df_test_final  = pd.concat([df_test.reset_index(drop=True),  test_stats.reset_index(drop=True)],  axis=1)
X_full_final   = pd.concat([X_full.reset_index(drop=True),   full_stats.reset_index(drop=True)],  axis=1)

df_train_final.to_csv("data/layer2/X_train.csv", index=False)
df_val_final.to_csv("data/layer2/X_val.csv",     index=False)
df_test_final.to_csv("data/layer2/X_test.csv",   index=False)
X_full_final.to_csv("data/layer2/full_with_manual_features.csv", index=False)

print(f"✅ CSV files exported. Features per row: {len(train_stats.columns)}")
print("Feature columns:", train_stats.columns.tolist())
