from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

import numpy as np
import pandas as pd


def _get_sequence_columns(df: pd.DataFrame) -> list[str]:
    seq_cols = [col for col in df.columns if col.startswith("feature_")]
    seq_cols.sort(key=lambda name: int(name.split("_")[1]))
    if not seq_cols:
        raise ValueError("Khong tim thay cot sequence dang feature_*.")  # noqa: RUF001
    return seq_cols


def _calculate_entropy(sequence: np.ndarray) -> float:
    if len(sequence) == 0:
        return 0.0
    counts = Counter(sequence)
    probs = np.array(list(counts.values()), dtype=float) / len(sequence)
    return float(-np.sum(probs * np.log2(probs + 1e-9)))


def _count_rollbacks(sequence: Iterable[float]) -> tuple[int, int, list[str], list[str]]:
    seq = [int(x) for x in sequence if pd.notnull(x) and int(x) != 0]
    n = len(seq)

    rollback_3 = 0
    rollback_4 = 0
    rb3_actions: set[str] = set()
    rb4_actions: set[str] = set()

    if n < 3:
        return 0, 0, [], []

    # Rollback 3 buoc: A-B-A, yeu cau B != A.
    for i in range(n - 2):
        if seq[i] == seq[i + 2] and seq[i] != seq[i + 1]:
            rollback_3 += 1
            rb3_actions.add(str(seq[i]))

    # Rollback 4 buoc: A-B-C-A, yeu cau B != A va C != A.
    for i in range(n - 3):
        if (
            seq[i] == seq[i + 3]
            and seq[i] != seq[i + 1]
            and seq[i] != seq[i + 2]
        ):
            rollback_4 += 1
            rb4_actions.add(str(seq[i]))

    return rollback_3, rollback_4, sorted(rb3_actions), sorted(rb4_actions)


def _row_to_action_sequence(row: pd.Series, seq_cols: list[str]) -> str:
    actions = row[seq_cols].dropna().astype(int).astype(str).tolist()
    return "-".join(actions)


def featuring_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seq_cols = _get_sequence_columns(df)

    df["length"] = df[seq_cols].notna().sum(axis=1)

    def row_entropy(row: pd.Series) -> float:
        seq = row[seq_cols].dropna().values.astype(int)
        return _calculate_entropy(seq)

    df["entropy"] = df.apply(row_entropy, axis=1)

    rollback_results = df[seq_cols].apply(
        lambda row: _count_rollbacks(row.values),
        axis=1,
    )
    rollback_df = pd.DataFrame(rollback_results.tolist(), index=df.index)

    df["rb_3_steps"] = rollback_df[0]
    df["rb_4_steps"] = rollback_df[1]
    df["first_action_rb"] = [
        sorted(set(a + b)) for a, b in zip(rollback_df[2], rollback_df[3])
    ]
    df["action_sequence"] = df.apply(
        lambda row: _row_to_action_sequence(row, seq_cols),
        axis=1,
    )
    return df


def generate_edge_case_report(
    df: pd.DataFrame,
    mode: Literal["or", "and"] = "or",
    min_rb3: int = 2,
    min_rb4: int = 1,
) -> pd.DataFrame:
    """
    mode='or': chon case co rollback_3 >= min_rb3 HOAC rollback_4 >= min_rb4.
    mode='and': chon case co rollback_3 >= min_rb3 VA rollback_4 >= min_rb4.
    """
    if mode not in {"or", "and"}:
        raise ValueError("mode phai la 'or' hoac 'and'.")

    cond_rb3 = df["rb_3_steps"] >= min_rb3
    cond_rb4 = df["rb_4_steps"] >= min_rb4
    mask = (cond_rb3 | cond_rb4) if mode == "or" else (cond_rb3 & cond_rb4)
    edge_cases = df[mask].copy()

    def create_fact(row: pd.Series) -> str:
        rb3 = int(row["rb_3_steps"])
        rb4 = int(row["rb_4_steps"])
        length = int(row["length"])
        entropy = round(float(row["entropy"]), 2)
        return (
            f"Phat hien {rb3} lan lap 3 buoc (A-B-A), "
            f"{rb4} lan lap 4 buoc (A-B-C-A). "
            f"Do dai chuoi: {length} thao tac. "
            f"Chi so hon loan (Entropy): {entropy}."
        )

    def create_edge_rule(row: pd.Series) -> str:
        hit_rb3 = int(row["rb_3_steps"]) >= min_rb3
        hit_rb4 = int(row["rb_4_steps"]) >= min_rb4
        if hit_rb3 and hit_rb4:
            return f"rb3>={min_rb3}+rb4>={min_rb4}"
        if hit_rb3:
            return f"rb3>={min_rb3}"
        return f"rb4>={min_rb4}"

    report_df = pd.DataFrame()
    report_df["id"] = edge_cases["id"]
    report_df["action_sequence"] = edge_cases["action_sequence"]
    report_df["first_action_rb"] = edge_cases["first_action_rb"]
    report_df["rb_3_steps"] = edge_cases["rb_3_steps"].astype(int)
    report_df["rb_4_steps"] = edge_cases["rb_4_steps"].astype(int)
    report_df["length"] = edge_cases["length"].astype(int)
    report_df["entropy"] = edge_cases["entropy"].astype(float)
    report_df["edge_rule"] = edge_cases.apply(create_edge_rule, axis=1)
    report_df["fact"] = edge_cases.apply(create_fact, axis=1)

    return report_df.reset_index(drop=True)
