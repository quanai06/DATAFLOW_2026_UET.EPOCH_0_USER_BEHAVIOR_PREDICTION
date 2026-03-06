"""Group-first SLM triage for supervisor workflow.

Core idea:
- Do NOT analyze each row independently.
- Build behavior signatures, group similar edge-cases, then label each group.
- Return compact row-level routing string for fast human review.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Optional SLM backend (lazy import). If unavailable, use deterministic fallback.
_MODEL = None
_TOKENIZER = None
_SLM_AVAILABLE = True


@dataclass
class RowFeature:
    row_id: str
    sequence_action: str
    anchor: str
    aba_count: int
    b_unique: int
    rb4: int
    length: int
    entropy: float
    nent: float
    ent_level: str
    signature: str


# Runtime context primed by pipeline once per batch.
_ROW_BY_SEQUENCE: Dict[str, RowFeature] = {}
_GROUP_INFO_BY_SIG: Dict[str, Dict[str, str]] = {}
_CACHE: Dict[Tuple[str, str], str] = {}
_ENT_THRESHOLDS: Tuple[float, float] = (0.30, 0.70)  # low, high on normalized entropy


def _ensure_model():
    global _MODEL, _TOKENIZER, _SLM_AVAILABLE
    if not _SLM_AVAILABLE:
        raise RuntimeError("SLM backend unavailable")
    if _MODEL is None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            _MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            _MODEL.eval()
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            if _TOKENIZER.pad_token_id is None:
                _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id
        except Exception:
            _SLM_AVAILABLE = False
            raise RuntimeError("Cannot load torch/transformers model")
    return _MODEL, _TOKENIZER


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    if q <= 0:
        return float(arr[0])
    if q >= 1:
        return float(arr[-1])
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    w = pos - lo
    return float(arr[lo] * (1 - w) + arr[hi] * w)


def _parse_sequence(sequence_action: str) -> List[int]:
    if sequence_action is None:
        return []
    s = str(sequence_action).strip()
    if not s or s.lower() == "nan":
        return []
    seq: List[int] = []
    for tok in s.split("-"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(float(tok))
            if v != 0:
                seq.append(v)
        except Exception:
            continue
    return seq


def _entropy(seq: List[int]) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    counts = Counter(seq)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p + 1e-12)
    return float(ent)


def _normalized_entropy(entropy: float, unique_count: int) -> float:
    if unique_count <= 1:
        return 0.0
    return float(entropy / (math.log2(unique_count) + 1e-12))


def _extract_features_from_fact(fact_text: str) -> Dict[str, float]:
    if not fact_text:
        return {}
    s = str(fact_text)
    out: Dict[str, float] = {}

    m = re.search(r"Phát hiện\s+(\d+)\s+lần lặp 3 bước", s)
    if m:
        out["rb3"] = float(m.group(1))
    m = re.search(r"(\d+)\s+lần lặp 4 bước", s)
    if m:
        out["rb4"] = float(m.group(1))
    m = re.search(r"Độ dài chuỗi:\s*(\d+)", s)
    if m:
        out["length"] = float(m.group(1))
    m = re.search(r"Entropy\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        out["entropy"] = float(m.group(1))
    return out


def _rb4_count(seq: List[int]) -> int:
    n = len(seq)
    if n < 4:
        return 0
    rb4 = 0
    for i in range(n - 3):
        if seq[i] == seq[i + 3] and seq[i] != seq[i + 1]:
            rb4 += 1
    return rb4


def _anchor(seq: List[int]) -> str:
    if not seq:
        return "-"
    counts = Counter(seq)
    best_count = max(counts.values())
    candidates = {v for v, c in counts.items() if c == best_count}
    # tie-break by first appearance for stability in workflow
    for v in seq:
        if v in candidates:
            return str(v)
    return str(seq[0])


def _anchor_aba_stats(seq: List[int], anchor: str) -> Tuple[int, int]:
    if anchor == "-":
        return 0, 0
    a = int(anchor)
    n = len(seq)
    aba = 0
    mids = set()
    for i in range(n - 2):
        if seq[i] == a and seq[i + 2] == a and seq[i + 1] != a:
            aba += 1
            mids.add(seq[i + 1])
    return aba, len(mids)


def _ent_level(nent: float, low_thr: float, high_thr: float) -> str:
    if nent >= high_thr + 1e-6:
        return "HIGH"
    if nent <= low_thr - 1e-6:
        return "LOW"
    return "MID"


def _len_bucket(length: int) -> str:
    if length >= 21:
        return "LONG"
    if length >= 12:
        return "MID"
    return "SHORT"


def _build_signature(
    anchor: str,
    aba_count: int,
    b_unique: int,
    rb4: int,
    length: int,
    ent_level: str,
) -> str:
    return (
        f"A{anchor}|ABA{aba_count}|BU{b_unique}|RB4{1 if rb4 > 0 else 0}|"
        f"L{_len_bucket(length)}|E{ent_level}"
    )


def _rule_tag_priority(row: RowFeature) -> Tuple[str, str]:
    # 5 tags requested: ANCHOR_LOOP, RB4, LONG, HIGH_VAR, MIXED
    if row.rb4 > 0:
        tag = "RB4"
        p = "HIGH" if row.ent_level == "HIGH" or row.rb4 >= 2 else "MED"
        return tag, p

    if row.length >= 21 and row.aba_count == 0:
        return "LONG", "MED"

    if row.aba_count >= 2 and row.b_unique <= 3:
        # strong anchor looping, usually repetitive not immediately critical
        p = "MED" if row.aba_count >= 3 else "LOW"
        return "ANCHOR_LOOP", p

    if row.ent_level == "HIGH":
        return "HIGH_VAR", "MED"

    return "MIXED", "LOW"


def _slm_group_label(signature: str, sample_row: RowFeature) -> Tuple[str, str]:
    """
    Optional SLM group labeling. If unavailable, fallback to deterministic rules.
    Output constrained to requested set.
    """
    fallback_tag, fallback_p = _rule_tag_priority(sample_row)
    try:
        model, tokenizer = _ensure_model()
        prompt = (
            "You classify behavior groups.\n"
            "Allowed TAG: ANCHOR_LOOP, RB4, LONG, HIGH_VAR, MIXED.\n"
            "Allowed P: HIGH, MED, LOW.\n"
            "Return exactly: TAG=<TAG>|P=<P>\n"
            f"signature={signature}\n"
            f"anchor={sample_row.anchor}, aba={sample_row.aba_count}, bunique={sample_row.b_unique}, "
            f"rb4={sample_row.rb4}, len={sample_row.length}, ent_level={sample_row.ent_level}."
        )
        messages = [
            {"role": "system", "content": "Respond in one line with strict format."},
            {"role": "user", "content": prompt},
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = generated[:, inputs["input_ids"].shape[1]:]
        out = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

        m = re.search(r"TAG\s*=\s*([A-Z_]+)\s*\|\s*P\s*=\s*([A-Z]+)", out)
        if not m:
            return fallback_tag, fallback_p
        tag = m.group(1)
        p = m.group(2)
        if tag not in {"ANCHOR_LOOP", "RB4", "LONG", "HIGH_VAR", "MIXED"}:
            tag = fallback_tag
        if p not in {"HIGH", "MED", "LOW"}:
            p = fallback_p
        return tag, p
    except Exception:
        return fallback_tag, fallback_p


def _make_row_feature(row_id: str, sequence_action: str, fact_text: Optional[str]) -> RowFeature:
    seq = _parse_sequence(sequence_action)

    if seq:
        length = len(seq)
        entropy = _entropy(seq)
        unique = len(set(seq))
        rb4 = _rb4_count(seq)
        anchor = _anchor(seq)
        aba_count, b_unique = _anchor_aba_stats(seq, anchor)
    else:
        fx = _extract_features_from_fact(fact_text or "")
        length = int(fx.get("length", 0))
        entropy = float(fx.get("entropy", 0.0))
        rb4 = int(fx.get("rb4", 0))
        # no raw sequence -> weak fallback
        anchor = "-"
        aba_count = int(fx.get("rb3", 0))
        b_unique = 0
        unique = 2 if entropy > 0 else 1

    nent = _normalized_entropy(entropy, unique)
    low_thr, high_thr = _ENT_THRESHOLDS
    ent_level = _ent_level(nent, low_thr, high_thr)

    signature = _build_signature(
        anchor=anchor,
        aba_count=aba_count,
        b_unique=b_unique,
        rb4=rb4,
        length=length,
        ent_level=ent_level,
    )

    return RowFeature(
        row_id=str(row_id),
        sequence_action=str(sequence_action),
        anchor=anchor,
        aba_count=aba_count,
        b_unique=b_unique,
        rb4=rb4,
        length=length,
        entropy=entropy,
        nent=nent,
        ent_level=ent_level,
        signature=signature,
    )


def prime_group_context(report_df) -> None:
    """
    Build full batch context once:
    - dynamic entropy thresholds from current batch
    - grouping by behavior signature
    - group TAG/P labels and representative row
    """
    global _ROW_BY_SEQUENCE, _GROUP_INFO_BY_SIG, _CACHE, _ENT_THRESHOLDS

    _ROW_BY_SEQUENCE = {}
    _GROUP_INFO_BY_SIG = {}
    _CACHE = {}

    # 1) Compute dynamic entropy thresholds from full batch
    nents: List[float] = []
    temp_rows: List[Tuple[str, str, str]] = []  # (id, seq, fact)
    for _, r in report_df.iterrows():
        rid = str(r.get("id", ""))
        seq = str(r.get("action_sequence", ""))
        fact = str(r.get("fact", ""))
        temp_rows.append((rid, seq, fact))

        parsed = _parse_sequence(seq)
        if parsed:
            ent = _entropy(parsed)
            nent = _normalized_entropy(ent, len(set(parsed)))
            nents.append(nent)

    if len(nents) >= 5:
        low_thr = _quantile(nents, 0.30)
        high_thr = _quantile(nents, 0.70)
    elif nents:
        med = _quantile(nents, 0.50)
        low_thr = max(0.0, med - 0.10)
        high_thr = min(1.0, med + 0.10)
    else:
        low_thr, high_thr = 0.30, 0.70

    _ENT_THRESHOLDS = (low_thr, high_thr)

    # 2) Build row features and group map
    sig_to_rows: Dict[str, List[RowFeature]] = defaultdict(list)
    for rid, seq, fact in temp_rows:
        rf = _make_row_feature(rid, seq, fact)
        _ROW_BY_SEQUENCE[rf.sequence_action] = rf
        sig_to_rows[rf.signature].append(rf)

    # 3) Assign stable group ids by descending group size
    sorted_sigs = sorted(sig_to_rows.keys(), key=lambda s: (-len(sig_to_rows[s]), s))
    for idx, sig in enumerate(sorted_sigs, start=1):
        members = sig_to_rows[sig]
        rep = members[0]
        count = len(members)
        gid = f"G{idx:03d}"

        tag, p = _slm_group_label(sig, rep)

        _GROUP_INFO_BY_SIG[sig] = {
            "gid": gid,
            "tag": tag,
            "p": p,
            "count": str(count),
            "rep": rep.row_id,
            "anchor": rep.anchor,
            "aba": str(rep.aba_count),
            "buniq": str(rep.b_unique),
        }


def _fallback_line(sequence_action: str, fact_text: str) -> str:
    rf = _make_row_feature("NA", sequence_action, fact_text)
    tag, p = _rule_tag_priority(rf)
    return (
        f"G=G000|TAG={tag}|P={p}|COUNT=1|REP=NA|"
        f"A={rf.anchor}|ABA={rf.aba_count}|Buniq={rf.b_unique}"
    )


def get_slm_analysis(fact_text, sequence_action):
    """
    Return required compact routing format:
    G=<id>|TAG=<...>|P=<...>|COUNT=<...>|REP=<...>|A=<...>|ABA=<...>|Buniq=<...>
    """
    key = (str(fact_text), str(sequence_action))
    if key in _CACHE:
        return _CACHE[key]

    seq_key = str(sequence_action)
    rf = _ROW_BY_SEQUENCE.get(seq_key)
    if rf is None:
        out = _fallback_line(seq_key, str(fact_text))
        _CACHE[key] = out
        return out

    info = _GROUP_INFO_BY_SIG.get(rf.signature)
    if info is None:
        out = _fallback_line(seq_key, str(fact_text))
        _CACHE[key] = out
        return out

    out = (
        f"G={info['gid']}|TAG={info['tag']}|P={info['p']}|COUNT={info['count']}|REP={info['rep']}|"
        f"A={info['anchor']}|ABA={info['aba']}|Buniq={info['buniq']}"
    )
    _CACHE[key] = out
    return out
