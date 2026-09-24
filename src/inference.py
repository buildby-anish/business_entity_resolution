"""Stage 11: probabilities -> final matches (with the validated thresholds)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_matches(pairs: pd.DataFrame, probs: np.ndarray, s1_ids, t_low: float, t_high: float) -> dict:
    """Every S1 in s1_ids gets an entry (empty list = singleton)."""
    if pairs.empty:
        return {s: [] for s in s1_ids}
    df = pd.DataFrame({"s1_id": pairs["s1_id"].values, "cand_id": pairs["cand_id"].values, "p": probs})
    best = df.groupby("s1_id")["p"].transform("max")
    kept = df[(df["p"] >= t_low) & (best >= t_high)]
    grouped = kept.groupby("s1_id")["cand_id"].apply(lambda x: sorted(set(x)))
    return {s: grouped.get(s, []) for s in s1_ids}
