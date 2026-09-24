"""Stage 7: turn ground truth into pairwise training rows (label 1 = in ground truth, 0 = not)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_pairs(pairs: pd.DataFrame, truth: dict[str, set[str]]) -> pd.DataFrame:
    positives = {(s, c) for s, recs in truth.items() for c in recs}
    out = pairs.copy()
    out["label"] = np.fromiter(((s, c) in positives for s, c in zip(out["s1_id"], out["cand_id"])),
                               dtype=np.int8, count=len(out))
    return out


def sample_training_pairs(df: pd.DataFrame, cfg, seed: int) -> pd.DataFrame:
    """TRAIN rows only. Keep every positive, the hardest negatives, and a few random negatives.
    Never applied to validation/test: thresholds must be tuned on the natural pair distribution."""
    if not cfg.neg_sampling:
        return df
    rng = np.random.default_rng(seed)
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].sort_values(["s1_id", "cheap_score"], ascending=[True, False], kind="mergesort")
    neg = neg.assign(_hard=neg.groupby("s1_id").cumcount())
    hard = neg[neg["_hard"] < cfg.neg_hard_per_s1]
    rest = neg[neg["_hard"] >= cfg.neg_hard_per_s1].copy()
    rest["_r"] = rng.random(len(rest))
    rest = rest.sort_values(["s1_id", "_r"], kind="mergesort")
    rest = rest[rest.groupby("s1_id").cumcount() < cfg.neg_random_per_s1]
    keep = pd.concat([pos, hard.drop(columns="_hard"), rest.drop(columns=["_hard", "_r"])])
    return keep.sample(frac=1.0, random_state=seed).reset_index(drop=True)
