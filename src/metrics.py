"""Local scorer. Implements the PS metric exactly: F0.5 per S1 entity, macro-averaged over ALL S1 (singletons included).

  truth empty & prediction empty      -> 1.0     (PS)
  truth empty & any prediction        -> 0.0     (PS)
  truth non-empty & empty prediction  -> 0.0     (recall = 0; PS does not spell this case out)
  otherwise                           -> 1.25*P*R / (0.25*P + R)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def f05_arrays(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    tp, n_pred, n_true = (np.asarray(a, dtype=np.float64) for a in (tp, n_pred, n_true))
    prec = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    rec = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    denom = 0.25 * prec + rec
    f = np.divide(1.25 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    f = np.where((n_true == 0) & (n_pred > 0), 0.0, f)
    return f


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    """pred/truth: {s1_id -> iterable of ids}. Every S1 in s1_ids counts."""
    tp, npred, ntrue = [], [], []
    for s in s1_ids:
        p, t = set(pred.get(s, ())), set(truth.get(s, ()))
        tp.append(len(p & t)); npred.append(len(p)); ntrue.append(len(t))
    return float(f05_arrays(np.array(tp), np.array(npred), np.array(ntrue)).mean())


def blocking_metrics(pairs: pd.DataFrame, truth: dict, s1_ids, pool_sizes: dict) -> dict:
    """pairs must carry a 'label' column (see dataset_builder.label_pairs)."""
    n_true = sum(len(truth[s]) for s in s1_ids)
    per_s1 = pairs.groupby("s1_id").size().reindex(list(s1_ids), fill_value=0)
    all_pairs = len(s1_ids) * sum(pool_sizes.values())
    return {
        "n_s1": len(s1_ids), "n_true_pairs": int(n_true), "n_candidates": int(len(pairs)),
        "blocking_recall": float(pairs["label"].sum() / n_true) if n_true else float("nan"),
        "reduction_ratio": float(1 - len(pairs) / all_pairs) if all_pairs else float("nan"),
        "avg_candidates_per_s1": float(per_s1.mean()), "max_candidates_per_s1": int(per_s1.max()),
        "s1_with_no_candidates": int((per_s1 == 0).sum()),
    }


def sweep_thresholds(pairs: pd.DataFrame, probs: np.ndarray, truth: dict, s1_ids, grid) -> pd.DataFrame:
    """Two-threshold rule: an S1 may have matches only if its best candidate >= t_high; then every
    candidate >= t_low is kept (t_low <= t_high). t_low == t_high is the ordinary single threshold.
    True matches that blocking missed still count as false negatives (n_true comes from the truth)."""
    s1_ids = list(s1_ids)
    index = {s: i for i, s in enumerate(s1_ids)}
    n = len(s1_ids)
    n_true = np.array([len(truth[s]) for s in s1_ids])
    si = pairs["s1_id"].map(index).values
    label = pairs["label"].values.astype(bool)
    maxp = np.full(n, -1.0)
    np.maximum.at(maxp, si, probs)
    rows = []
    for t_low in grid:
        for t_high in grid:
            if t_high < t_low:
                continue
            keep = (probs >= t_low) & (maxp[si] >= t_high)
            tp = np.bincount(si[keep & label], minlength=n)
            fp = np.bincount(si[keep & ~label], minlength=n)
            f = f05_arrays(tp, tp + fp, n_true)
            rows.append((t_low, t_high, f.mean(), f[n_true == 0].mean() if (n_true == 0).any() else np.nan,
                         f[n_true > 0].mean() if (n_true > 0).any() else np.nan))
    return pd.DataFrame(rows, columns=["t_low", "t_high", "macro_f05", "f05_singletons", "f05_matched"])
