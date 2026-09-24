"""Stage 6: pairwise features for every FINAL candidate pair.

The SAME function builds features for train, validation and test (Golden Rule #17) and the column
order is fixed by FEATURES (Golden Rule #18). No ids, no labels ever enter a feature (Rule #3).

PS-suggested families: Jaccard, Levenshtein, TF-IDF cosine. Everything else is a proposed extra.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

TEXT_FEATURES = [
    "name_exact", "core_exact", "name_jaccard", "core_jaccard", "name_contain",
    "name_lev", "name_token_sort", "name_prefix", "name_len_diff", "name_tok_diff",
    "addr_exact", "addr_jaccard", "addr_lev", "addr_number_jaccard", "addr_postcode_match",
    "addr_missing_any",
]
VECTOR_FEATURES = ["name_tfidf_word", "name_tfidf_char", "addr_tfidf_word", "addr_tfidf_char"]
CONTEXT_FEATURES = ["cheap_score", "country_match", "is_s3", "rank_in_source", "gap_to_best", "n_cands"]
FEATURES = TEXT_FEATURES + VECTOR_FEATURES + CONTEXT_FEATURES     # the model's fixed input order


def _prep(df: pd.DataFrame) -> dict:
    fs = lambda col: [frozenset(s.split()) for s in df[col]]
    return {
        "name": df["name_norm"].tolist(), "core": df["name_core"].tolist(), "addr": df["addr_norm"].tolist(),
        "name_tok": fs("name_norm"), "core_tok": fs("name_core"), "addr_tok": fs("addr_norm"),
        "num_tok": fs("addr_nums"), "post_tok": fs("addr_post"), "country": df["country_norm"].tolist(),
    }


def _jac(a, b) -> float:
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)


def _pair(A: dict, i: int, B: dict, k: int) -> list[float]:
    an, bn, ac, bc = A["name"][i], B["name"][k], A["core"][i], B["core"][k]
    ct_a, ct_b = A["core_tok"][i], B["core_tok"][k]
    contain = float(bool(ct_a) and bool(ct_b) and (ct_a <= ct_b or ct_b <= ct_a))
    prefix = len(os.path.commonprefix([an, bn])) / max(len(an), len(bn), 1)
    aa, ba = A["addr"][i], B["addr"][k]
    have_addr = bool(aa) and bool(ba)
    pa, pb = A["post_tok"][i], B["post_tok"][k]
    post = float(len(pa & pb) > 0) if (pa and pb) else np.nan
    return [
        float(an == bn and an != ""), float(ac == bc and ac != ""),
        _jac(A["name_tok"][i], B["name_tok"][k]), _jac(ct_a, ct_b), contain,
        Levenshtein.normalized_similarity(an, bn),                # 1 - edit_distance / max_len
        fuzz.token_sort_ratio(an, bn) / 100.0, prefix,
        float(abs(len(an) - len(bn))), float(abs(len(A["name_tok"][i]) - len(B["name_tok"][k]))),
        float(aa == ba) if have_addr else np.nan,
        _jac(A["addr_tok"][i], B["addr_tok"][k]),
        Levenshtein.normalized_similarity(aa, ba) if have_addr else np.nan,
        _jac(A["num_tok"][i], B["num_tok"][k]), post,
        float(not have_addr),
    ]


def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """pairs: output of build_candidates. pools: {"S2": s2_df, "S3": s3_df} (preprocessed)."""
    out = pairs.reset_index(drop=True).copy()
    if out.empty:
        for c in FEATURES:
            out[c] = np.float32(0)
        return out

    A = _prep(s1)
    P = {k: _prep(v) for k, v in pools.items()}
    tf = np.full((len(out), len(TEXT_FEATURES)), np.nan, dtype=np.float32)
    country_match = np.zeros(len(out), dtype=np.float32)
    for source, idx in out.groupby("source").indices.items():
        B = P[source]
        for pos, i, k in zip(idx, out["s1_row"].values[idx], out["pool_row"].values[idx]):
            tf[pos] = _pair(A, i, B, k)
            ca, cb = A["country"][i], B["country"][k]
            country_match[pos] = float(ca == cb and ca != "")
    for j, name in enumerate(TEXT_FEATURES):
        out[name] = tf[:, j]

    out["name_tfidf_word"] = out["name_word_cos"]
    out["name_tfidf_char"] = out["name_char_cos"]
    miss = out["addr_missing_any"] == 1
    out["addr_tfidf_word"] = out["addr_word_cos"].where(~miss)
    out["addr_tfidf_char"] = out["addr_char_cos"].where(~miss)
    out["country_match"] = country_match
    out["is_s3"] = (out["source"] == "S3").astype(np.float32)
    out["rank_in_source"] = out["rank"].astype(np.float32)
    out["gap_to_best"] = out["cheap_score"] - out.groupby("s1_id")["cheap_score"].transform("max")
    out["n_cands"] = out.groupby("s1_id")["cand_id"].transform("size").astype(np.float32)
    for c in FEATURES:
        out[c] = out[c].astype(np.float32)
    return out
