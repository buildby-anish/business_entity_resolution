"""Stage 4: blocking = cheap retrieval of plausible S2/S3 partners for each S1 entity.

Three recall-oriented channels are UNIONED (a true pair only needs to be found by one):
  name_word  rare name tokens          (common tokens are ignored -> bounded work)
  addr_word  rare address tokens       (postcodes / localities are rare tokens)
  name_char  character 3-4-grams       (typos, transliteration)
Then a cheap score ranks the union and the best k_final per S1 per source survive:
that survivor list is the FINAL candidate set the model scores.

Everything here is data-independent except the fitted TextModel (train-only statistics).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .text_model import TextModel

RETRIEVAL_KEYS = ("name_word", "addr_word", "name_char")


class TextIndex:
    """TF-IDF matrices for one (S1 records, pool) pair, built with a *fitted* TextModel."""

    def __init__(self, tm: TextModel, s1: pd.DataFrame, pool: pd.DataFrame, cfg):
        self.cfg = cfg
        col = {"name": "name_norm", "addr": "addr_norm"}
        self.feat, self.ret = {}, {}
        for key in TextModel.KEYS:
            field = key.split("_")[0]
            self.feat[key] = (tm.transform(key, s1[col[field]]), tm.transform(key, pool[col[field]]))
        for key in RETRIEVAL_KEYS:
            field = key.split("_")[0]
            self.ret[key] = (tm.transform(key, s1[col[field]], prune=True),
                             tm.transform(key, pool[col[field]], prune=True))

    def pair_cos(self, key: str, q: np.ndarray, y: np.ndarray) -> np.ndarray:
        A, B = self.feat[key]
        out = np.empty(len(q), dtype=np.float32)
        step = self.cfg.pair_chunk
        for s in range(0, len(q), step):
            sl = slice(s, s + step)
            out[sl] = np.asarray(A[q[sl]].multiply(B[y[sl]]).sum(axis=1)).ravel()
        return out


def topk_pairs(Xq, Y, k: int, chunk_rows: int):
    """For every row of Xq return its top-k rows of Y by dot product -> (q_idx, y_idx) arrays."""
    YT = Y.T.tocsr()
    qs, ys = [], []
    for start in range(0, Xq.shape[0], chunk_rows):
        S = (Xq[start:start + chunk_rows] @ YT).tocsr()
        indptr, indices, data = S.indptr, S.indices, S.data
        for i in range(S.shape[0]):
            a, b = indptr[i], indptr[i + 1]
            if a == b:
                continue
            idx, sc = indices[a:b], data[a:b]
            if b - a > k:
                idx = idx[np.argpartition(-sc, k - 1)[:k]]
            qs.append(np.full(len(idx), start + i, dtype=np.int64))
            ys.append(idx.astype(np.int64))
    if not qs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(qs), np.concatenate(ys)


def generate_candidates(tm: TextModel, s1: pd.DataFrame, pool: pd.DataFrame, cfg):
    """Return (pairs_df, TextIndex). pairs_df columns: s1_row, pool_row, s1_id, cand_id, *_cos, cheap_score, rank."""
    idx = TextIndex(tm, s1, pool, cfg)
    s1_c, pool_c = s1["country_norm"].to_numpy(dtype=object), pool["country_norm"].to_numpy(dtype=object)
    if cfg.require_same_country:
        groups = [(np.where(s1_c == c)[0], np.where(pool_c == c)[0]) for c in np.unique(s1_c)]
    else:
        groups = [(np.arange(len(s1)), np.arange(len(pool)))]

    ks = {"name_word": cfg.k_name_word, "addr_word": cfg.k_addr_word, "name_char": cfg.k_name_char}
    qs, ys = [], []
    for s1_rows, pool_rows in groups:
        if len(s1_rows) == 0 or len(pool_rows) == 0:
            continue
        for key in RETRIEVAL_KEYS:
            Xa, Yb = idx.ret[key]
            q, y = topk_pairs(Xa[s1_rows], Yb[pool_rows], ks[key], cfg.chunk_rows)
            qs.append(s1_rows[q])
            ys.append(pool_rows[y])

    cols = ["s1_row", "pool_row", "s1_id", "cand_id", "name_word_cos", "name_char_cos",
            "addr_word_cos", "addr_char_cos", "cheap_score", "rank"]
    if not qs or sum(len(a) for a in qs) == 0:
        return pd.DataFrame(columns=cols), idx

    key = np.unique(np.concatenate(qs) * len(pool) + np.concatenate(ys))     # union + dedupe
    q, y = key // len(pool), key % len(pool)
    pairs = pd.DataFrame({"s1_row": q, "pool_row": y})
    for k in TextModel.KEYS:
        pairs[f"{k}_cos"] = idx.pair_cos(k, q, y)
    pairs["cheap_score"] = (cfg.w_name * pairs["name_char_cos"] + (1 - cfg.w_name) * pairs["addr_char_cos"])
    pairs = pairs.sort_values(["s1_row", "cheap_score"], ascending=[True, False], kind="mergesort")
    pairs["rank"] = pairs.groupby("s1_row").cumcount() + 1
    pairs = pairs[pairs["rank"] <= cfg.k_final].copy()                       # FINAL candidates
    pairs["s1_id"] = s1["entity_id"].to_numpy(dtype=object)[pairs["s1_row"].to_numpy()]
    pairs["cand_id"] = pool["entity_id"].to_numpy(dtype=object)[pairs["pool_row"].to_numpy()]
    return pairs[cols].reset_index(drop=True), idx
