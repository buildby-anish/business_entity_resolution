"""TF-IDF that is FITTED ON TRAIN ONLY and merely applied to validation / test (Golden Rules #1, #15, #17).

Why hashing? A normal TfidfVectorizer learns a vocabulary. Tokens that never occurred in train
(e.g. every French word in the test set) would silently vanish. Here:
  * tokens / char n-grams are mapped to buckets with a stateless hash  -> nothing is "learned" for the vocabulary
  * document frequencies (df) are learned from TRAIN text only
  * a bucket never seen in train gets the maximum idf (= "very rare, very informative"),
    so unseen countries / words still produce usable vectors.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class HashedTfidf:
    def __init__(self, analyzer: str, ngram_range=(1, 1), n_features: int = 2 ** 20):
        kwargs = dict(analyzer=analyzer, ngram_range=ngram_range, n_features=n_features,
                      alternate_sign=False, norm=None, lowercase=False, dtype=np.float32)
        if analyzer == "word":
            kwargs["token_pattern"] = r"(?u)\b\w+\b"
        self.hv = HashingVectorizer(**kwargs)
        self.n_features = n_features
        self.df: np.ndarray | None = None
        self.n_docs = 0

    def fit(self, texts) -> "HashedTfidf":
        X = self.hv.transform(list(texts)).tocsr()
        X.data[:] = 1.0                                        # binary presence -> document frequency
        self.df = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        self.n_docs = X.shape[0]
        return self

    def _idf(self, prune_frac: float | None = None, min_abs_df: int = 0) -> np.ndarray:
        idf = np.log((1.0 + self.n_docs) / (1.0 + self.df)) + 1.0     # unseen bucket (df=0) -> max idf
        if prune_frac is not None:
            too_common = self.df > max(prune_frac * self.n_docs, min_abs_df)
            idf = np.where(too_common, 0.0, idf)
        return idf.astype(np.float32)

    def transform(self, texts, prune_frac: float | None = None, min_abs_df: int = 0) -> sp.csr_matrix:
        assert self.df is not None, "call fit() on TRAIN text first"
        X = self.hv.transform(list(texts)).tocsr()
        X.data = (1.0 + np.log(X.data)).astype(np.float32)     # sublinear tf
        X = X @ sp.diags(self._idf(prune_frac, min_abs_df), format="csr")
        X = normalize(X, norm="l2", copy=False)
        X.eliminate_zeros()
        return X.tocsr()


class TextModel:
    """Four encoders (name/address x word/char). Fit once on TRAIN, then only transform()."""

    KEYS = ("name_word", "name_char", "addr_word", "addr_char")

    def __init__(self, cfg):
        self.cfg = cfg
        self.enc = {
            "name_word": HashedTfidf("word", (1, 1), cfg.n_hash_features),
            "name_char": HashedTfidf("char_wb", cfg.char_ngram, cfg.n_hash_features),
            "addr_word": HashedTfidf("word", (1, 1), cfg.n_hash_features),
            "addr_char": HashedTfidf("char_wb", cfg.char_ngram, cfg.n_hash_features),
        }

    def fit(self, names, addrs) -> "TextModel":
        names, addrs = list(names), list(addrs)
        for k in ("name_word", "name_char"):
            self.enc[k].fit(names)
        for k in ("addr_word", "addr_char"):
            self.enc[k].fit(addrs)
        return self

    def transform(self, key: str, texts, prune: bool = False) -> sp.csr_matrix:
        if prune:   # retrieval matrices ignore very common tokens (limited, road, ...)
            return self.enc[key].transform(texts, self.cfg.retrieval_max_df_frac, self.cfg.retrieval_min_abs_df)
        return self.enc[key].transform(texts)
