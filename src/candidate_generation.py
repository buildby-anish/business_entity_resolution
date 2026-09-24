"""Stage 5: run blocking against S2 and S3 and assemble the FINAL candidate set.

The table returned here is exactly what the model scores -> it is what candidate_pairs.tsv must contain (PS).
"""
from __future__ import annotations

import pandas as pd

from .blocking import generate_candidates


def build_candidates(tm, s1, s2, s3, cfg) -> pd.DataFrame:
    parts = []
    for label, pool in (("S2", s2), ("S3", s3)):
        pairs, _ = generate_candidates(tm, s1, pool, cfg)
        pairs["source"] = label
        parts.append(pairs)
    return pd.concat(parts, ignore_index=True)


def candidates_to_mapping(pairs: pd.DataFrame, s1_ids) -> dict[str, list[str]]:
    grouped = pairs.groupby("s1_id")["cand_id"].apply(lambda x: sorted(set(x)))
    return {s: grouped.get(s, []) for s in s1_ids}
