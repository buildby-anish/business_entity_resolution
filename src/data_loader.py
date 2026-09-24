"""Stage 1: reading the TSV files correctly (tab separator, everything as string)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pd.DataFrame:
    # sep="\t" is mandatory (PS). dtype=str + keep_default_na=False keeps empty
    # addresses / empty matched_entity_ids as "" instead of NaN.
    # QUOTE_NONE: a stray " inside a business name must not swallow following rows.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def read_source(path) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)} "
                         f"(did you read it without sep='\\t'?)")
    df = df[SOURCE_COLS].copy()
    for c in SOURCE_COLS:
        df[c] = df[c].str.strip()
    if df["entity_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate entity_id values")
    return df.reset_index(drop=True)


def parse_ground_truth(df: pd.DataFrame, s1_ids) -> dict[str, set[str]]:
    """{S1 id -> set of matching S2/S3 ids}. Every S1 id gets an entry (empty set = singleton)."""
    truth: dict[str, set[str]] = {s: set() for s in s1_ids}
    for s1, ids in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        s1 = s1.strip()
        recs = {x.strip() for x in ids.split(",") if x.strip()}
        truth.setdefault(s1, set()).update(recs)
    return truth


@dataclass
class Dataset:
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    truth: dict[str, set[str]] | None = None   # None for the test set (no labels)


def load_dataset(data_dir, which: str) -> Dataset:
    """which = 'train' or 'test'. Layout follows the PS: dataset/<which>/<which>_source{1,2,3}.tsv"""
    d = Path(data_dir) / which
    s1 = read_source(d / f"{which}_source1.tsv")
    s2 = read_source(d / f"{which}_source2.tsv")
    s3 = read_source(d / f"{which}_source3.tsv")
    truth = None
    if which == "train":
        gt = read_tsv(d / "train_ground_truth.tsv")
        truth = parse_ground_truth(gt, s1["entity_id"])
        unknown = set(truth) - set(s1["entity_id"])
        if unknown:
            print(f"[warn] ground truth mentions {len(unknown)} S1 ids that are not in train_source1")
            for u in unknown:
                truth.pop(u)
    return Dataset(s1, s2, s3, truth)
