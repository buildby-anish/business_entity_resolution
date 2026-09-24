"""Stage 3: entity-level grouped + stratified 70/15/15 TRAIN/VAL/HOLDOUT split (your ml_rules).

1. Split the S1 ids (NOT rows / pairs) into three groups: train, val, holdout. One S1 id = one
   group, so its candidate pairs can never end up on more than one side by accident.
2. Stratify on country x match-count bucket (0 / 1 / 2+) so every pool has a realistic mix of
   singletons and multi-match entities.
3. Move S2/S3 records with their S1:
     matched to a train S1     -> train pool
     matched to a val S1       -> val pool
     matched to a holdout S1   -> holdout pool
     matched to S1 ids on MORE THAN ONE side (rare - one real-world business record matched to
       several S1 entities) -> kept in every side its owners belong to, with a [warn]. This is
       the ml_rules-safe choice: dropping it from one side would make that side's ground truth
       unreachable and silently wrong; the alternative (dropping the S1<->record link instead)
       would require re-deriving ground truth, which we do not do here.
     orphans (match nobody)    -> split deterministically, proportional to the fracs, per country
4. Save one assignment file (entity_id, source, country, split, role); every experiment reuses it.

holdout is meant to stay UNTOUCHED for validate/tuning (that's val's job). cmd_final currently
retrains on the full train file regardless of split, so it already includes holdout; using
holdout for a separate final sanity check is a later stage (see the project brief, item 5) -
not done here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import Dataset

SPLITS = ("train", "val", "holdout")


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2+"


def _strata(s1: pd.DataFrame, truth, min_count: int = 4) -> pd.Series:
    n_true = s1["entity_id"].map(lambda x: len(truth.get(x, ())))
    strata = s1["country"].str.strip().str.casefold() + "|" + n_true.map(_bucket)
    counts = strata.value_counts()
    return strata.where(strata.map(counts) >= min_count, "rare")   # merge strata too small to stratify


def _split_s1_ids(s1: pd.DataFrame, truth, fracs: dict, seed: int) -> dict[str, str]:
    """Two sequential splits: all S1 -> train vs rest, then rest -> val vs holdout."""
    ids = s1["entity_id"].to_numpy(dtype=object)
    strata = _strata(s1, truth)

    try:
        train_ids, rest_ids = train_test_split(
            ids, train_size=fracs["train"], random_state=seed, stratify=strata.to_numpy(dtype=object))
    except ValueError:
        print("[warn] stratified train/rest split failed (too few examples per stratum); using plain random split")
        train_ids, rest_ids = train_test_split(ids, train_size=fracs["train"], random_state=seed)

    rest_mask = s1["entity_id"].isin(set(rest_ids))
    rest_strata = strata[rest_mask]
    val_share = fracs["val"] / (fracs["val"] + fracs["holdout"])
    try:
        if (rest_strata.value_counts() < 2).any():
            raise ValueError("stratum too small")
        val_ids, hold_ids = train_test_split(
            rest_ids, train_size=val_share, random_state=seed, stratify=rest_strata.to_numpy(dtype=object))
    except ValueError:
        print("[warn] stratified val/holdout split failed (too few examples per stratum); using plain random split")
        val_ids, hold_ids = train_test_split(rest_ids, train_size=val_share, random_state=seed)

    split_of_s1 = {i: "train" for i in train_ids}
    split_of_s1.update({i: "val" for i in val_ids})
    split_of_s1.update({i: "holdout" for i in hold_ids})
    assert set(split_of_s1) == set(ids)
    return split_of_s1


def make_split(ds: Dataset, fracs: dict, seed: int, out_dir) -> dict:
    if abs(sum(fracs.values()) - 1.0) > 1e-9:
        raise ValueError(f"fracs must sum to 1.0, got {fracs}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    split_of_s1 = _split_s1_ids(ds.s1, ds.truth, fracs, seed)

    # who "owns" each matched S2/S3 record - usually one S1, occasionally several
    owners: dict[str, set[str]] = {}
    for s1_id, recs in ds.truth.items():
        for r in recs:
            owners.setdefault(r, set()).add(s1_id)

    rows = [(r.entity_id, "S1", r.country, split_of_s1[r.entity_id], "reference") for r in ds.s1.itertuples()]

    summary = {f"n_s1_{s}": int((pd.Series(split_of_s1) == s).sum()) for s in SPLITS}
    shared_warn = 0
    for label, df in (("S2", ds.s2), ("S3", ds.s3)):
        matched_n = orphan_n = 0
        orphan_rows = []
        for r in df.itertuples():
            o = owners.get(r.entity_id)
            if o is None:
                orphan_rows.append(r)
                continue
            matched_n += 1
            sides = {split_of_s1[s1_id] for s1_id in o if s1_id in split_of_s1}
            if len(sides) > 1:
                shared_warn += 1
            for side in sides:
                rows.append((r.entity_id, label, r.country, side, "matched"))
        orphan_n = len(orphan_rows)
        # orphans: deterministic proportional split per country (never touches ground truth)
        by_country: dict[str, list] = {}
        for r in orphan_rows:
            by_country.setdefault(r.country, []).append(r.entity_id)
        for country, eids in by_country.items():
            eids = np.array(eids, dtype=object)
            rng.shuffle(eids)
            n = len(eids)
            n_train = int(round(fracs["train"] * n))
            n_val = int(round(fracs["val"] * n))
            for eid in eids[:n_train]:
                rows.append((eid, label, country, "train", "orphan"))
            for eid in eids[n_train:n_train + n_val]:
                rows.append((eid, label, country, "val", "orphan"))
            for eid in eids[n_train + n_val:]:                 # remainder -> holdout, absorbs rounding drift
                rows.append((eid, label, country, "holdout", "orphan"))
        summary[f"{label.lower()}_matched"] = matched_n
        summary[f"{label.lower()}_orphans"] = orphan_n

    if shared_warn:
        print(f"[warn] {shared_warn} S2/S3 records are matched to S1 entities on more than one side; "
              f"kept in every side they belong to (see docstring).")
    summary["records_kept_on_multiple_sides"] = shared_warn

    out = pd.DataFrame(rows, columns=["entity_id", "source", "country", "split", "role"])
    out.to_csv(out_dir / "split_assignments.tsv", sep="\t", index=False)
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def apply_split(ds: Dataset, split_dir) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train_ds, val_ds, holdout_ds) built from split_assignments.tsv."""
    path = Path(split_dir) / "split_assignments.tsv"
    assign = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    ids_by_split = {s: set(assign.loc[assign["split"] == s, "entity_id"]) for s in SPLITS}

    parts = {}
    for side in SPLITS:
        ids = ids_by_split[side]
        s1 = ds.s1[ds.s1["entity_id"].isin(ids)].reset_index(drop=True)
        s2 = ds.s2[ds.s2["entity_id"].isin(ids)].reset_index(drop=True)
        s3 = ds.s3[ds.s3["entity_id"].isin(ids)].reset_index(drop=True)
        truth = {s: ds.truth[s] for s in s1["entity_id"]}
        parts[side] = Dataset(s1, s2, s3, truth)

    s1_sets = {side: set(parts[side].s1["entity_id"]) for side in SPLITS}
    assert not (s1_sets["train"] & s1_sets["val"]), "S1 leakage: train/val"
    assert not (s1_sets["train"] & s1_sets["holdout"]), "S1 leakage: train/holdout"
    assert not (s1_sets["val"] & s1_sets["holdout"]), "S1 leakage: val/holdout"
    assert s1_sets["train"] | s1_sets["val"] | s1_sets["holdout"] == set(ds.s1["entity_id"]), "S1 id lost"
    return parts["train"], parts["val"], parts["holdout"]
