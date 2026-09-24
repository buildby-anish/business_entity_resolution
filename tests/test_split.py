"""Leakage guard for the entity-level split.  Run:  python -m tests.test_split
Needs a dataset in the PS layout (use scripts/make_synthetic_data.py) -> path in --data-dir."""
import argparse
import tempfile

from src.data_loader import load_dataset
from src.splitting import apply_split, make_split

FRACS = {"train": 0.70, "val": 0.15, "holdout": 0.15}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="dataset")
    a = p.parse_args()
    ds = load_dataset(a.data_dir, "train")
    with tempfile.TemporaryDirectory() as d:
        make_split(ds, FRACS, 42, d)
        tr, va, ho = apply_split(ds, d)
        parts = {"train": tr, "val": va, "holdout": ho}

        s1_sets = {side: set(p.s1.entity_id) for side, p in parts.items()}
        assert not (s1_sets["train"] & s1_sets["val"]), "S1 entity in both train and val"
        assert not (s1_sets["train"] & s1_sets["holdout"]), "S1 entity in both train and holdout"
        assert not (s1_sets["val"] & s1_sets["holdout"]), "S1 entity in both val and holdout"
        assert s1_sets["train"] | s1_sets["val"] | s1_sets["holdout"] == set(ds.s1.entity_id), "S1 entity lost"

        pools = {side: set(p.s2.entity_id) | set(p.s3.entity_id) for side, p in parts.items()}
        for side, part in parts.items():
            for s1, recs in part.truth.items():
                assert recs <= pools[side], f"{side}: a true match of {s1} is missing from its own pool"
                # a record CAN legitimately appear in another side's pool too, but only if it is
                # matched to an S1 entity that lives on that other side (see splitting.py docstring)
                for other, other_pool in pools.items():
                    if other == side:
                        continue
                    leaked = recs & other_pool
                    if not leaked:
                        continue
                    for rec in leaked:
                        owners = {s for s, rs in ds.truth.items() if rec in rs}
                        assert owners & s1_sets[other], (
                            f"{side}: a true match of {s1} leaked into {other}'s pool "
                            f"without a matching S1 owner there")
        print("split OK: disjoint S1 across train/val/holdout, matches travel with their S1, "
              "any pool overlap is explained by a shared owner")


if __name__ == "__main__":
    main()
