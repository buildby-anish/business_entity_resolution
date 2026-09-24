"""Command-line entry point.  Run from the business_entity_resolution/ folder:

    python -m src.run_pipeline split     --data-dir dataset        # 1) make + save the entity-level split
    python -m src.run_pipeline validate  --data-dir dataset        # 2) train on train part, score + tune threshold on validation
    python -m src.run_pipeline final     --data-dir dataset        # 3) retrain on train+validation, predict test, write output/
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .candidate_generation import build_candidates, candidates_to_mapping
from .config import Config
from .data_loader import Dataset, load_dataset
from .dataset_builder import label_pairs, sample_training_pairs
from .features import FEATURES, compute_features
from .inference import select_matches
from .metrics import blocking_metrics, sweep_thresholds
from .output import CAND_COLS, MATCH_COLS, self_check, write_id_lists
from .preprocessing import preprocess
from .splitting import apply_split, make_split
from .text_model import TextModel
from .train import predict_proba, save_bundle, train_model


@contextmanager
def timer(msg):
    t = time.time()
    print(f"[..] {msg}")
    yield
    print(f"[ok] {msg}  ({time.time() - t:.1f}s)")


def preprocess_dataset(ds: Dataset) -> Dataset:
    return Dataset(preprocess(ds.s1), preprocess(ds.s2), preprocess(ds.s3), ds.truth)


def fit_text_model(ds: Dataset, cfg: Config) -> TextModel:
    """TF-IDF statistics come from TRAIN text only (ml_rules #1/#15)."""
    names = pd.concat([ds.s1["name_norm"], ds.s2["name_norm"], ds.s3["name_norm"]])
    addrs = pd.concat([ds.s1["addr_norm"], ds.s2["addr_norm"], ds.s3["addr_norm"]])
    return TextModel(cfg).fit(names, addrs)


def featurize(ds: Dataset, tm: TextModel, cfg: Config) -> pd.DataFrame:
    """blocking -> FINAL candidates -> features. Identical code path for train / validation / test."""
    pairs = build_candidates(tm, ds.s1, ds.s2, ds.s3, cfg)
    return compute_features(pairs, ds.s1, {"S2": ds.s2, "S3": ds.s3})


def _grid(cfg: Config):
    return np.round(np.arange(cfg.thr_min, cfg.thr_max + 1e-9, cfg.thr_step), 4)


# --------------------------------------------------------------------------- commands
def _fracs(cfg: Config) -> dict:
    return {"train": cfg.train_frac, "val": cfg.val_frac, "holdout": cfg.holdout_frac}


def cmd_split(cfg: Config):
    raw = load_dataset(cfg.data_dir, "train")
    summary = make_split(raw, _fracs(cfg), cfg.seed, Path(cfg.work_dir) / "split")
    print(json.dumps(summary, indent=2))


def cmd_validate(cfg: Config):
    work = Path(cfg.work_dir)
    split_dir = work / "split"
    raw = load_dataset(cfg.data_dir, "train")
    if not (split_dir / "split_assignments.tsv").exists():
        make_split(raw, _fracs(cfg), cfg.seed, split_dir)
    train_raw, val_raw, holdout_raw = apply_split(raw, split_dir)
    train_ds, val_ds = preprocess_dataset(train_raw), preprocess_dataset(val_raw)
    print(f"train S1={len(train_ds.s1):,} S2={len(train_ds.s2):,} S3={len(train_ds.s3):,} | "
          f"val S1={len(val_ds.s1):,} S2={len(val_ds.s2):,} S3={len(val_ds.s3):,} | "
          f"holdout S1={len(holdout_raw.s1):,} (untouched - not used by this command)")

    with timer("fit TF-IDF on TRAIN only"):
        tm = fit_text_model(train_ds, cfg)
    with timer("train part: blocking + features"):
        f_tr = label_pairs(featurize(train_ds, tm, cfg), train_ds.truth)
    bm_train = blocking_metrics(f_tr, train_ds.truth, list(train_ds.s1["entity_id"]),
                                {"S2": len(train_ds.s2), "S3": len(train_ds.s3)})
    print("train blocking:", json.dumps(bm_train))
    with timer("train LightGBM"):
        model = train_model(sample_training_pairs(f_tr, cfg, cfg.seed), cfg)

    with timer("validation part: blocking + features (transform only, no fitting)"):
        f_va = label_pairs(featurize(val_ds, tm, cfg), val_ds.truth)
    val_ids = list(val_ds.s1["entity_id"])
    bm_val = blocking_metrics(f_va, val_ds.truth, val_ids, {"S2": len(val_ds.s2), "S3": len(val_ds.s3)})
    print("validation blocking:", json.dumps(bm_val))

    with timer("threshold sweep on validation (macro F0.5, singletons included)"):
        probs = predict_proba(model, f_va)
        sweep = sweep_thresholds(f_va, probs, val_ds.truth, val_ids, _grid(cfg))
    ordered = sweep.sort_values(["macro_f05", "t_high", "t_low"], ascending=False)
    best = ordered.iloc[0]
    single = sweep[sweep["t_low"] == sweep["t_high"]].sort_values("macro_f05", ascending=False).iloc[0]
    print(f"best two-threshold : t_low={best.t_low:.2f} t_high={best.t_high:.2f} macro F0.5={best.macro_f05:.4f} "
          f"(singletons {best.f05_singletons:.3f}, matched {best.f05_matched:.3f})")
    print(f"best single thresh.: t={single.t_low:.2f} macro F0.5={single.macro_f05:.4f}")

    out = work / "validation"
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    meta = {"t_low": float(best.t_low), "t_high": float(best.t_high), "val_macro_f05": float(best.macro_f05),
            "val_macro_f05_single_threshold": float(single.macro_f05), "features": FEATURES,
            "blocking_train": bm_train, "blocking_val": bm_val}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    save_bundle(out / "bundle.joblib", model, tm, cfg)
    print(f"saved -> {out}")


def cmd_final(cfg: Config):
    work = Path(cfg.work_dir)
    meta_path = work / "validation" / "meta.json"
    if not meta_path.exists():
        raise SystemExit("Run `validate` first: the threshold must come from validation, never from a guess.")
    meta = json.loads(meta_path.read_text())
    t_low, t_high = meta["t_low"], meta["t_high"]

    train_ds = preprocess_dataset(load_dataset(cfg.data_dir, "train"))       # train + validation together
    with timer("fit TF-IDF on ALL TRAIN data"):
        tm = fit_text_model(train_ds, cfg)
    with timer("all train: blocking + features"):
        f_tr = label_pairs(featurize(train_ds, tm, cfg), train_ds.truth)
    with timer("train final LightGBM"):
        model = train_model(sample_training_pairs(f_tr, cfg, cfg.seed), cfg)

    test_ds = preprocess_dataset(load_dataset(cfg.data_dir, "test"))
    test_s1 = list(test_ds.s1["entity_id"])
    with timer("TEST: blocking + features (transform only)"):
        f_te = featurize(test_ds, tm, cfg)
    probs = predict_proba(model, f_te)
    matches = select_matches(f_te, probs, test_s1, t_low, t_high)
    candidates = candidates_to_mapping(f_te, test_s1)         # exactly what the model scored

    out = Path(cfg.output_dir)
    write_id_lists(out / "matching_results.tsv", test_s1, matches, MATCH_COLS)
    write_id_lists(out / "candidate_pairs.tsv", test_s1, candidates, CAND_COLS)
    save_bundle(work / "final_bundle.joblib", model, tm, cfg)

    issues = self_check(out / "matching_results.tsv", out / "candidate_pairs.tsv", test_s1,
                        test_ds.s2["entity_id"], test_ds.s3["entity_id"])
    n_empty = sum(1 for v in matches.values() if not v)
    print(f"test S1={len(test_s1):,} | with >=1 match={len(test_s1) - n_empty:,} | empty={n_empty:,} "
          f"| thresholds t_low={t_low:.2f} t_high={t_high:.2f}")
    by_country = test_ds.s1.assign(has=[bool(matches[s]) for s in test_s1]).groupby("country")["has"].agg(["size", "sum"])
    print("per country (rows, rows with matches):\n", by_country.to_string())
    print("self-check:", "OK" if not issues else issues)
    print("Now run the OFFICIAL validator: python3 utils/validate_submission.py --matching output/matching_results.tsv "
          "--candidate output/candidate_pairs.tsv --test-dir dataset/test")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["split", "validate", "final"])
    p.add_argument("--data-dir", default="dataset")
    p.add_argument("--work-dir", default="artifacts")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-neg-sampling", action="store_true", help="train on ALL candidate pairs (more RAM)")
    a = p.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, output_dir=a.output_dir,
                 train_frac=a.train_frac, val_frac=a.val_frac, holdout_frac=a.holdout_frac,
                 seed=a.seed, neg_sampling=not a.no_neg_sampling)
    {"split": cmd_split, "validate": cmd_validate, "final": cmd_final}[a.command](cfg)


if __name__ == "__main__":
    main()
