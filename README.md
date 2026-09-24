# business_entity_resolution — baseline pipeline

Baseline for the Amazon ML Challenge *Business Entity Resolution* PS: for every Source 1 (S1) test entity,
find the matching Source 2 / Source 3 records. Output = `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

## Layout

```
business_entity_resolution/
├── src/
│   ├── config.py               all knobs
│   ├── data_loader.py          TSV reading (sep="\t"), ground-truth parsing
│   ├── preprocessing.py        rule-based, layered normalisation (nothing learned)
│   ├── splitting.py            entity-level grouped + stratified split (your ml_rules)
│   ├── text_model.py           TF-IDF fitted on TRAIN only (hashed, so unseen words/countries still work)
│   ├── blocking.py             3-channel retrieval + cheap rerank -> FINAL candidates
│   ├── candidate_generation.py S2 + S3 candidates, candidate_pairs mapping
│   ├── features.py             pairwise features (fixed column order)
│   ├── dataset_builder.py      ground truth -> labelled pairs, negative sampling (train only)
│   ├── metrics.py              PS-exact macro F0.5, blocking metrics, threshold sweep
│   ├── train.py                LightGBM (MIT)
│   ├── inference.py            probabilities -> matches (two-threshold singleton rule)
│   ├── output.py               TSV writers + local self-check
│   └── run_pipeline.py         CLI: split | validate | final
├── scripts/make_synthetic_data.py   FAKE data in the PS layout for smoke tests
├── tests/test_metrics.py, tests/test_split.py
├── requirements.txt
└── README.md
```

## Run (from this folder)

```bash
pip install -r requirements.txt

# 0) smoke test on FAKE data (2 minutes)
python scripts/make_synthetic_data.py --out dataset_fake
python -m tests.test_metrics
python -m tests.test_split --data-dir dataset_fake
python -m src.run_pipeline split    --data-dir dataset_fake
python -m src.run_pipeline validate --data-dir dataset_fake
python -m src.run_pipeline final    --data-dir dataset_fake

# real data: expects dataset/train/*.tsv and dataset/test/*.tsv (PS layout)
python -m src.run_pipeline split
python -m src.run_pipeline validate      # prints blocking recall + macro F0.5, saves thresholds
python -m src.run_pipeline final         # retrains on train+validation, writes output/*.tsv

# then the OFFICIAL validator from student_resource/
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## How your ml_rules are applied

| Rule | Where |
|---|---|
| Split by S1 entity, stratified (country x match-count bucket 0/1/2+), 70/15/15 train/val/holdout | `splitting.make_split` |
| Matched S2/S3 move with their S1 (kept on every side a shared record's owners land on, with a warning); orphans split 70/15/15 per country | `splitting.make_split` |
| Save the split once (`artifacts/split/*.txt`) | `run_pipeline split` |
| Fit on train, transform validation/test (#1, #15) | `text_model.TextModel.fit` only ever sees train text |
| Same feature pipeline + fixed column order (#17, #18) | `featurize()` and `features.FEATURES` |
| No target/ID in features (#3, #13) | `features.py` uses text only |
| Threshold chosen on validation, never on test (#2) | `validate` writes `meta.json`; `final` only reads it |
| No SMOTE, no forced 50/50 balance (#6, #7) | negatives are down-sampled on TRAIN only; validation stays natural |
| Retrain on train+validation for the final model | `run_pipeline final` |

## PS rules this code respects

* TSV read/written with tabs; one row per S1 test entity; empty list for singletons; sorted, de-duplicated S2/S3 ids only.
* `candidate_pairs.tsv` = the exact final candidate set the model scored; matches are a subset by construction.
* Country is an open-set string (`country_match` only). No hard-coded US/India, nothing is dropped (France is handled).
* No external data, APIs, geocoding or lookups. Model: LightGBM (MIT), far below 8B parameters.

## Known limits of this baseline (next things to improve)

1. Feature loops are plain Python; for ~1M S1 run them in chunks with `joblib` and cache Parquet between stages.
2. Blocking has no name-prefix channel and no address-only rescue for DBA names — inspect missed true pairs on real data.
3. Abbreviation map is tiny; grow it from what you see in train (never from external sources).
4. Positives missed by blocking are not injected into training.
5. Check on real train data: do true pairs always share country? (`require_same_country`). Can an S2/S3 record match >1 S1?
6. Fill `Documentation_template.md`; pin versions in `requirements.txt` (`pip freeze`).
