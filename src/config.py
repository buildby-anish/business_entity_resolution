"""All tunable knobs in one place. Nothing here is required by the PS."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- paths (PS layout: dataset/train/*.tsv, dataset/test/*.tsv, output/*.tsv)
    data_dir: str = "dataset"
    work_dir: str = "artifacts"      # split files, fitted text model, LightGBM model, reports
    output_dir: str = "output"       # matching_results.tsv, candidate_pairs.tsv

    # ---- split (your ml_rules: split by S1 entity, stratify, move S2/S3 with their S1)
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    holdout_frac: float = 0.15

    # ---- blocking
    require_same_country: bool = True      # verify on train first (see README, Step 0)
    char_ngram: tuple = (3, 4)
    n_hash_features: int = 2 ** 20
    retrieval_max_df_frac: float = 0.05    # tokens/n-grams more common than this are ignored for RETRIEVAL
    retrieval_min_abs_df: int = 30
    k_name_word: int = 30
    k_addr_word: int = 30
    k_name_char: int = 30
    k_final: int = 30                      # FINAL candidates kept per S1 per source (S2 and S3 separately)
    w_name: float = 0.6                    # cheap_score = w_name*name_char_cos + (1-w_name)*addr_char_cos
    chunk_rows: int = 1000                 # S1 rows per sparse-matmul chunk
    pair_chunk: int = 200_000              # pairs per row-wise cosine chunk

    # ---- training-pair sampling (threshold is tuned on UNSAMPLED validation pairs)
    neg_sampling: bool = True
    neg_hard_per_s1: int = 15              # highest cheap_score negatives per S1
    neg_random_per_s1: int = 10            # random extra negatives per S1

    # ---- model
    lgbm_params: dict = field(default_factory=lambda: dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    ))

    # ---- threshold search grid
    thr_min: float = 0.30
    thr_max: float = 0.99
    thr_step: float = 0.02
