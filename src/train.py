"""Stage 8: LightGBM (MIT licence) on the pairwise features. Trees need no feature scaling (Golden Rule #10)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES


def train_model(train_df: pd.DataFrame, cfg):
    import lightgbm as lgb
    X, y = train_df[FEATURES], train_df["label"].values
    print(f"[train] rows={len(X):,} positives={int(y.sum()):,} ({y.mean():.1%})")
    model = lgb.LGBMClassifier(**cfg.lgbm_params)
    model.fit(X, y)
    return model


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty(0)
    X = df[FEATURES]                       # fixed column order (Golden Rule #18)
    return model.predict_proba(X)[:, 1]


def save_bundle(path, model, text_model, cfg) -> None:
    joblib.dump({"model": model, "text_model": text_model, "features": FEATURES, "cfg": cfg}, path)


def load_bundle(path) -> dict:
    return joblib.load(path)
