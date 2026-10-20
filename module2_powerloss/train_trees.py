"""Classical regressors for power loss, plus the baselines they must beat.

Models (all trained on the Module 1 train split, scored on its test split):
  mean_baseline   always predicts the training-set mean loss
  env_xgboost     XGBoost on environment only (irradiance, time of day): no image
  random_forest   Random Forest on PCA-reduced Module 1 embeddings + environment
  xgboost         XGBoost on the same features

Writes outputs/tree_test_predictions.csv and outputs/xgboost.json.

Usage:  python -m module2_powerloss.train_trees   (run features.py first)
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from . import config
from .data import load_index


def load_features():
    df = load_index()
    cache = np.load(config.EMBEDDINGS, allow_pickle=True)
    emb = pd.DataFrame(cache["embeddings"], index=cache["paths"])
    X_img = emb.loc[df["path"]].to_numpy()
    return df, X_img


def mae(y, p):
    return float(np.abs(y - p).mean())


def xgb(device):
    return XGBRegressor(
        n_estimators=5000, learning_rate=0.03, max_depth=7, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3, reg_lambda=1.0,
        tree_method="hist", device=device, early_stopping_rounds=100,
        random_state=config.SEED,
    )


def main():
    import torch
    # XGBoost warns when a GPU-trained booster predicts on CPU arrays; results are identical.
    warnings.filterwarnings("ignore", message=".*device.*", category=UserWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, X_img = load_features()
    split = df["split"].to_numpy()
    tr, va, te = (split == s for s in ("train", "val", "test"))
    y = df["power_loss_pct"].to_numpy()
    X_env = df[config.ENV_FEATURES].to_numpy()

    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.SEED).fit(X_img[tr])
    print(f"PCA {config.PCA_COMPONENTS} components keep "
          f"{pca.explained_variance_ratio_.sum():.1%} of embedding variance")
    X = np.hstack([pca.transform(X_img), X_env])

    preds = {"mean_baseline": np.full(te.sum(), y[tr].mean())}

    t0 = time.time()
    m = xgb(device).fit(X_env[tr], y[tr], eval_set=[(X_env[va], y[va])], verbose=False)
    preds["env_xgboost"] = m.predict(X_env[te])
    print(f"env_xgboost   val MAE {mae(y[va], m.predict(X_env[va])):.2f}  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=2,
                               n_jobs=-1, random_state=config.SEED).fit(X[tr], y[tr])
    preds["random_forest"] = rf.predict(X[te])
    print(f"random_forest val MAE {mae(y[va], rf.predict(X[va])):.2f}  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    m = xgb(device).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    preds["xgboost"] = m.predict(X[te])
    print(f"xgboost       val MAE {mae(y[va], m.predict(X[va])):.2f}  "
          f"({m.best_iteration} trees, {time.time() - t0:.0f}s)")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    m.save_model(config.OUT_DIR / "xgboost.json")

    out = df.loc[te, ["path", "power_loss_pct", "severity", "irradiance"]].reset_index(drop=True)
    for k, v in preds.items():
        out[k] = np.clip(v, 0, 100)
    out.to_csv(config.OUT_DIR / "tree_test_predictions.csv", index=False)
    for k in preds:
        print(f"{k:14s} test MAE {mae(out['power_loss_pct'], out[k]):.2f}")


if __name__ == "__main__":
    main()
