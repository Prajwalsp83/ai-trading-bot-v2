"""
Phase 5b — LightGBM meta-labeler training.

Reads data/ml_dataset.parquet (or .csv), trains a binary classifier that
predicts P(trade wins | features), and writes:

    models/meta_labeler.pkl       — trained LightGBM model
    models/meta_labeler.meta.json — feature list + chosen threshold + metrics

Training methodology:
  1. Chronological train/val split (no random shuffle — would leak future)
  2. 3-fold walk-forward CV for sanity check
  3. Final model trained on first 80%, validated on last 20%
  4. Threshold search: pick the *most permissive* threshold that holds
     a >= TARGET_WIN_RATE win rate on the validation set
  5. Report feature importances, confusion matrix, P&L sim

Run:
    pip install lightgbm scikit-learn pyarrow
    python scripts/train_meta_labeler.py
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix


HERE = Path(__file__).resolve().parent.parent
DATA_PATH_PARQUET = HERE / "data" / "ml_dataset.parquet"
DATA_PATH_CSV     = HERE / "data" / "ml_dataset.csv"
MODELS_DIR        = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH        = MODELS_DIR / "meta_labeler.pkl"
META_PATH         = MODELS_DIR / "meta_labeler.meta.json"

# Threshold tuning target
TARGET_WIN_RATE = 0.40
THRESHOLD_GRID = np.arange(0.30, 0.91, 0.01).round(2)

# Categorical features (LightGBM handles natively)
CATEGORICAL_COLS = ["strategy", "side", "session", "regime"]

# Drop these columns when building feature matrix (not predictive / leak risk)
DROP_COLS = ["ts_iso", "close", "entry", "sl", "tp", "label",
             "exit_reason", "bars_held"]


# ============================ HELPERS ===============================
def load_dataset() -> pd.DataFrame:
    if DATA_PATH_PARQUET.exists():
        df = pd.read_parquet(DATA_PATH_PARQUET)
        src = DATA_PATH_PARQUET
    elif DATA_PATH_CSV.exists():
        df = pd.read_csv(DATA_PATH_CSV)
        src = DATA_PATH_CSV
    else:
        print(f"ERROR: no dataset at {DATA_PATH_PARQUET} or {DATA_PATH_CSV}")
        print("Run: python scripts/generate_ml_dataset.py --months 6")
        sys.exit(1)
    print(f"Loaded {len(df)} rows from {src}")
    df["ts"] = pd.to_datetime(df["ts_iso"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame):
    """Split into X (feature matrix), y (labels), ts (timestamps).
    Encodes categorical cols as Categorical dtype for LightGBM."""
    y = df["label"].astype(int)
    ts = df["ts"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + ["ts"], errors="ignore")
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, y, ts


def chronological_split(X: pd.DataFrame, y: pd.Series, ts: pd.Series,
                        train_frac: float = 0.80):
    """Split: first train_frac of bars -> train, rest -> val."""
    n = len(X)
    cutoff = int(n * train_frac)
    return (X.iloc[:cutoff], y.iloc[:cutoff], ts.iloc[:cutoff],
            X.iloc[cutoff:], y.iloc[cutoff:], ts.iloc[cutoff:])


def train_lgbm(X_train, y_train, X_val, y_val) -> lgb.Booster:
    """Train with sensible defaults for tabular financial data."""
    cat_features = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_set   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_features,
                             reference=train_set)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    booster = lgb.train(
        params, train_set,
        num_boost_round=500,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=50)],
    )
    return booster


def walk_forward_cv(X: pd.DataFrame, y: pd.Series, n_folds: int = 3):
    """Chronological CV: expand training window, validate on the next slice."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    aucs = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        val_end   = fold_size * (k + 1)
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_va, y_va = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        if y_va.nunique() < 2:
            continue
        booster = train_lgbm(X_tr, y_tr, X_va, y_va)
        probs = booster.predict(X_va, num_iteration=booster.best_iteration)
        auc = roc_auc_score(y_va, probs)
        aucs.append(auc)
        print(f"  fold {k}/{n_folds}: train_size={len(X_tr)} val_size={len(X_va)} AUC={auc:.4f}")
    return aucs


def find_best_threshold(y_val: pd.Series, probs: np.ndarray,
                        target_win_rate: float = 0.40):
    """Find the most permissive threshold that holds win rate >= target_win_rate.

    Returns (chosen_threshold, summary_table, reason). Adaptive grid handles
    cases where model probs are all very low (e.g. when base WIN rate is low)."""
    # Adaptive grid: cover actual prob distribution + the standard 0.30-0.90 range
    grid = sorted(set(np.round(np.concatenate([
        THRESHOLD_GRID,
        np.quantile(probs, np.linspace(0.05, 0.95, 19)),
    ]), 3)))

    rows = []
    for t in grid:
        keep = probs >= t
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue
        win_rate = float(y_val[keep].mean())
        expectancy = win_rate * 1.67 - (1 - win_rate) * 1.0
        rows.append({
            "threshold": float(t),
            "n_kept": n_keep,
            "win_rate": round(win_rate, 4),
            "expectancy_R": round(expectancy, 4),
        })

    if not rows:
        # Should never happen with adaptive grid, but defensive
        return float(probs.max()), pd.DataFrame(columns=["threshold","n_kept","win_rate","expectancy_R"]), \
               f"no rows; using max prob {probs.max():.3f}"

    summary = pd.DataFrame(rows)

    qualified = summary[summary["win_rate"] >= target_win_rate]
    if len(qualified) > 0:
        # Most permissive (lowest threshold) among qualifying
        best = qualified.sort_values("threshold").iloc[0]
        chosen = float(best["threshold"])
        reason = f"lowest threshold with WR >= {target_win_rate*100:.0f}%"
    else:
        # Fallback: highest-expectancy threshold (which may still be negative)
        best = summary.sort_values("expectancy_R", ascending=False).iloc[0]
        chosen = float(best["threshold"])
        reason = (f"no threshold met {target_win_rate*100:.0f}% target — "
                  f"using max-expectancy threshold ({best['expectancy_R']:+.3f} R)")
        print(f"  WARNING: {reason}")
        if best["expectancy_R"] < 0:
            print(f"  WARNING: even the best threshold has NEGATIVE expectancy. "
                  f"ML model has no useful edge on this data.")

    return chosen, summary, reason


# =============================== MAIN ================================
def main() -> int:
    df = load_dataset()
    X, y, ts = build_features(df)
    print(f"Features: {list(X.columns)} ({X.shape[1]} cols)")
    print(f"Label balance: WIN={y.sum()} ({y.mean()*100:.1f}%) / LOSS={len(y)-y.sum()}")

    # Walk-forward CV — sanity check that AUC is consistently above 0.5
    print("\n=== Walk-forward CV (3 folds) ===")
    cv_aucs = walk_forward_cv(X, y, n_folds=3)
    if cv_aucs:
        print(f"  mean AUC: {np.mean(cv_aucs):.4f}  std: {np.std(cv_aucs):.4f}")
        if np.mean(cv_aucs) < 0.52:
            print("  WARNING: AUC < 0.52 — model has barely any signal. "
                  "Either features are weak or strategies are unedge.")

    # Final train: 80% train / 20% val
    print("\n=== Final training (80/20 chronological split) ===")
    X_tr, y_tr, ts_tr, X_va, y_va, ts_va = chronological_split(X, y, ts)
    print(f"  Train: {len(X_tr)} rows ({ts_tr.iloc[0]} -> {ts_tr.iloc[-1]})")
    print(f"  Val:   {len(X_va)} rows ({ts_va.iloc[0]} -> {ts_va.iloc[-1]})")
    print(f"  Train WIN rate: {y_tr.mean()*100:.1f}%")
    print(f"  Val   WIN rate: {y_va.mean()*100:.1f}%")

    booster = train_lgbm(X_tr, y_tr, X_va, y_va)
    val_probs = booster.predict(X_va, num_iteration=booster.best_iteration)
    val_auc = roc_auc_score(y_va, val_probs)
    val_logloss = log_loss(y_va, val_probs)
    print(f"\n  Final val AUC:     {val_auc:.4f}")
    print(f"  Final val logloss: {val_logloss:.4f}")

    # Threshold search
    print(f"\n=== Threshold search (target WR >= {TARGET_WIN_RATE*100:.0f}%) ===")
    chosen_thr, thr_summary, reason = find_best_threshold(y_va, val_probs, TARGET_WIN_RATE)
    print(f"  Chosen threshold: {chosen_thr:.2f}  ({reason})")
    print(f"\n  Threshold summary (val set):")
    print(thr_summary.to_string(index=False))

    # Confusion at chosen threshold
    preds_at_thr = (val_probs >= chosen_thr).astype(int)
    if preds_at_thr.sum() > 0:
        cm = confusion_matrix(y_va, preds_at_thr)
        print(f"\n  Confusion (threshold={chosen_thr:.2f}):")
        print(f"           pred_LOSS  pred_WIN")
        print(f"  LOSS:    {cm[0,0]:>9}  {cm[0,1]:>8}")
        print(f"  WIN:     {cm[1,0]:>9}  {cm[1,1]:>8}")

    # Feature importance
    print(f"\n=== Top 15 features (gain importance) ===")
    fi = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    print(fi.head(15).to_string(index=False))

    # Persist
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(booster, f)

    meta = {
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "n_train": len(X_tr), "n_val": len(X_va),
        "features": list(X.columns),
        "categorical_features": [c for c in CATEGORICAL_COLS if c in X.columns],
        "chosen_threshold": chosen_thr,
        "threshold_reason": reason,
        "val_auc": float(val_auc),
        "val_logloss": float(val_logloss),
        "cv_aucs": [float(a) for a in cv_aucs],
        "threshold_summary": thr_summary.to_dict(orient="records"),
        "target_win_rate": TARGET_WIN_RATE,
        "tp_sl_ratio_assumed": 1.67,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n=== Saved ===")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Metadata: {META_PATH}")
    print(f"\nIn live bot, filter signals where score < {chosen_thr:.2f} (P(win) below threshold).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
