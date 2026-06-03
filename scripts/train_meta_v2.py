"""
Phase G.2 — Retrain meta-labeler on combined backtest dataset.

Reads data/ml_dataset_combined.parquet (1,350 samples from 4 strategies).
Trains HistGradientBoostingClassifier with chronological train/val split.
Compares val AUC to the existing model (0.75). If better, swaps the .pkl.

The deployed live bot hot-reloads on file mtime change (next restart picks up
the new model). Falls back to old model if anything breaks.

Run:
    python scripts/train_meta_v2.py
    python scripts/train_meta_v2.py --no-replace   # just train, don't overwrite
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
from sklearn.inspection import permutation_importance


HERE = Path(__file__).resolve().parent.parent
DATASET = HERE / "data" / "ml_dataset_combined.parquet"
MODELS_DIR = HERE / "models"
MODEL_PATH = MODELS_DIR / "meta_labeler.pkl"
META_PATH = MODELS_DIR / "meta_labeler.meta.json"
BACKUP_MODEL = MODELS_DIR / "meta_labeler.prev.pkl"
BACKUP_META = MODELS_DIR / "meta_labeler.prev.meta.json"

CAT_COLS = ["strategy", "side", "session", "regime"]
DROP_COLS = ["ts_open", "label"]


# ============================== HELPERS =============================
def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts_open" in df.columns:
        df["ts_open"] = pd.to_datetime(df["ts_open"], utc=True)
        df = df.sort_values("ts_open").reset_index(drop=True)
    return df


def _encode(X_train, X_val):
    cat = [c for c in CAT_COLS if c in X_train.columns]
    if not cat:
        return X_train, X_val, None
    enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                          unknown_value=-1, dtype="float64")
    enc.fit(X_train[cat].astype(str))
    X_train = X_train.copy(); X_val = X_val.copy()
    X_train[cat] = enc.transform(X_train[cat].astype(str))
    X_val[cat] = enc.transform(X_val[cat].astype(str))
    return X_train, X_val, enc


def _train_one(X_tr, y_tr, X_va, y_va):
    X_tr, X_va, enc = _encode(X_tr, X_va)
    m = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
        verbose=0,
    )
    m.fit(X_tr, y_tr)
    return m, enc


def _wf_cv(X, y, ts, n_folds=3):
    """Chronological walk-forward CV: expand training window each fold."""
    n = len(X)
    fold = n // (n_folds + 1)
    aucs = []
    for k in range(1, n_folds + 1):
        train_end = fold * k
        val_end = fold * (k + 1)
        X_tr = X.iloc[:train_end]; y_tr = y.iloc[:train_end]
        X_va = X.iloc[train_end:val_end]; y_va = y.iloc[train_end:val_end]
        if y_va.nunique() < 2:
            continue
        m, enc = _train_one(X_tr, y_tr, X_va, y_va)
        Xv = X_va.copy()
        if enc is not None:
            cat = [c for c in CAT_COLS if c in Xv.columns]
            Xv[cat] = enc.transform(Xv[cat].astype(str))
        probs = m.predict_proba(Xv)[:, 1]
        auc = roc_auc_score(y_va, probs)
        aucs.append(auc)
        print(f"  fold {k}: tr={len(X_tr):>4} va={len(X_va):>4}  AUC={auc:.4f}")
    return aucs


def _threshold_table(y_val, probs):
    rows = []
    grid = sorted(set(np.round(np.concatenate([
        np.arange(0.20, 0.91, 0.01),
        np.quantile(probs, np.linspace(0.05, 0.95, 19)),
    ]), 3)))
    for t in grid:
        keep = probs >= t
        n = int(keep.sum())
        if n == 0:
            continue
        wr = float(y_val[keep].mean())
        rows.append({
            "threshold": float(t), "n_kept": n,
            "win_rate": round(wr, 4),
            "expectancy_R": round(wr * 1.67 - (1 - wr) * 1.0, 4),
        })
    return pd.DataFrame(rows)


# ============================== MAIN ================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-replace", action="store_true",
                   help="Don't overwrite the live model (just train + report)")
    p.add_argument("--target-wr", type=float, default=0.40,
                   help="Target win rate for threshold selection")
    args = p.parse_args()

    if not DATASET.exists():
        print(f"ERROR: {DATASET} not found. Run build_combined_ml_dataset.py first.")
        return 1

    df = _load(DATASET)
    print(f"Loaded {len(df)} samples from {DATASET}")
    print(f"WIN rate overall: {df['label'].mean()*100:.1f}%")
    if "ts_open" in df.columns:
        print(f"Span: {df['ts_open'].iloc[0]} -> {df['ts_open'].iloc[-1]}")

    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    ts = df["ts_open"] if "ts_open" in df.columns else pd.Series(range(len(df)))

    print(f"Features: {list(X.columns)} ({X.shape[1]} cols)")

    # === Walk-forward CV ===
    print("\n=== Walk-forward CV (3 folds, chronological) ===")
    aucs = _wf_cv(X, y, ts, n_folds=3)
    if aucs:
        print(f"  mean AUC: {np.mean(aucs):.4f}  std: {np.std(aucs):.4f}")

    # === Final 80/20 split ===
    print("\n=== Final training (80/20 chronological) ===")
    cutoff = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:cutoff], y.iloc[:cutoff]
    X_va, y_va = X.iloc[cutoff:], y.iloc[cutoff:]
    print(f"  train: {len(X_tr)} rows  ({ts.iloc[0]} -> {ts.iloc[cutoff-1]})")
    print(f"  val:   {len(X_va)} rows  ({ts.iloc[cutoff]} -> {ts.iloc[-1]})")
    print(f"  train WIN rate: {y_tr.mean()*100:.1f}%")
    print(f"  val   WIN rate: {y_va.mean()*100:.1f}%")

    model, enc = _train_one(X_tr, y_tr, X_va, y_va)
    Xv = X_va.copy()
    cat = [c for c in CAT_COLS if c in Xv.columns]
    if enc is not None:
        Xv[cat] = enc.transform(Xv[cat].astype(str))
    val_probs = model.predict_proba(Xv)[:, 1]
    val_auc = roc_auc_score(y_va, val_probs)
    val_loss = log_loss(y_va, val_probs)
    print(f"\n  Final val AUC:     {val_auc:.4f}")
    print(f"  Final val logloss: {val_loss:.4f}")

    # === Threshold search ===
    print(f"\n=== Threshold search (target WR >= {args.target_wr*100:.0f}%) ===")
    thr_df = _threshold_table(y_va, val_probs)
    qualified = thr_df[thr_df["win_rate"] >= args.target_wr]
    if len(qualified) > 0:
        chosen = qualified.sort_values("threshold").iloc[0]
        chosen_thr = float(chosen["threshold"])
        reason = f"lowest threshold with WR >= {args.target_wr*100:.0f}%"
    else:
        chosen = thr_df.sort_values("expectancy_R", ascending=False).iloc[0]
        chosen_thr = float(chosen["threshold"])
        reason = f"no threshold met {args.target_wr*100:.0f}%; max-expectancy"
        print(f"  WARNING: {reason}")
    print(f"  Chosen threshold: {chosen_thr:.2f}  ({reason})")
    print(f"\n  Threshold summary:")
    print(thr_df.to_string(index=False))

    # === Feature importance via permutation ===
    print(f"\n=== Top 15 features (permutation importance) ===")
    try:
        perm = permutation_importance(model, Xv, y_va, n_repeats=10,
                                       random_state=42, n_jobs=-1)
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
        print(fi.head(15).to_string(index=False))
    except Exception as e:
        print(f"  (importance failed: {e})")

    # === Compare with existing model ===
    print(f"\n=== Comparison vs existing model ===")
    existing_auc = None
    if META_PATH.exists():
        try:
            existing = json.load(open(META_PATH))
            existing_auc = existing.get("val_auc")
            print(f"  Existing model val_auc: {existing_auc}")
        except Exception as e:
            print(f"  (could not read existing meta: {e})")
    print(f"  New model val_auc: {val_auc:.4f}")

    should_replace = (not args.no_replace) and (
        existing_auc is None or val_auc > existing_auc
    )

    # === Wrapper that the bot uses (must match _meta_scorer interface) ===
    class _Wrapper:
        def __init__(self, model, encoder, feature_names):
            self.model = model
            self.encoder = encoder
            self.feature_names = feature_names
        def _enc(self, X):
            if self.encoder is None: return X
            X = X.copy()
            cat = [c for c in CAT_COLS if c in X.columns]
            X[cat] = self.encoder.transform(X[cat].astype(str))
            return X
        def predict(self, X):
            return self.model.predict_proba(self._enc(X))[:, 1]

    wrapper = _Wrapper(model, enc, list(X.columns))

    if should_replace:
        # Backup old
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, BACKUP_MODEL)
            print(f"  backed up old model -> {BACKUP_MODEL.name}")
        if META_PATH.exists():
            shutil.copy2(META_PATH, BACKUP_META)
        # Save new
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(wrapper, f)
        meta = {
            "trained_at_utc": datetime.utcnow().isoformat() + "Z",
            "framework": "HistGradientBoosting",
            "dataset": str(DATASET.name),
            "n_total": int(len(df)),
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            "features": list(X.columns),
            "categorical_features": [c for c in CAT_COLS if c in X.columns],
            "chosen_threshold": chosen_thr,
            "threshold_reason": reason,
            "val_auc": float(val_auc),
            "val_logloss": float(val_loss),
            "cv_aucs": [float(a) for a in aucs],
            "previous_val_auc": existing_auc,
            "target_win_rate": args.target_wr,
            "tp_sl_ratio_assumed": 1.67,
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\n  REPLACED model: {MODEL_PATH}")
        print(f"  Bot will pick this up on next restart (hot-reload via mtime).")
    else:
        if args.no_replace:
            print(f"\n  --no-replace set; model NOT swapped")
        else:
            print(f"\n  New model AUC {val_auc:.4f} not better than existing {existing_auc:.4f}")
            print(f"  Model NOT swapped — old one stays live")

    return 0


if __name__ == "__main__":
    sys.exit(main())
