#!/usr/bin/env python3
"""
Evaluate whether CVE-derived continuous features improve ML classification.

Loads the baseline hybrid features and the CVE structural features,
appends the CVE features to the hybrid structural vector, retrains
logistic regression, and compares test AUC/F1 vs baseline.

Proceed to add ML features only if delta AUC >= 0.001.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CVE-derived features for ML value."
    )
    parser.add_argument(
        "--hybrid-npz",
        default="outputs/hybrid_features.npz",
        help="Path to baseline hybrid features NPZ",
    )
    parser.add_argument(
        "--cve-npz",
        default="outputs/cve_structural_features.npz",
        help="Path to CVE structural features NPZ",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for evaluation report",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load baseline hybrid features
    hybrid = np.load(args.hybrid_npz, allow_pickle=True)
    X_struct_baseline = hybrid["X_struct"]
    y = hybrid["y"]
    paths_hybrid = list(hybrid["paths"])
    struct_names_baseline = list(hybrid["struct_feature_names"])

    # Load CVE features
    cve = np.load(args.cve_npz, allow_pickle=True)
    X_cve = cve["X_cve"]
    paths_cve = list(cve["paths"])
    cve_feature_names = list(cve["feature_names"])

    # Align by path
    path_to_idx_cve = {p: i for i, p in enumerate(paths_cve)}
    aligned_indices = []
    for p in paths_hybrid:
        if p in path_to_idx_cve:
            aligned_indices.append(path_to_idx_cve[p])
        else:
            aligned_indices.append(None)

    # Filter to files present in both
    valid_mask = [idx is not None for idx in aligned_indices]
    valid_indices_hybrid = [i for i, v in enumerate(valid_mask) if v]
    valid_indices_cve = [aligned_indices[i] for i in valid_indices_hybrid]

    if len(valid_indices_hybrid) < 100:
        print(f"ERROR: Only {len(valid_indices_hybrid)} files matched between datasets.", file=sys.stderr)
        return 1

    X_base = X_struct_baseline[valid_indices_hybrid].astype(np.float64)
    X_new = X_cve[valid_indices_cve].astype(np.float64)
    y_aligned = y[valid_indices_hybrid]

    # Select only the 4 candidate features for augmentation
    candidate_names = [
        "log_max_declared_opcode_count",
        "spp_max",
        "compression_variety",
        "tile_count_ratio",
    ]
    candidate_indices = [cve_feature_names.index(n) for n in candidate_names]
    X_augment = X_new[:, candidate_indices]

    # Combined features
    X_combined = np.hstack([X_base, X_augment])

    print(f"Aligned samples: {len(y_aligned)}")
    print(f"Baseline features: {X_base.shape[1]}")
    print(f"Augmented features: {X_combined.shape[1]} (+{X_augment.shape[1]})")
    print(f"Class distribution: benign={int((y_aligned == 0).sum())}, malicious={int((y_aligned == 1).sum())}")
    print()

    # Cross-validated evaluation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    baseline_aucs = []
    baseline_f1s = []
    combined_aucs = []
    combined_f1s = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_base, y_aligned)):
        # Baseline
        scaler_b = StandardScaler()
        X_train_b = scaler_b.fit_transform(X_base[train_idx])
        X_test_b = scaler_b.transform(X_base[test_idx])
        y_train = y_aligned[train_idx]
        y_test = y_aligned[test_idx]

        clf_b = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        clf_b.fit(X_train_b, y_train)
        prob_b = clf_b.predict_proba(X_test_b)[:, 1]
        pred_b = clf_b.predict(X_test_b)

        auc_b = roc_auc_score(y_test, prob_b)
        f1_b = f1_score(y_test, pred_b)
        baseline_aucs.append(auc_b)
        baseline_f1s.append(f1_b)

        # Combined
        scaler_c = StandardScaler()
        X_train_c = scaler_c.fit_transform(X_combined[train_idx])
        X_test_c = scaler_c.transform(X_combined[test_idx])

        clf_c = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        clf_c.fit(X_train_c, y_train)
        prob_c = clf_c.predict_proba(X_test_c)[:, 1]
        pred_c = clf_c.predict(X_test_c)

        auc_c = roc_auc_score(y_test, prob_c)
        f1_c = f1_score(y_test, pred_c)
        combined_aucs.append(auc_c)
        combined_f1s.append(f1_c)

        print(f"  Fold {fold+1}: baseline AUC={auc_b:.6f} F1={f1_b:.6f} | "
              f"combined AUC={auc_c:.6f} F1={f1_c:.6f}")

    mean_auc_b = np.mean(baseline_aucs)
    mean_f1_b = np.mean(baseline_f1s)
    mean_auc_c = np.mean(combined_aucs)
    mean_f1_c = np.mean(combined_f1s)
    delta_auc = mean_auc_c - mean_auc_b
    delta_f1 = mean_f1_c - mean_f1_b
    recommend = delta_auc >= 0.001

    print()
    print(f"Mean baseline AUC: {mean_auc_b:.6f}  F1: {mean_f1_b:.6f}")
    print(f"Mean combined AUC: {mean_auc_c:.6f}  F1: {mean_f1_c:.6f}")
    print(f"Delta AUC: {delta_auc:+.6f}  Delta F1: {delta_f1:+.6f}")
    print(f"Recommendation: {'ADD features (delta AUC >= 0.001)' if recommend else 'SKIP features (delta AUC < 0.001)'}")

    report = {
        "n_samples": int(len(y_aligned)),
        "n_folds": args.n_folds,
        "baseline_features": int(X_base.shape[1]),
        "combined_features": int(X_combined.shape[1]),
        "candidate_feature_names": candidate_names,
        "baseline_mean_auc": float(mean_auc_b),
        "baseline_mean_f1": float(mean_f1_b),
        "combined_mean_auc": float(mean_auc_c),
        "combined_mean_f1": float(mean_f1_c),
        "delta_auc": float(delta_auc),
        "delta_f1": float(delta_f1),
        "recommend_add_features": recommend,
        "per_fold_baseline_auc": [float(x) for x in baseline_aucs],
        "per_fold_combined_auc": [float(x) for x in combined_aucs],
    }

    report_path = os.path.join(args.output_dir, "cve_feature_ml_eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
