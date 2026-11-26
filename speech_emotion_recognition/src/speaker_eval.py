"""
Speaker-Independent Evaluation Module for Speech Emotion Recognition.
Compares standard random train/test splits (speaker-dependent) against
GroupKFold / Leave-Speakers-Out splits (speaker-independent).
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, f1_score

from src.models_tabular import build_svm_pipeline, build_rf_pipeline


def evaluate_speaker_dependence(
    X: np.ndarray,
    y: np.ndarray,
    actor_ids: np.ndarray,
    class_names: List[str],
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Perform rigorous comparison between:
    1. Speaker-Dependent (Random Split): Same speakers present in both train and test.
    2. Speaker-Independent (GroupKFold): Held-out speakers strictly unseen in training.
    """
    print("Running Speaker-Dependent vs Speaker-Independent Evaluation...")

    # --- 1. Speaker-Dependent (Standard Stratified Random Split 80/20) ---
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    svm_rand = build_svm_pipeline()
    svm_rand.fit(X_train_rand, y_train_rand)
    svm_rand_pred = svm_rand.predict(X_test_rand)
    svm_rand_acc = float(accuracy_score(y_test_rand, svm_rand_pred))
    svm_rand_f1 = float(f1_score(y_test_rand, svm_rand_pred, average='macro', zero_division=0))

    rf_rand = build_rf_pipeline()
    rf_rand.fit(X_train_rand, y_train_rand)
    rf_rand_pred = rf_rand.predict(X_test_rand)
    rf_rand_acc = float(accuracy_score(y_test_rand, rf_rand_pred))
    rf_rand_f1 = float(f1_score(y_test_rand, rf_rand_pred, average='macro', zero_division=0))

    # --- 2. Speaker-Independent (GroupKFold by Actor ID) ---
    gkf = GroupKFold(n_splits=n_splits)

    svm_gkf_accs = []
    svm_gkf_f1s = []
    rf_gkf_accs = []
    rf_gkf_f1s = []
    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=actor_ids)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        test_actors = sorted(list(set(actor_ids[test_idx])))

        # Train SVM
        svm_model = build_svm_pipeline()
        svm_model.fit(X_tr, y_tr)
        svm_p = svm_model.predict(X_te)
        fold_svm_acc = float(accuracy_score(y_te, svm_p))
        fold_svm_f1 = float(f1_score(y_te, svm_p, average='macro', zero_division=0))
        svm_gkf_accs.append(fold_svm_acc)
        svm_gkf_f1s.append(fold_svm_f1)

        # Train RF
        rf_model = build_rf_pipeline()
        rf_model.fit(X_tr, y_tr)
        rf_p = rf_model.predict(X_te)
        fold_rf_acc = float(accuracy_score(y_te, rf_p))
        fold_rf_f1 = float(f1_score(y_te, rf_p, average='macro', zero_division=0))
        rf_gkf_accs.append(fold_rf_acc)
        rf_gkf_f1s.append(fold_rf_f1)

        fold_details.append({
            "fold": fold_idx + 1,
            "test_actors": [int(a) for a in test_actors],
            "num_test_samples": len(test_idx),
            "svm_accuracy": round(fold_svm_acc, 4),
            "svm_f1": round(fold_svm_f1, 4),
            "rf_accuracy": round(fold_rf_acc, 4),
            "rf_f1": round(fold_rf_f1, 4)
        })

    svm_indep_acc = float(np.mean(svm_gkf_accs))
    svm_indep_f1 = float(np.mean(svm_gkf_f1s))
    rf_indep_acc = float(np.mean(rf_gkf_accs))
    rf_indep_f1 = float(np.mean(rf_gkf_f1s))

    results = {
        "speaker_dependent": {
            "description": "Standard Random Split (80/20) where speaker voices leak across train and test sets",
            "svm_accuracy": round(svm_rand_acc, 4),
            "svm_f1": round(svm_rand_f1, 4),
            "rf_accuracy": round(rf_rand_acc, 4),
            "rf_f1": round(rf_rand_f1, 4)
        },
        "speaker_independent": {
            "description": f"{n_splits}-Fold GroupKFold strictly partitioning by Speaker / Actor ID (unseen speakers)",
            "svm_accuracy_mean": round(svm_indep_acc, 4),
            "svm_accuracy_std": round(float(np.std(svm_gkf_accs)), 4),
            "svm_f1_mean": round(svm_indep_f1, 4),
            "rf_accuracy_mean": round(rf_indep_acc, 4),
            "rf_accuracy_std": round(float(np.std(rf_gkf_accs)), 4),
            "rf_f1_mean": round(rf_indep_f1, 4),
            "fold_details": fold_details
        },
        "generalization_gap": {
            "svm_gap_percentage": round((svm_rand_acc - svm_indep_acc) * 100, 2),
            "rf_gap_percentage": round((rf_rand_acc - rf_indep_acc) * 100, 2),
            "analysis": "A significant drop in accuracy on unseen speakers is standard in Speech Emotion Recognition due to speaker acoustic idiosyncrasies (timbre, fundamental frequency baseline)."
        }
    }

    return results


def save_speaker_eval_results(results: Dict[str, Any], filepath: str = "models/speaker_eval_results.json"):
    """Save speaker evaluation results."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Speaker-independent evaluation results saved to {filepath}")


def load_speaker_eval_results(filepath: str = "models/speaker_eval_results.json") -> Dict[str, Any]:
    """Load speaker evaluation results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
