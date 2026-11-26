"""
Evaluation Module for Speech Emotion Recognition.
Computes genuine test metrics, confusion matrices, and comparison tables.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics on actual test data.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')

    clf_report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4),
        "precision_weighted": round(p_weighted, 4),
        "recall_macro": round(r_macro, 4),
        "recall_weighted": round(r_weighted, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": [[round(float(v), 4) for v in row] for row in cm_norm],
        "class_names": class_names,
        "classification_report": clf_report
    }


def save_metrics(metrics: Dict[str, Any], filepath: str = "models/metrics.json"):
    """Save evaluation metrics to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str = "models/metrics.json") -> Dict[str, Any]:
    """Load evaluation metrics from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_comparison_dataframe(all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Format side-by-side comparison dataframe across multiple models.
    """
    rows = []
    for model_name, m in all_metrics.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Macro F1": f"{m['f1_macro'] * 100:.2f}%",
            "Weighted F1": f"{m['f1_weighted'] * 100:.2f}%",
            "Precision (Macro)": f"{m['precision_macro'] * 100:.2f}%",
            "Recall (Macro)": f"{m['recall_macro'] * 100:.2f}%"
        })
    return pd.DataFrame(rows)
