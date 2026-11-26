"""
Tabular Machine Learning Models for Speech Emotion Recognition.
Implements Support Vector Machine (SVM) and Random Forest classifiers.
"""

import os
import joblib
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def build_svm_pipeline(
    C: float = 3.0,
    kernel: str = 'rbf',
    gamma: str = 'scale'
) -> Pipeline:
    """
    Build standard-scaled Support Vector Classifier pipeline with calibrated probabilities.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ])


def build_rf_pipeline(
    n_estimators: int = 150,
    max_depth: int = 16,
    min_samples_split: int = 4
) -> Pipeline:
    """
    Build standard-scaled Random Forest Classifier pipeline.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 3.0,
    kernel: str = 'rbf'
) -> Pipeline:
    """
    Train an SVM classifier on extracted acoustic tabular features.
    """
    pipeline = build_svm_pipeline(C=C, kernel=kernel)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 16
) -> Pipeline:
    """
    Train a Random Forest classifier on extracted acoustic tabular features.
    """
    pipeline = build_rf_pipeline(n_estimators=n_estimators, max_depth=max_depth)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline(pipeline: Pipeline, filepath: str):
    """Save trained scikit-learn pipeline."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Model successfully saved to {filepath}")


def load_pipeline(filepath: str) -> Pipeline:
    """Load saved scikit-learn pipeline."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def predict_single_sample(
    pipeline: Pipeline,
    feature_vector: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Generate emotion prediction and probability distribution for a single audio sample.
    """
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    probabilities = pipeline.predict_proba(feature_vector)[0]
    classes = pipeline.classes_

    # Map classes to probabilities
    prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    # Sort by probability descending
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    top_emotion = list(sorted_probs.keys())[0]
    top_confidence = sorted_probs[top_emotion]

    return {
        "predicted_emotion": top_emotion,
        "confidence": top_confidence,
        "probabilities": sorted_probs
    }
