"""
Comprehensive End-to-End Training Pipeline for Speech Emotion Recognition.
Trains SVM, Random Forest, and PyTorch Mel-Spectrogram CNN on real data.
Computes genuine test metrics, confusion matrices, and speaker-independent evaluation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set up project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset import (
    extract_or_load_tabular_features,
    extract_or_load_spectrograms,
    MelSpectrogramDataset
)
from src.models_tabular import (
    train_svm,
    train_random_forest,
    save_pipeline
)
from src.models_cnn import (
    EmotionCNN,
    train_cnn,
    save_cnn
)
from src.evaluator import (
    evaluate_predictions,
    save_metrics,
    build_comparison_dataframe
)
from src.speaker_eval import (
    evaluate_speaker_dependence,
    save_speaker_eval_results
)


def run_training_pipeline(force_recompute: bool = False, cnn_epochs: int = 15):
    manifest_path = BASE_DIR / "data" / "dataset_manifest.csv"
    if not manifest_path.exists():
        print("Manifest not found. Running dataset download & preparation first...")
        from scripts.download_or_prep_data import download_dataset, build_manifest
        download_dataset()
        df = build_manifest()
    else:
        df = pd.read_csv(manifest_path)

    print(f"Loaded dataset manifest with {len(df)} samples.")
    print(f"Emotions present: {df['emotion'].value_counts().to_dict()}")

    # 1. Tabular features extraction
    tab_cache = str(BASE_DIR / "data" / "features_tabular.npz")
    X_tab, y_tab, actor_ids, feature_names, emotion_classes = extract_or_load_tabular_features(
        df, cache_path=tab_cache, force_recompute=force_recompute
    )

    # 2. Spectrogram extraction for CNN
    spec_cache = str(BASE_DIR / "data" / "spectrograms.npz")
    X_spec, y_spec, _, _ = extract_or_load_spectrograms(
        df, cache_path=spec_cache, force_recompute=force_recompute
    )

    # Consistent train/test split across all models (80% train, 20% test)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y_tab
    )

    X_train_tab, X_test_tab = X_tab[train_idx], X_tab[test_idx]
    y_train, y_test = y_tab[train_idx], y_tab[test_idx]
    X_train_spec, X_test_spec = X_spec[train_idx], X_spec[test_idx]

    print(f"Training set: {len(train_idx)} samples | Test set: {len(test_idx)} samples")

    # -----------------------------
    # Train Tabular Models (SVM & RF)
    # -----------------------------
    print("\n" + "="*50)
    print("Training Support Vector Machine (SVM)...")
    svm_pipeline = train_svm(X_train_tab, y_train, C=3.0, kernel='rbf')
    svm_path = str(BASE_DIR / "models" / "svm_model.joblib")
    save_pipeline(svm_pipeline, svm_path)

    print("\n" + "="*50)
    print("Training Random Forest Classifier...")
    rf_pipeline = train_random_forest(X_train_tab, y_train, n_estimators=150, max_depth=16)
    rf_path = str(BASE_DIR / "models" / "rf_model.joblib")
    save_pipeline(rf_pipeline, rf_path)

    # -----------------------------
    # Train Deep Learning Model (CNN)
    # -----------------------------
    print("\n" + "="*50)
    print("Training PyTorch Mel Spectrogram CNN...")
    class_to_idx = {cls: idx for idx, cls in enumerate(emotion_classes)}
    y_train_num = np.array([class_to_idx[e] for e in y_train])
    y_test_num = np.array([class_to_idx[e] for e in y_test])

    train_ds = MelSpectrogramDataset(X_train_spec, y_train_num)
    test_ds = MelSpectrogramDataset(X_test_spec, y_test_num)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    cnn_model = EmotionCNN(num_classes=len(emotion_classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for CNN training: {device}")

    cnn_model, history = train_cnn(
        cnn_model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=cnn_epochs,
        lr=1e-3,
        device=device
    )

    cnn_path = str(BASE_DIR / "models" / "cnn_model.pt")
    save_cnn(cnn_model, cnn_path, emotion_classes)

    # -----------------------------
    # Compute Real Test Metrics
    # -----------------------------
    print("\n" + "="*50)
    print("Computing Genuine Evaluation Metrics on Hold-Out Test Set...")

    # SVM Predictions
    svm_preds = svm_pipeline.predict(X_test_tab)
    svm_metrics = evaluate_predictions(y_test, svm_preds, emotion_classes)

    # Random Forest Predictions
    rf_preds = rf_pipeline.predict(X_test_tab)
    rf_metrics = evaluate_predictions(y_test, rf_preds, emotion_classes)

    # CNN Predictions
    cnn_model.eval()
    cnn_preds_num = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            logits = cnn_model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            cnn_preds_num.extend(preds)

    cnn_preds = np.array([emotion_classes[i] for i in cnn_preds_num])
    cnn_metrics = evaluate_predictions(y_test, cnn_preds, emotion_classes)

    all_metrics = {
        "SVM": svm_metrics,
        "Random Forest": rf_metrics,
        "CNN": cnn_metrics
    }
    metrics_path = str(BASE_DIR / "models" / "metrics.json")
    save_metrics(all_metrics, metrics_path)

    # Print Comparison Table
    print("\n--- MODEL PERFORMANCE COMPARISON (GENUINE TEST SET) ---")
    comp_df = build_comparison_dataframe(all_metrics)
    print(comp_df.to_string(index=False))

    # -----------------------------
    # Speaker-Independent Evaluation
    # -----------------------------
    print("\n" + "="*50)
    print("Executing Speaker-Independent vs Speaker-Dependent Evaluation...")
    speaker_results = evaluate_speaker_dependence(
        X_tab, y_tab, actor_ids, emotion_classes, n_splits=5
    )
    speaker_path = str(BASE_DIR / "models" / "speaker_eval_results.json")
    save_speaker_eval_results(speaker_results, speaker_path)

    print("\n--- SPEAKER EVALUATION RESULTS ---")
    print(f"Speaker-Dependent SVM Accuracy:   {speaker_results['speaker_dependent']['svm_accuracy']*100:.2f}%")
    print(f"Speaker-Independent SVM Accuracy: {speaker_results['speaker_independent']['svm_accuracy_mean']*100:.2f}% "
          f"(±{speaker_results['speaker_independent']['svm_accuracy_std']*100:.2f}%)")
    print(f"Generalization Drop:              {speaker_results['generalization_gap']['svm_gap_percentage']:.2f}%")

    print("\nEnd-to-end training pipeline successfully finished!")


if __name__ == "__main__":
    run_training_pipeline()
