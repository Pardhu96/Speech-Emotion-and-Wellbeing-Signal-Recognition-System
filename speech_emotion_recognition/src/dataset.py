"""
Dataset and Cache Management for Speech Emotion Recognition.
Provides parallel feature extraction, caching, and PyTorch Dataset wrappers.
"""

import os
import concurrent.futures
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

from src.audio_processing import preprocess_audio
from src.feature_extraction import extract_all_tabular_features, extract_mel_spectrogram


class MelSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for Log Mel Spectrograms (2D CNN).
    Returns: (spectrogram_tensor, label_idx)
    Tensor shape: (1, n_mels, time_steps) e.g., (1, 128, 94)
    """
    def __init__(
        self,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        actor_ids: Optional[np.ndarray] = None
    ):
        self.spectrograms = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.actor_ids = actor_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = (self.spectrograms[idx], self.labels[idx])
        if self.actor_ids is not None:
            return item + (self.actor_ids[idx],)
        return item


def _process_single_audio_file(row_data: Tuple[str, str, int]) -> Optional[Tuple[np.ndarray, np.ndarray, str, int]]:
    """Worker function for concurrent extraction."""
    filepath, emotion, actor_id = row_data
    try:
        y, sr, _ = preprocess_audio(filepath, target_sr=16000, duration_sec=3.0)
        feat_vec, _, _ = extract_all_tabular_features(y, sr)
        mel = extract_mel_spectrogram(y, sr=sr, n_mels=128)
        return feat_vec, mel, emotion, actor_id
    except Exception as e:
        print(f"Warning: Failed to process {filepath}: {e}")
        return None


def extract_and_cache_all_features(
    manifest_df: pd.DataFrame,
    tab_cache_path: str = "data/features_tabular.npz",
    spec_cache_path: str = "data/spectrograms.npz",
    force_recompute: bool = False,
    max_workers: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract both tabular features and spectrograms in a single parallel pass across 1440 audio files.
    Returns: (X_tab, X_spec, y_labels, actor_ids, feature_names, emotion_classes)
    """
    if os.path.exists(tab_cache_path) and os.path.exists(spec_cache_path) and not force_recompute:
        print(f"Loading cached features from {tab_cache_path} and {spec_cache_path}...")
        tab_data = np.load(tab_cache_path, allow_pickle=True)
        spec_data = np.load(spec_cache_path, allow_pickle=True)
        return (
            tab_data["X"],
            spec_data["X"],
            tab_data["y"],
            tab_data["actor_ids"],
            list(tab_data["feature_names"]),
            list(tab_data["emotion_classes"])
        )

    print(f"Extracting features for {len(manifest_df)} audio files using {max_workers} parallel workers...")
    
    # Get feature names from a dummy 1-second sample
    dummy_y = np.zeros(16000, dtype=np.float32)
    _, feature_names, _ = extract_all_tabular_features(dummy_y, 16000)

    rows = [(row["filepath"], row["emotion"], int(row["actor_id"])) for _, row in manifest_df.iterrows()]
    
    X_tab_list = []
    X_spec_list = []
    y_list = []
    actor_id_list = []

    # Use ThreadPoolExecutor to avoid Windows spawn issues with soundfile/numba
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_single_audio_file, rows))

    for res in results:
        if res is not None:
            feat_vec, mel, emotion, actor_id = res
            X_tab_list.append(feat_vec)
            X_spec_list.append(mel)
            y_list.append(emotion)
            actor_id_list.append(actor_id)

    X_tab = np.array(X_tab_list, dtype=np.float32)
    X_spec = np.array(X_spec_list, dtype=np.float32)
    y = np.array(y_list)
    actor_ids = np.array(actor_id_list, dtype=np.int32)
    emotion_classes = sorted(list(set(y)))

    # Save Tabular Cache
    os.makedirs(os.path.dirname(tab_cache_path), exist_ok=True)
    np.savez_compressed(
        tab_cache_path,
        X=X_tab,
        y=y,
        actor_ids=actor_ids,
        feature_names=np.array(feature_names),
        emotion_classes=np.array(emotion_classes)
    )

    # Save Spectrogram Cache
    os.makedirs(os.path.dirname(spec_cache_path), exist_ok=True)
    np.savez_compressed(
        spec_cache_path,
        X=X_spec,
        y=y,
        actor_ids=actor_ids,
        emotion_classes=np.array(emotion_classes)
    )

    print(f"Extraction complete! Tabular features: {X_tab.shape}, Spectrograms: {X_spec.shape}")
    return X_tab, X_spec, y, actor_ids, feature_names, emotion_classes


def extract_or_load_tabular_features(
    manifest_df: pd.DataFrame,
    cache_path: str = "data/features_tabular.npz",
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    X_tab, _, y, actor_ids, feature_names, emotion_classes = extract_and_cache_all_features(
        manifest_df, tab_cache_path=cache_path, force_recompute=force_recompute
    )
    return X_tab, y, actor_ids, feature_names, emotion_classes


def extract_or_load_spectrograms(
    manifest_df: pd.DataFrame,
    cache_path: str = "data/spectrograms.npz",
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    _, X_spec, y, actor_ids, _, emotion_classes = extract_and_cache_all_features(
        manifest_df, spec_cache_path=cache_path, force_recompute=force_recompute
    )
    return X_spec, y, actor_ids, emotion_classes
