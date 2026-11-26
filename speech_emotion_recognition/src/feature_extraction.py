"""
Feature Extraction Module for Speech Emotion Recognition.
Extracts MFCCs, Pitch (F0), RMS Energy, Zero Crossing Rate (ZCR),
Spectral Features, and 2D Log Mel Spectrograms.
"""

import numpy as np
import librosa
from typing import Dict, Tuple, Any, List


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float = 8000.0
) -> np.ndarray:
    """
    Extract Log Mel Spectrogram (in dB) for 2D CNN classification.
    Returns: 2D numpy array of shape (n_mels, time_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def extract_pitch(
    y: np.ndarray,
    sr: int = 16000,
    fmin: float = 65.0,
    fmax: float = 400.0,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extract fundamental frequency (pitch / F0) contour and statistics using Yin / pyin.
    Returns: (pitch_contour, times, stats_dict)
    """
    try:
        # Using librosa.yin for reliable and fast F0 tracking
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        # Filter unvoiced or outlier values
        voiced = f0[(f0 >= fmin) & (f0 <= fmax) & (~np.isnan(f0))]
    except Exception:
        f0 = np.zeros(len(y) // hop_length + 1)
        times = np.linspace(0, len(y) / sr, len(f0))
        voiced = np.array([])

    if len(voiced) > 0:
        pitch_mean = float(np.mean(voiced))
        pitch_std = float(np.std(voiced))
        pitch_max = float(np.max(voiced))
        pitch_min = float(np.min(voiced))
        voicing_fraction = float(len(voiced) / len(f0)) if len(f0) > 0 else 0.0
    else:
        pitch_mean, pitch_std, pitch_max, pitch_min, voicing_fraction = 0.0, 0.0, 0.0, 0.0, 0.0

    stats = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_max": pitch_max,
        "pitch_min": pitch_min,
        "voicing_fraction": voicing_fraction
    }
    return f0, times, stats


def extract_rms_energy(
    y: np.ndarray,
    hop_length: int = 512
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract Root Mean Square (RMS) energy curve and statistics.
    Returns: (rms_curve, stats_dict)
    """
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    stats = {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_max": float(np.max(rms)),
        "rms_min": float(np.min(rms))
    }
    return rms, stats


def extract_zcr(
    y: np.ndarray,
    hop_length: int = 512
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract Zero-Crossing Rate (ZCR) curve and statistics.
    Returns: (zcr_curve, stats_dict)
    """
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=hop_length)[0]
    stats = {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "zcr_max": float(np.max(zcr))
    }
    return zcr, stats


def extract_spectral_features(
    y: np.ndarray,
    sr: int = 16000,
    hop_length: int = 512
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Extract Spectral Centroid, Bandwidth, Rolloff, Contrast, and Flatness.
    Returns: (curves_dict, stats_dict)
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

    curves = {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness
    }

    stats = {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness))
    }

    # Contrast across the 7 sub-bands
    for b in range(contrast.shape[0]):
        stats[f"spectral_contrast_band_{b}_mean"] = float(np.mean(contrast[b]))

    return curves, stats


def extract_mfcc(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 20,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extract MFCCs, Delta MFCCs, and Delta-Delta (accelerated) MFCCs.
    Returns: (mfcc_matrix, delta_matrix, delta2_matrix, stats_dict)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    stats = {}
    for i in range(n_mfcc):
        stats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        stats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        stats[f"mfcc_delta_{i+1}_mean"] = float(np.mean(delta_mfcc[i]))
        stats[f"mfcc_delta_{i+1}_std"] = float(np.std(delta_mfcc[i]))
        stats[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(delta2_mfcc[i]))

    return mfcc, delta_mfcc, delta2_mfcc, stats


def extract_all_tabular_features(
    y: np.ndarray,
    sr: int = 16000
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Extract full aggregated 1D feature vector for Classical ML (SVM, Random Forest),
    along with feature names and detailed visualization data.
    """
    # 1. Pitch
    pitch_contour, pitch_times, pitch_stats = extract_pitch(y, sr)

    # 2. RMS Energy
    rms_curve, rms_stats = extract_rms_energy(y)

    # 3. ZCR
    zcr_curve, zcr_stats = extract_zcr(y)

    # 4. Spectral Features
    spectral_curves, spectral_stats = extract_spectral_features(y, sr)

    # 5. MFCCs (20 coefficients + deltas)
    mfcc_matrix, delta_mfcc, delta2_mfcc, mfcc_stats = extract_mfcc(y, sr, n_mfcc=20)

    # Combine all stats into a flat dictionary
    feature_dict = {}
    feature_dict.update(mfcc_stats)
    feature_dict.update(pitch_stats)
    feature_dict.update(rms_stats)
    feature_dict.update(zcr_stats)
    feature_dict.update(spectral_stats)

    # Alphabetically or deterministically sort feature keys
    feature_names = sorted(feature_dict.keys())
    feature_vector = np.array([feature_dict[k] for k in feature_names], dtype=np.float32)

    # Visualizations payload for Streamlit UI
    vis_data = {
        "pitch_contour": pitch_contour,
        "pitch_times": pitch_times,
        "rms_curve": rms_curve,
        "zcr_curve": zcr_curve,
        "spectral_curves": spectral_curves,
        "mfcc_matrix": mfcc_matrix,
        "delta_mfcc": delta_mfcc,
        "delta2_mfcc": delta2_mfcc,
        "feature_dict": feature_dict
    }

    return feature_vector, feature_names, vis_data
