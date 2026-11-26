"""
Audio Preprocessing Module for Speech Emotion Recognition.
Handles audio loading, format conversion, trimming, padding/truncation, and normalization.
"""

import io
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Union, BinaryIO


def load_audio(
    audio_source: Union[str, BinaryIO, bytes, io.BytesIO],
    target_sr: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file from file path, bytes, or file-like object.
    Resamples to target_sr and converts to mono.
    """
    if isinstance(audio_source, (bytes, bytearray)):
        audio_source = io.BytesIO(audio_source)

    try:
        # Load using librosa (uses soundfile under the hood)
        y, sr = librosa.load(audio_source, sr=target_sr, mono=mono)
    except Exception as e:
        # Fallback to soundfile directly if librosa fails on some buffer types
        try:
            if hasattr(audio_source, 'seek'):
                audio_source.seek(0)
            data, sr = sf.read(audio_source)
            if data.ndim > 1 and mono:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            y = data.astype(np.float32)
        except Exception as e2:
            raise ValueError(f"Failed to load audio: {str(e)} | Fallback error: {str(e2)}")

    return y.astype(np.float32), sr


def trim_silence(
    y: np.ndarray,
    top_db: float = 25.0
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim leading and trailing silence from audio signal.
    Returns trimmed signal and the (start_index, end_index) tuple.
    """
    if len(y) == 0:
        return y, (0, 0)
    trimmed_y, index = librosa.effects.trim(y, top_db=top_db)
    # Guard against over-trimming completely silent audio
    if len(trimmed_y) < 100:
        return y, (0, len(y))
    return trimmed_y, index


def normalize_audio(
    y: np.ndarray,
    target_peak: float = 0.95
) -> np.ndarray:
    """
    Peak-normalize the audio signal to prevent clipping and ensure consistent amplitude.
    """
    if len(y) == 0:
        return y
    max_amp = np.max(np.abs(y))
    if max_amp > 1e-6:
        return (y / max_amp) * target_peak
    return y


def pad_or_truncate(
    y: np.ndarray,
    target_length: int = 48000
) -> np.ndarray:
    """
    Ensure the audio signal is exactly target_length samples.
    - If shorter: symmetric or right-side zero padding.
    - If longer: center-crop or take the first target_length samples.
    """
    curr_len = len(y)
    if curr_len == target_length:
        return y
    elif curr_len < target_length:
        # Zero-pad symmetrically
        pad_total = target_length - curr_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(y, (pad_left, pad_right), mode='constant')
    else:
        # Center crop
        start = (curr_len - target_length) // 2
        return y[start : start + target_length]


def preprocess_audio(
    audio_source: Union[str, BinaryIO, bytes, io.BytesIO],
    target_sr: int = 16000,
    duration_sec: float = 3.0,
    trim: bool = True
) -> Tuple[np.ndarray, int, dict]:
    """
    Full audio preprocessing pipeline:
    1. Load audio & resample to target_sr
    2. Optional silence trimming
    3. Amplitude normalization
    4. Pad or truncate to fixed duration
    Returns: (preprocessed_signal, sample_rate, metadata_dict)
    """
    raw_y, sr = load_audio(audio_source, target_sr=target_sr, mono=True)
    raw_duration = len(raw_y) / sr if sr > 0 else 0

    if trim:
        y, (start_idx, end_idx) = trim_silence(raw_y, top_db=25.0)
    else:
        y = raw_y
        start_idx, end_idx = 0, len(raw_y)

    norm_y = normalize_audio(y)
    target_samples = int(duration_sec * target_sr)
    final_y = pad_or_truncate(norm_y, target_length=target_samples)

    metadata = {
        "original_duration_sec": round(raw_duration, 3),
        "processed_duration_sec": round(len(final_y) / sr, 3),
        "sample_rate": sr,
        "trimmed_start_sec": round(start_idx / sr, 3),
        "trimmed_end_sec": round(end_idx / sr, 3),
        "rms_energy": float(np.sqrt(np.mean(final_y**2))),
        "peak_amplitude": float(np.max(np.abs(final_y)))
    }

    return final_y, sr, metadata
