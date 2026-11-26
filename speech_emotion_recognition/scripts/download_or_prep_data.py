"""
Download and prepare authentic RAVDESS Speech Emotion Recognition dataset.
Source: Official Zenodo archive (https://zenodo.org/records/1188976)
"""

import os
import sys
import zipfile
import shutil
import urllib.request
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
SAMPLES_DIR = DATA_DIR / "samples"
ZIP_PATH = DATA_DIR / "Audio_Speech_Actors_01-24.zip"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.csv"

ZENODO_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"

# RAVDESS emotion code mappings
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}

STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}


def download_dataset():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if files already extracted
    wav_files = list(AUDIO_DIR.rglob("*.wav"))
    if len(wav_files) >= 1000:
        print(f"Dataset already extracted: found {len(wav_files)} WAV files in {AUDIO_DIR}")
        return

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 200000000:
        print(f"Downloading RAVDESS dataset from Zenodo ({ZENODO_URL})...")
        req = urllib.request.Request(
            ZENODO_URL,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, 'wb') as out_file:
            total_size = int(resp.headers.get('content-length', 0))
            downloaded = 0
            block_size = 1024 * 512
            while True:
                buf = resp.read(block_size)
                if not buf:
                    break
                out_file.write(buf)
                downloaded += len(buf)
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    sys.stdout.write(f"\rDownloading: {percent:5.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                    sys.stdout.flush()
        print("\nDownload complete!")

    print(f"Extracting {ZIP_PATH} to {AUDIO_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(AUDIO_DIR)
    print("Extraction complete!")


def build_manifest():
    print("Building dataset manifest...")
    records = []
    for root, _, files in os.walk(AUDIO_DIR):
        for f in sorted(files):
            if f.endswith(".wav"):
                parts = f.replace(".wav", "").split("-")
                if len(parts) == 7:
                    modality, vocal_chan, emotion_code, intensity_code, stmt_code, rep_code, actor_code = parts
                    actor_int = int(actor_code)
                    gender = "female" if actor_int % 2 == 0 else "male"
                    emotion = EMOTION_MAP.get(emotion_code, "unknown")
                    
                    full_path = os.path.join(root, f)
                    records.append({
                        "filename": f,
                        "filepath": full_path,
                        "rel_path": os.path.relpath(full_path, BASE_DIR),
                        "modality": modality,
                        "vocal_channel": vocal_chan,
                        "emotion_code": emotion_code,
                        "emotion": emotion,
                        "intensity": INTENSITY_MAP.get(intensity_code, "unknown"),
                        "statement": STATEMENT_MAP.get(stmt_code, "unknown"),
                        "repetition": rep_code,
                        "actor_id": actor_int,
                        "gender": gender
                    })

    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest written to {MANIFEST_PATH}: {len(df)} samples across {df['actor_id'].nunique()} actors.")
    
    # Create representative samples for quick testing in UI
    print("Selecting sample audio clips for live app demonstrations...")
    for emotion in df["emotion"].unique():
        male_sample = df[(df["emotion"] == emotion) & (df["actor_id"] == 1)].head(1)
        if not male_sample.empty:
            src = male_sample.iloc[0]["filepath"]
            dst = SAMPLES_DIR / f"{emotion}_male_actor01.wav"
            shutil.copy2(src, dst)
            
        female_sample = df[(df["emotion"] == emotion) & (df["actor_id"] == 2)].head(1)
        if not female_sample.empty:
            src = female_sample.iloc[0]["filepath"]
            dst = SAMPLES_DIR / f"{emotion}_female_actor02.wav"
            shutil.copy2(src, dst)
            
    print(f"Sample audio clips copied to {SAMPLES_DIR}")
    return df


if __name__ == "__main__":
    download_dataset()
    build_manifest()
