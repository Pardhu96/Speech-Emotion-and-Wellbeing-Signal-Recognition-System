# 🎙️ Speech Emotion Recognition (SER) Studio

An end-to-end Machine Learning and Deep Learning system for Speech Emotion Recognition built with Python, Librosa, Scikit-Learn, PyTorch, and Streamlit.

---

## 🌟 Key Features

1. **Acoustic Feature Extraction**:
   - **MFCCs**: 20 coefficients + 1st & 2nd order delta dynamics.
   - **Pitch (F0)**: Fundamental frequency tracking via Yin algorithm.
   - **RMS Energy**: Root-mean-square amplitude tracking (arousal/loudness).
   - **Zero-Crossing Rate (ZCR)**: Percussiveness and noise characteristics.
   - **Spectral Dynamics**: Centroid (brightness), Rolloff, Bandwidth, Contrast, Flatness.
   - **Mel Spectrograms**: 128 mel frequency bins for 2D Deep Convolutional Neural Networks.

2. **Genuine Machine Learning & Deep Learning Models**:
   - **Support Vector Machine (SVM)**: RBF Kernel with standard feature scaling and calibrated probability estimation.
   - **Random Forest Classifier**: Ensemble of 150 decision trees with balanced class weighting.
   - **PyTorch 2D CNN**: 4 Conv-BatchNorm-ReLU-MaxPool blocks with AdaptiveAvgPool and dropout classification head.
   - **Ensemble Mode**: Softmax voting across SVM, RF, and CNN.

3. **Authentic Dataset & Validation**:
   - Uses the official **RAVDESS** benchmark dataset (1440 speech recordings across 24 professional actors, 12 male & 12 female).
   - **No fabricated metrics**: All numbers, confusion matrices, and probabilities are computed on genuine hold-out test data.

4. **Speaker-Independent Cross-Validation**:
   - Compares standard **Speaker-Dependent** (random 80/20 split) vs **Speaker-Independent** (`GroupKFold` by Actor ID) validation to measure real-world vocal generalization degradation.

5. **Interactive Streamlit Web Interface**:
   - Live file upload (WAV, MP3, FLAC, OGG) or preset authentic sample selector.
   - Real-time audio waveform and energy contour visualizations.
   - Heatmaps for Mel Spectrograms and MFCCs.
   - Side-by-side acoustic feature comparison between different emotional states.
   - Interactive Plotly confusion matrix heatmaps (raw counts and normalized percentages).

---

## 🚀 Quick Start

### 1. Installation
Install dependencies in your Python 3.10+ environment:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation & Training
Run the complete automated pipeline (downloads RAVDESS, extracts features, trains SVM, RF, and CNN, and calculates genuine test metrics):
```bash
python scripts/download_or_prep_data.py
python scripts/train_all.py
```

### 3. Launch Web Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
speech_emotion_recognition/
├── app.py                      # Streamlit interactive web application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── src/
│   ├── __init__.py
│   ├── audio_processing.py     # Resampling, silence trimming, normalization, padding
│   ├── feature_extraction.py   # MFCC, Pitch, RMS, ZCR, Spectral, Mel Spectrogram
│   ├── dataset.py              # PyTorch Dataset, caching, and manifest handling
│   ├── models_tabular.py       # SVM and Random Forest pipelines & inference
│   ├── models_cnn.py           # PyTorch 2D CNN architecture, training & checkpointing
│   ├── evaluator.py            # Accuracy, Precision, Recall, F1, Confusion Matrices
│   └── speaker_eval.py         # GroupKFold speaker-independent evaluation
├── scripts/
│   ├── download_or_prep_data.py# Dataset downloader and manifest generator
│   └── train_all.py            # Complete end-to-end model training script
├── data/
│   ├── audio/                  # Extracted RAVDESS WAV files
│   ├── samples/                # Curated audio samples for quick app testing
│   └── dataset_manifest.csv    # Parsed dataset metadata
└── models/
    ├── svm_model.joblib        # Trained SVM pipeline
    ├── rf_model.joblib         # Trained Random Forest pipeline
    ├── cnn_model.pt            # Trained PyTorch CNN checkpoint
    ├── metrics.json            # Genuine test set evaluation metrics
    └── speaker_eval_results.json # Speaker-independent validation results
```
