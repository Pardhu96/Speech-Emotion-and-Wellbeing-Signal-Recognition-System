"""
Speech Emotion Recognition (SER) Studio
Production-grade interactive application built with Streamlit, Librosa, Scikit-Learn, and PyTorch.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.audio_processing import preprocess_audio, load_audio, trim_silence
from src.feature_extraction import (
    extract_all_tabular_features,
    extract_mel_spectrogram
)
from src.models_tabular import (
    load_pipeline,
    predict_single_sample
)
from src.models_cnn import (
    load_cnn,
    predict_single_spectrogram
)
from src.evaluator import load_metrics, build_comparison_dataframe
from src.speaker_eval import load_speaker_eval_results

# Emotion UI metadata mapping
EMOTION_META = {
    "angry": {"emoji": "😠", "color": "#EF4444", "description": "High energy, elevated pitch, high vocal effort"},
    "happy": {"emoji": "😊", "color": "#10B981", "description": "High pitch variability, bright timbre, moderate-high energy"},
    "sad": {"emoji": "😢", "color": "#3B82F6", "description": "Lower pitch, reduced energy, slow speaking rate"},
    "neutral": {"emoji": "😐", "color": "#6B7280", "description": "Baseline pitch, even cadence, steady rhythm"},
    "calm": {"emoji": "😌", "color": "#8B5CF6", "description": "Relaxed vocal folds, soft amplitude, low jitter"},
    "fearful": {"emoji": "😨", "color": "#F59E0B", "description": "Wide pitch excursions, breathy phonation, high ZCR"},
    "disgust": {"emoji": "🤢", "color": "#EC4899", "description": "Vocal fry, low pitch, creaky voice quality"},
    "surprised": {"emoji": "😲", "color": "#14B8A6", "description": "Abrupt pitch spikes, sharp energy onset"}
}


# --- Cache Model and Resource Loaders ---
@st.cache_resource
def get_tabular_models():
    svm_path = BASE_DIR / "models" / "svm_model.joblib"
    rf_path = BASE_DIR / "models" / "rf_model.joblib"
    svm = load_pipeline(str(svm_path)) if svm_path.exists() else None
    rf = load_pipeline(str(rf_path)) if rf_path.exists() else None
    return svm, rf


@st.cache_resource
def get_cnn_model():
    cnn_path = BASE_DIR / "models" / "cnn_model.pt"
    if cnn_path.exists():
        model, classes = load_cnn(str(cnn_path), device="cpu")
        return model, classes
    return None, []


@st.cache_data
def get_metrics_data():
    metrics_path = BASE_DIR / "models" / "metrics.json"
    if metrics_path.exists():
        return load_metrics(str(metrics_path))
    return None


@st.cache_data
def get_speaker_eval_data():
    speaker_path = BASE_DIR / "models" / "speaker_eval_results.json"
    if speaker_path.exists():
        return load_speaker_eval_results(str(speaker_path))
    return None


@st.cache_data
def get_manifest_data():
    manifest_path = BASE_DIR / "data" / "dataset_manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return None


# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Speech Emotion Recognition Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .emotion-badge {
        font-size: 1.6rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/voice-recognition.png", width=64)
    st.sidebar.title("SER Studio")
    st.sidebar.caption("Deep Learning & Acoustic Intelligence for Emotional Speech")

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Active Classifier")
    active_model_name = st.sidebar.selectbox(
        "Choose Model for Live Predictions",
        options=["Ensemble (SVM + RF + CNN)", "Support Vector Machine (SVM)", "Random Forest", "PyTorch CNN (Mel Spectrogram)"],
        index=0
    )

    # Check model readiness
    svm_model, rf_model = get_tabular_models()
    cnn_model, cnn_classes = get_cnn_model()
    metrics = get_metrics_data()
    speaker_eval = get_speaker_eval_data()
    manifest_df = get_manifest_data()

    models_ready = (svm_model is not None) and (rf_model is not None) and (cnn_model is not None)

    st.sidebar.divider()
    st.sidebar.subheader("🚦 Model System Status")
    if models_ready:
        st.sidebar.success("All 3 Models Loaded & Operational")
        if metrics:
            st.sidebar.metric("SVM Test Accuracy", f"{metrics.get('SVM', {}).get('accuracy', 0)*100:.1f}%")
            st.sidebar.metric("RF Test Accuracy", f"{metrics.get('Random Forest', {}).get('accuracy', 0)*100:.1f}%")
            st.sidebar.metric("CNN Test Accuracy", f"{metrics.get('CNN', {}).get('accuracy', 0)*100:.1f}%")
    else:
        st.sidebar.warning("Models not trained yet.")
        st.sidebar.info("Go to the 'Dataset & Training Hub' tab to run training.")

    st.sidebar.divider()
    st.sidebar.info(
        "**Dataset Source**: Authentic RAVDESS Benchmark (24 Professional Actors, 8 Emotions, Balanced Genders)."
    )

    # Header
    st.markdown('<div class="main-header">🎙️ Speech Emotion Recognition (SER) Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-modal acoustic feature extraction, classical ML pipelines, and 2D Spectrogram Convolutional Neural Networks.</div>', unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Inference & Waveform Lab",
        "🔬 Acoustic Feature Lab",
        "📊 Model Performance & Metrics",
        "👤 Speaker-Independent Evaluation",
        "📁 Dataset & Training Hub"
    ])

    # =========================================================================
    # TAB 1: Live Inference & Waveform Lab
    # =========================================================================
    with tab1:
        st.markdown("### 🎙️ Audio Input & Real-Time Emotion Classification")
        st.write("Upload any WAV / MP3 / FLAC audio file, or select from the library of authentic RAVDESS recordings.")

        input_col1, input_col2 = st.columns([1, 1])

        with input_col1:
            input_mode = st.radio("Input Source Mode", ["Select Preset Sample Audio", "Upload Custom Audio File"], horizontal=True)

        audio_bytes = None
        audio_filename = "sample.wav"

        with input_col2:
            if input_mode == "Select Preset Sample Audio":
                samples_dir = BASE_DIR / "data" / "samples"
                sample_files = list(samples_dir.glob("*.wav")) if samples_dir.exists() else []
                
                if sample_files:
                    sample_options = {f.name: f for f in sorted(sample_files)}
                    selected_sample_name = st.selectbox("Select Audio Sample", options=list(sample_options.keys()))
                    selected_path = sample_options[selected_sample_name]
                    with open(selected_path, "rb") as f:
                        audio_bytes = f.read()
                    audio_filename = selected_sample_name
                else:
                    st.info("No sample files found in data/samples yet. Please upload a file or train the models.")
            else:
                uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3, FLAC, OGG)", type=["wav", "mp3", "flac", "ogg"])
                if uploaded_file is not None:
                    audio_bytes = uploaded_file.read()
                    audio_filename = uploaded_file.name

        if audio_bytes is not None:
            st.divider()
            col_audio, col_meta = st.columns([1, 1])
            with col_audio:
                st.subheader("🔊 Audio Playback")
                st.audio(audio_bytes)
                st.caption(f"Loaded: `{audio_filename}`")

            # Run Real Preprocessing & Feature Extraction
            with st.spinner("Extracting acoustic features and spectrogram..."):
                y_proc, sr, proc_meta = preprocess_audio(audio_bytes, target_sr=16000, duration_sec=3.0)
                feat_vec, feat_names, vis_data = extract_all_tabular_features(y_proc, sr)
                mel_spec = extract_mel_spectrogram(y_proc, sr=sr, n_mels=128)

            with col_meta:
                st.subheader("📋 Preprocessing Summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Sample Rate", f"{proc_meta['sample_rate']} Hz")
                m2.metric("Raw Duration", f"{proc_meta['original_duration_sec']}s")
                m3.metric("RMS Energy", f"{proc_meta['rms_energy']:.4f}")
                st.caption(f"Silence trimmed: from {proc_meta['trimmed_start_sec']}s to {proc_meta['trimmed_end_sec']}s. Normalized to 3.0s window.")

            # Model Predictions
            st.divider()
            st.subheader("🤖 Emotion Predictions")

            if not models_ready:
                st.warning("Models are currently not loaded. Please train the models in the 'Dataset & Training Hub' tab first.")
            else:
                # 1. SVM Prediction
                svm_res = predict_single_sample(svm_model, feat_vec, svm_model.classes_)
                # 2. Random Forest Prediction
                rf_res = predict_single_sample(rf_model, feat_vec, rf_model.classes_)
                # 3. CNN Prediction
                cnn_res = predict_single_spectrogram(cnn_model, mel_spec, cnn_classes)

                # Active Model Decision
                all_emotions = sorted(list(svm_res["probabilities"].keys()))
                if "Ensemble" in active_model_name:
                    ensemble_probs = {}
                    for emo in all_emotions:
                        p_svm = svm_res["probabilities"].get(emo, 0.0)
                        p_rf = rf_res["probabilities"].get(emo, 0.0)
                        p_cnn = cnn_res["probabilities"].get(emo, 0.0)
                        ensemble_probs[emo] = (p_svm + p_rf + p_cnn) / 3.0
                    sorted_ensemble = dict(sorted(ensemble_probs.items(), key=lambda item: item[1], reverse=True))
                    active_pred = list(sorted_ensemble.keys())[0]
                    active_conf = sorted_ensemble[active_pred]
                    active_probs = sorted_ensemble
                elif "SVM" in active_model_name:
                    active_pred = svm_res["predicted_emotion"]
                    active_conf = svm_res["confidence"]
                    active_probs = svm_res["probabilities"]
                elif "Random Forest" in active_model_name:
                    active_pred = rf_res["predicted_emotion"]
                    active_conf = rf_res["confidence"]
                    active_probs = rf_res["probabilities"]
                else:
                    active_pred = cnn_res["predicted_emotion"]
                    active_conf = cnn_res["confidence"]
                    active_probs = cnn_res["probabilities"]

                meta = EMOTION_META.get(active_pred, {"emoji": "🎭", "color": "#3B82F6", "description": ""})

                # Top Result Card
                res_col1, res_col2 = st.columns([1, 1.4])

                with res_col1:
                    st.markdown(f"""
                    <div style="background-color: {meta['color']}15; border-left: 6px solid {meta['color']}; padding: 1.2rem; border-radius: 8px;">
                        <span style="font-size: 1.1rem; color: #475569; font-weight: 600;">Detected Emotion</span><br>
                        <span style="font-size: 2.2rem; font-weight: 700; color: {meta['color']};">{meta['emoji']} {active_pred.upper()}</span><br>
                        <span style="font-size: 1.1rem; font-weight: 600; color: #1E293B;">Confidence: {active_conf*100:.1f}%</span><br>
                        <p style="margin-top: 0.5rem; color: #64748B; font-size: 0.95rem;">{meta['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### 🔄 Cross-Model Comparison")
                    comp_pred_df = pd.DataFrame([
                        {"Model": "SVM (RBF)", "Predicted": svm_res["predicted_emotion"].title(), "Confidence": f"{svm_res['confidence']*100:.1f}%"},
                        {"Model": "Random Forest", "Predicted": rf_res["predicted_emotion"].title(), "Confidence": f"{rf_res['confidence']*100:.1f}%"},
                        {"Model": "PyTorch CNN", "Predicted": cnn_res["predicted_emotion"].title(), "Confidence": f"{cnn_res['confidence']*100:.1f}%"}
                    ])
                    st.dataframe(comp_pred_df, hide_index=True, use_container_width=True)

                with res_col2:
                    st.markdown("#### 📊 Emotion Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Emotion": [e.title() for e in active_probs.keys()],
                        "Probability": list(active_probs.values()),
                        "Color": [EMOTION_META.get(e, {}).get("color", "#3B82F6") for e in active_probs.keys()]
                    })
                    fig_prob = px.bar(
                        prob_df,
                        x="Probability",
                        y="Emotion",
                        orientation="h",
                        color="Emotion",
                        color_discrete_map={row["Emotion"]: row["Color"] for _, row in prob_df.iterrows()},
                        text=[f"{p*100:.1f}%" for p in prob_df["Probability"]]
                    )
                    fig_prob.update_layout(
                        showlegend=False,
                        height=280,
                        margin=dict(l=0, r=20, t=10, b=10),
                        xaxis=dict(range=[0, 1], tickformat=".0%"),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

            # Acoustic Visualizations
            st.divider()
            st.subheader("📈 Interactive Acoustic Visualizations")

            plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs([
                "Waveform & Energy",
                "Log Mel Spectrogram",
                "MFCC Heatmap",
                "Pitch (F0) & Spectral Dynamics"
            ])

            times = np.linspace(0, len(y_proc) / sr, len(y_proc))

            with plot_tab1:
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    fig_wave = go.Figure()
                    fig_wave.add_trace(go.Scatter(x=times, y=y_proc, mode='lines', line=dict(color='#3B82F6', width=1)))
                    fig_wave.update_layout(
                        title="Normalized Audio Waveform (Time Domain)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_wave, use_container_width=True)

                with col_w2:
                    rms_t = np.linspace(0, len(y_proc) / sr, len(vis_data["rms_curve"]))
                    fig_rms = go.Figure()
                    fig_rms.add_trace(go.Scatter(x=rms_t, y=vis_data["rms_curve"], fill='tozeroy', line=dict(color='#F59E0B')))
                    fig_rms.update_layout(
                        title="RMS Energy Contour (Arousal / Loudness)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="RMS Amplitude",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_rms, use_container_width=True)

            with plot_tab2:
                fig_mel = px.imshow(
                    mel_spec,
                    labels=dict(x="Time Frames", y="Mel Frequency Channel", color="dB"),
                    origin='lower',
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                fig_mel.update_layout(
                    title="2D Log Mel Spectrogram (Input to PyTorch CNN)",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_mel, use_container_width=True)

            with plot_tab3:
                fig_mfcc = px.imshow(
                    vis_data["mfcc_matrix"],
                    labels=dict(x="Time Frames", y="MFCC Coefficient Index", color="Value"),
                    origin='lower',
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig_mfcc.update_layout(
                    title="Mel-Frequency Cepstral Coefficients (MFCCs 1-20)",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_mfcc, use_container_width=True)

            with plot_tab4:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    pitch_t = vis_data["pitch_times"]
                    f0 = vis_data["pitch_contour"]
                    voiced_mask = (f0 > 65) & (f0 < 400)
                    fig_pitch = go.Figure()
                    fig_pitch.add_trace(go.Scatter(
                        x=pitch_t[voiced_mask],
                        y=f0[voiced_mask],
                        mode='markers+lines',
                        marker=dict(size=4, color='#10B981'),
                        line=dict(color='#10B981', width=1.5)
                    ))
                    fig_pitch.update_layout(
                        title=f"Fundamental Frequency Pitch F0 (Mean: {vis_data['feature_dict']['pitch_mean']:.1f} Hz)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Frequency (Hz)",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_pitch, use_container_width=True)

                with col_p2:
                    spec_t = np.linspace(0, len(y_proc) / sr, len(vis_data["spectral_curves"]["spectral_centroid"]))
                    fig_cent = go.Figure()
                    fig_cent.add_trace(go.Scatter(
                        x=spec_t,
                        y=vis_data["spectral_curves"]["spectral_centroid"],
                        mode='lines',
                        line=dict(color='#8B5CF6'),
                        name="Centroid"
                    ))
                    fig_cent.add_trace(go.Scatter(
                        x=spec_t,
                        y=vis_data["spectral_curves"]["spectral_rolloff"],
                        mode='lines',
                        line=dict(color='#EC4899', dash='dot'),
                        name="85% Rolloff"
                    ))
                    fig_cent.update_layout(
                        title="Spectral Centroid & Rolloff (Timbre Brightness)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Frequency (Hz)",
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_cent, use_container_width=True)

    # =========================================================================
    # TAB 2: Acoustic Feature Lab
    # =========================================================================
    with tab2:
        st.markdown("### 🔬 Acoustic Feature Science & Emotional Correlates")
        st.write("Understand the physical acoustics behind vocal emotion encoding and compare acoustic signatures.")

        feat_col1, feat_col2 = st.columns([1, 1])

        with feat_col1:
            st.markdown("""
            #### 1. Mel-Frequency Cepstral Coefficients (MFCCs)
            - **What it captures**: The envelope of the short-time power spectrum, reflecting the shape of the vocal tract.
            - **Emotional Role**: Differentiates between phonemes and subtle timbre changes between harsh vocal fry (anger) and soft, breathy tones (sadness).

            #### 2. Fundamental Frequency (Pitch / F0)
            - **What it captures**: Rate of vibration of vocal folds.
            - **Emotional Role**:
              - **Anger / Fear / Joy**: High mean pitch and wide pitch variance.
              - **Sadness / Calm**: Lower mean pitch, monotone contours.
            """)

        with feat_col2:
            st.markdown("""
            #### 3. Root Mean Square (RMS) Energy
            - **What it captures**: Instantaneous signal loudness and vocal effort.
            - **Emotional Role**: Strongly correlated with psychological arousal. High arousal emotions (Angry, Happy) exhibit 3-5x higher RMS than low arousal emotions (Sad, Neutral).

            #### 4. Zero-Crossing Rate (ZCR) & Spectral Features
            - **What it captures**: Noise frequency content, spectral brightness (Centroid), and spectral spread (Bandwidth).
            - **Emotional Role**: High ZCR and spectral centroid indicate whispery breathiness or harsh percussive bursts.
            """)

        st.divider()
        st.subheader("⚖️ Emotion Acoustic Comparator")
        st.write("Select two sample audio files to compare their acoustic feature statistics side-by-side:")

        samples_dir = BASE_DIR / "data" / "samples"
        sample_files = sorted(list(samples_dir.glob("*.wav"))) if samples_dir.exists() else []

        if len(sample_files) >= 2:
            comp_c1, comp_c2 = st.columns(2)
            with comp_c1:
                s1_name = st.selectbox("Sample Audio A", options=[f.name for f in sample_files], index=0)
            with comp_c2:
                s2_name = st.selectbox("Sample Audio B", options=[f.name for f in sample_files], index=min(1, len(sample_files)-1))

            s1_path = samples_dir / s1_name
            s2_path = samples_dir / s2_name

            y1, sr1, m1 = preprocess_audio(str(s1_path))
            y2, sr2, m2 = preprocess_audio(str(s2_path))

            _, _, vis1 = extract_all_tabular_features(y1, sr1)
            _, _, vis2 = extract_all_tabular_features(y2, sr2)

            comp_stats = pd.DataFrame([
                {
                    "Feature": "RMS Energy (Mean)",
                    s1_name: round(vis1["feature_dict"]["rms_mean"], 4),
                    s2_name: round(vis2["feature_dict"]["rms_mean"], 4),
                    "Difference": round(vis1["feature_dict"]["rms_mean"] - vis2["feature_dict"]["rms_mean"], 4)
                },
                {
                    "Feature": "Pitch F0 Mean (Hz)",
                    s1_name: round(vis1["feature_dict"]["pitch_mean"], 1),
                    s2_name: round(vis2["feature_dict"]["pitch_mean"], 1),
                    "Difference": round(vis1["feature_dict"]["pitch_mean"] - vis2["feature_dict"]["pitch_mean"], 1)
                },
                {
                    "Feature": "Pitch F0 Std Dev (Hz)",
                    s1_name: round(vis1["feature_dict"]["pitch_std"], 1),
                    s2_name: round(vis2["feature_dict"]["pitch_std"], 1),
                    "Difference": round(vis1["feature_dict"]["pitch_std"] - vis2["feature_dict"]["pitch_std"], 1)
                },
                {
                    "Feature": "Zero Crossing Rate",
                    s1_name: round(vis1["feature_dict"]["zcr_mean"], 4),
                    s2_name: round(vis2["feature_dict"]["zcr_mean"], 4),
                    "Difference": round(vis1["feature_dict"]["zcr_mean"] - vis2["feature_dict"]["zcr_mean"], 4)
                },
                {
                    "Feature": "Spectral Centroid (Hz)",
                    s1_name: round(vis1["feature_dict"]["spectral_centroid_mean"], 1),
                    s2_name: round(vis2["feature_dict"]["spectral_centroid_mean"], 1),
                    "Difference": round(vis1["feature_dict"]["spectral_centroid_mean"] - vis2["feature_dict"]["spectral_centroid_mean"], 1)
                },
                {
                    "Feature": "Spectral Flatness",
                    s1_name: round(vis1["feature_dict"]["spectral_flatness_mean"], 5),
                    s2_name: round(vis2["feature_dict"]["spectral_flatness_mean"], 5),
                    "Difference": round(vis1["feature_dict"]["spectral_flatness_mean"] - vis2["feature_dict"]["spectral_flatness_mean"], 5)
                }
            ])
            st.dataframe(comp_stats, use_container_width=True, hide_index=True)
        else:
            st.info("Sample files are being prepared. Once training completes, the comparison tool will be active.")

    # =========================================================================
    # TAB 3: Model Performance & Metrics
    # =========================================================================
    with tab3:
        st.markdown("### 📊 Genuine Model Performance & Test Data Evaluation")
        st.write("All metrics below are computed on an authentic hold-out test set (20% of dataset). No fabricated numbers.")

        if metrics is None:
            st.warning("Evaluation metrics have not been generated yet. Please run training from the 'Dataset & Training Hub' tab.")
        else:
            comp_df = build_comparison_dataframe(metrics)
            st.markdown("#### 🏆 Benchmark Summary Table")
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Metric Comparison Bar Chart
            st.markdown("#### 📈 Multi-Metric Comparative Analysis")
            chart_data = []
            for m_name, m_vals in metrics.items():
                chart_data.append({"Model": m_name, "Metric": "Accuracy", "Score": m_vals["accuracy"]})
                chart_data.append({"Model": m_name, "Metric": "Macro F1", "Score": m_vals["f1_macro"]})
                chart_data.append({"Model": m_name, "Metric": "Weighted F1", "Score": m_vals["f1_weighted"]})
                chart_data.append({"Model": m_name, "Metric": "Precision", "Score": m_vals["precision_macro"]})
                chart_data.append({"Model": m_name, "Metric": "Recall", "Score": m_vals["recall_macro"]})

            cdf = pd.DataFrame(chart_data)
            fig_bar = px.bar(
                cdf,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                text=[f"{v*100:.1f}%" for v in cdf["Score"]],
                color_discrete_sequence=["#3B82F6", "#10B981", "#8B5CF6"]
            )
            fig_bar.update_layout(
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                height=380,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Confusion Matrices
            st.divider()
            st.markdown("#### 🧩 Confusion Matrices")
            cm_col1, cm_col2 = st.columns([1, 3])

            with cm_col1:
                chosen_cm_model = st.selectbox("Select Model Confusion Matrix", options=list(metrics.keys()))
                norm_toggle = st.radio("Display Mode", ["Normalized Percentages (%)", "Raw Sample Counts"])
                is_norm = "Normalized" in norm_toggle

            with cm_col2:
                model_m = metrics[chosen_cm_model]
                classes = model_m["class_names"]
                cm_matrix = model_m["confusion_matrix_norm"] if is_norm else model_m["confusion_matrix"]
                
                # Format text
                if is_norm:
                    text_matrix = [[f"{v*100:.1f}%" for v in row] for row in cm_matrix]
                else:
                    text_matrix = [[str(v) for v in row] for row in cm_matrix]

                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_matrix,
                    x=[c.title() for c in classes],
                    y=[c.title() for c in classes],
                    text=text_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    colorscale="Blues",
                    colorbar=dict(title="%" if is_norm else "Count")
                ))
                fig_cm.update_layout(
                    title=f"{chosen_cm_model} Confusion Matrix ({'Normalized' if is_norm else 'Raw Counts'})",
                    xaxis_title="Predicted Emotion",
                    yaxis_title="True Emotion",
                    yaxis=dict(autorange="reversed"),
                    height=450,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            # Per-Class Precision / Recall / F1 Breakdown
            st.divider()
            st.markdown(f"#### 📑 Per-Class Performance Breakdown: `{chosen_cm_model}`")
            clf_rep = model_m.get("classification_report", {})
            class_rows = []
            for cls in classes:
                if cls in clf_rep:
                    class_rows.append({
                        "Emotion": cls.title(),
                        "Precision": f"{clf_rep[cls]['precision']*100:.1f}%",
                        "Recall": f"{clf_rep[cls]['recall']*100:.1f}%",
                        "F1-Score": f"{clf_rep[cls]['f1-score']*100:.1f}%",
                        "Support (Test Count)": int(clf_rep[cls]['support'])
                    })
            if class_rows:
                st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)

    # =========================================================================
    # TAB 4: Speaker-Independent Evaluation
    # =========================================================================
    with tab4:
        st.markdown("### 👤 Speaker-Independent Evaluation (Cross-Actor Generalization)")
        st.write(
            "In Speech Emotion Recognition, **speaker leakage** occurs when standard random splits place samples from the same "
            "speaker in both training and test sets. The model partially overfits to individual vocal timbre. "
            "**Speaker-independent evaluation** strictly isolates entire actors into hold-out test folds."
        )

        if speaker_eval is None:
            st.warning("Speaker evaluation data is not available yet. Please run training from the 'Dataset & Training Hub' tab.")
        else:
            sd = speaker_eval["speaker_dependent"]
            si = speaker_eval["speaker_independent"]
            gap = speaker_eval["generalization_gap"]

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric(
                    "Speaker-Dependent (Random Split)",
                    f"{sd['svm_accuracy']*100:.1f}%",
                    help="Same speakers present in train & test"
                )
            with sc2:
                st.metric(
                    "Speaker-Independent (GroupKFold)",
                    f"{si['svm_accuracy_mean']*100:.1f}%",
                    f"-{gap['svm_gap_percentage']:.1f}% gap",
                    delta_color="inverse",
                    help="Speakers strictly unseen during training"
                )
            with sc3:
                st.metric(
                    "RF Generalization Gap",
                    f"{gap['rf_gap_percentage']:.1f}% drop",
                    delta_color="inverse"
                )

            st.markdown("#### 🔬 Analysis & Findings")
            st.info(
                f"**Generalization Gap**: The SVM accuracy drops by **{gap['svm_gap_percentage']:.1f}%** when tested on unseen speakers. "
                "This behavior is well-documented in academic SER literature: emotional expressions carry individual baseline pitch "
                "and vocal tract differences that require adaptive normalization or larger multi-speaker corpora to generalize."
            )

            # Fold breakdown
            st.markdown("#### 📋 5-Fold GroupKFold Detailed Results")
            fold_df = pd.DataFrame(si["fold_details"])
            fold_df = fold_df.rename(columns={
                "fold": "Fold",
                "test_actors": "Held-Out Actors",
                "num_test_samples": "Test Samples",
                "svm_accuracy": "SVM Accuracy",
                "svm_f1": "SVM Macro F1",
                "rf_accuracy": "RF Accuracy",
                "rf_f1": "RF Macro F1"
            })
            fold_df["SVM Accuracy"] = fold_df["SVM Accuracy"].apply(lambda v: f"{v*100:.1f}%")
            fold_df["SVM Macro F1"] = fold_df["SVM Macro F1"].apply(lambda v: f"{v*100:.1f}%")
            fold_df["RF Accuracy"] = fold_df["RF Accuracy"].apply(lambda v: f"{v*100:.1f}%")
            fold_df["RF Macro F1"] = fold_df["RF Macro F1"].apply(lambda v: f"{v*100:.1f}%")
            st.dataframe(fold_df, hide_index=True, use_container_width=True)

    # =========================================================================
    # TAB 5: Dataset & Training Hub
    # =========================================================================
    with tab5:
        st.markdown("### 📁 Dataset & Model Training Pipeline")
        st.write("Inspect the underlying authentic RAVDESS dataset structure and trigger model training on demand.")

        if manifest_df is not None:
            d_c1, d_c2, d_c3, d_c4 = st.columns(4)
            d_c1.metric("Total Audio Samples", len(manifest_df))
            d_c2.metric("Unique Actors", manifest_df["actor_id"].nunique())
            d_c3.metric("Emotion Classes", manifest_df["emotion"].nunique())
            d_c4.metric("Gender Ratio", f"12 M / 12 F")

            col_pie1, col_pie2 = st.columns(2)
            with col_pie1:
                emo_counts = manifest_df["emotion"].value_counts().reset_index()
                emo_counts.columns = ["Emotion", "Count"]
                fig_pie = px.pie(
                    emo_counts,
                    names="Emotion",
                    values="Count",
                    title="Emotion Class Distribution in Dataset",
                    color="Emotion",
                    color_discrete_map={k: v["color"] for k, v in EMOTION_META.items()}
                )
                fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_pie2:
                gender_counts = manifest_df["gender"].value_counts().reset_index()
                gender_counts.columns = ["Gender", "Count"]
                fig_gen = px.bar(
                    gender_counts,
                    x="Gender",
                    y="Count",
                    color="Gender",
                    title="Gender Balance (Actors)",
                    color_discrete_sequence=["#3B82F6", "#EC4899"]
                )
                fig_gen.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                st.plotly_chart(fig_gen, use_container_width=True)

            st.divider()
            st.markdown("#### 🎧 Explore Dataset Audio Clips")
            sample_filter = st.selectbox("Filter by Emotion", options=["All"] + sorted(manifest_df["emotion"].unique().tolist()))
            filtered_df = manifest_df if sample_filter == "All" else manifest_df[manifest_df["emotion"] == sample_filter]
            st.dataframe(filtered_df[["filename", "emotion", "actor_id", "gender", "intensity", "statement"]].head(25), use_container_width=True)
        else:
            st.info("Dataset manifest is not generated yet. Click 'Run Dataset Prep & Training' below.")

        st.divider()
        st.subheader("🚀 Retrain Models Pipeline")
        st.write("Trigger complete end-to-end retraining of SVM, Random Forest, and PyTorch CNN.")

        recompute_features = st.checkbox("Force recompute feature cache from scratch", value=False)
        cnn_epoch_count = st.slider("CNN Training Epochs", min_value=5, max_value=30, value=15, step=5)

        if st.button("▶️ Run End-to-End Training Pipeline", type="primary"):
            status_box = st.empty()
            prog_bar = st.progress(0.0)

            try:
                status_box.info("Step 1/4: Preparing dataset and audio manifest...")
                prog_bar.progress(0.15)
                from scripts.download_or_prep_data import download_dataset, build_manifest
                download_dataset()
                build_manifest()

                status_box.info("Step 2/4: Extracting acoustic tabular features and Mel spectrograms...")
                prog_bar.progress(0.40)

                status_box.info("Step 3/4: Training SVM, Random Forest, and PyTorch CNN...")
                prog_bar.progress(0.70)
                from scripts.train_all import run_training_pipeline
                run_training_pipeline(force_recompute=recompute_features, cnn_epochs=cnn_epoch_count)

                prog_bar.progress(1.0)
                status_box.success("🎉 Training complete! Models and metrics have been saved.")
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                status_box.error(f"Error during training pipeline: {str(e)}")


if __name__ == "__main__":
    main()
