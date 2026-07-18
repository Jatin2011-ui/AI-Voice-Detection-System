"""
app.py
------
Streamlit dashboard for the AI Voice Authenticity Detector.

Run with:
  streamlit run app.py

Sections:
  - Sidebar    : About, How it works
  - Header     : Hero banner with badges
  - Audio Input: Upload WAV/MP3 or record from microphone
  - Playback   : Inline audio player
  - Transcription: Whisper STT output with BERT analysis indicator
  - Prediction : Real vs AI-Generated verdict + confidence meter
  - Visualization: Waveform + Mel Spectrogram tabs
"""

import os
import sys
import json
import time
import tempfile
import warnings

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_audio, load_model, MODEL_PATH
from utils.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_confidence_meter,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Voice Authenticity Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark premium theme) ────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root theme overrides ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a1628 100%);
    background-attachment: fixed;
}

/* ── Hide default Streamlit elements ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 40px rgba(66, 153, 225, 0.08);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(66,153,225,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #ed64a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #a0aec0;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.6;
    max-width: 680px;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(99, 179, 237, 0.1);
    border: 1px solid rgba(99, 179, 237, 0.25);
    border-radius: 30px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #63b3ed;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.badge-bert {
    background: rgba(159, 122, 234, 0.12);
    border: 1px solid rgba(159, 122, 234, 0.3);
    border-radius: 30px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #9f7aea;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* ── Section cards ── */
.section-card {
    background: rgba(22, 28, 46, 0.85);
    border: 1px solid rgba(99, 179, 237, 0.12);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: border-color 0.3s ease;
}
.section-card:hover {
    border-color: rgba(99, 179, 237, 0.25);
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Result panels ── */
.result-real {
    background: linear-gradient(135deg, rgba(0,200,150,0.08), rgba(0,200,150,0.03));
    border: 2px solid rgba(0, 200, 150, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-fake {
    background: linear-gradient(135deg, rgba(255,75,110,0.08), rgba(255,75,110,0.03));
    border: 2px solid rgba(255, 75, 110, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-confidence {
    font-size: 1rem;
    color: #a0aec0;
    margin-top: 0.4rem;
}
.result-icon {
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}

/* ── Metric boxes ── */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.metric-box {
    background: rgba(15, 20, 40, 0.6);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    flex: 1;
    min-width: 100px;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #63b3ed;
}
.metric-label {
    font-size: 0.72rem;
    color: #718096;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Transcription box ── */
.transcription-box {
    background: rgba(10, 14, 26, 0.8);
    border-left: 3px solid #9f7aea;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.7;
    font-style: italic;
}
.lang-badge {
    display: inline-block;
    background: rgba(159, 122, 234, 0.15);
    border: 1px solid rgba(159, 122, 234, 0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #9f7aea;
    margin-bottom: 0.6rem;
    font-style: normal;
}
.bert-badge {
    display: inline-block;
    background: rgba(237, 100, 166, 0.12);
    border: 1px solid rgba(237, 100, 166, 0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #ed64a6;
    margin-bottom: 0.6rem;
    margin-left: 6px;
    font-style: normal;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a1628 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.1);
}
.sidebar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #9f7aea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 0.5rem 0 1rem;
}
.sidebar-section {
    background: rgba(22, 33, 62, 0.5);
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-section h4 {
    color: #63b3ed;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
}
.sidebar-section p, .sidebar-section li {
    color: #718096;
    font-size: 0.83rem;
    line-height: 1.6;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(22, 28, 46, 0.5) !important;
    border: 2px dashed rgba(99, 179, 237, 0.25) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 179, 237, 0.5) !important;
    background: rgba(99, 179, 237, 0.04) !important;
}

/* ── Audio player ── */
audio {
    width: 100%;
    border-radius: 8px;
}

/* ── Info/warning blocks ── */
.info-tip {
    background: rgba(99, 179, 237, 0.06);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #a0aec0;
    font-size: 0.85rem;
}

/* ── Step number indicator ── */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #63b3ed, #9f7aea);
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    margin-right: 8px;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper — load trained model
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_trained_model():
    """Load and cache the trained model. Returns None if not trained yet."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except FileNotFoundError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎙️ VoiceAuth AI</div>', unsafe_allow_html=True)
    st.markdown("---")

    # About section
    st.markdown("""
    <div class="sidebar-section">
        <h4>📌 About</h4>
        <p>An AI-powered tool that detects whether a voice recording is from
        a <strong style="color:#00C896">real human</strong> or an
        <strong style="color:#FF4B6E">AI-generated</strong> speech system.</p>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="sidebar-section">
        <h4>⚙️ How It Works</h4>
        <ul>
            <li>Whisper STT transcribes the audio to text</li>
            <li>DistilBERT encodes text into semantic embeddings</li>
            <li>Librosa extracts acoustic audio features</li>
            <li>RandomForest classifies using both feature types</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Hero Banner
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎙️ AI Voice Authenticity Detector</div>
    <div class="hero-subtitle">
        Upload or record a voice sample. Our hybrid AI model — combining
        <strong>acoustic analysis</strong> with <strong>DistilBERT language understanding</strong>
        via <strong>Whisper speech-to-text</strong> — will determine whether the voice is
        a <strong>real human</strong> or <strong>AI-generated</strong>.
    </div>
    <div class="hero-badges">
        <span class="badge">🎵 Librosa Audio Features</span>
        <span class="badge-bert">🤖 DistilBERT NLP</span>
        <span class="badge">📝 Whisper STT</span>
        <span class="badge">🌲 RandomForest Classifier</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model not trained banner
model = get_trained_model()
if model is None:
    st.warning(
        "⚠️ **Model not found.** Please train the model first by running:\n\n"
        "```\npython train_model.py\n```\n\n"
        "Training uses Whisper + DistilBERT + Librosa features (~10–30 min depending on dataset size).",
        icon="⚠️"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Audio Input
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span class="step-num">1</span>Audio Input</div>', unsafe_allow_html=True)

col_upload, col_record = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown("**📁 Upload Audio File**")
    uploaded_file = st.file_uploader(
        label="Drag & drop or click to browse",
        type=["wav", "mp3"],
        help="Supported formats: WAV, MP3. Max recommended: 60 seconds.",
        key="audio_uploader"
    )

with col_record:
    st.markdown("**🎤 Record from Microphone**")
    st.markdown("""
    <div class="info-tip">
        Use the built-in browser recorder to capture your microphone directly.
        The recording will be treated the same as an uploaded file.
    </div>
    """, unsafe_allow_html=True)
    try:
        from streamlit_mic_recorder import mic_recorder
        mic_audio = mic_recorder(
            start_prompt="🔴 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            key="mic_recorder"
        )
    except ImportError:
        mic_audio = None
        st.markdown("""
        <div class="info-tip" style="border-color:rgba(255,193,7,0.3); background:rgba(255,193,7,0.05);">
            🎤 Install <code>streamlit-mic-recorder</code> for live recording:<br>
            <code>pip install streamlit-mic-recorder</code>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Resolve which audio to use ─────────────────────────────────────────────────
audio_bytes  = None
audio_source = None

if uploaded_file is not None:
    audio_bytes  = uploaded_file.read()
    audio_source = uploaded_file.name
elif mic_audio is not None and "bytes" in mic_audio:
    audio_bytes  = mic_audio["bytes"]
    audio_source = "Microphone Recording"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Audio Player + Transcription + Analysis (only if audio loaded)
# ══════════════════════════════════════════════════════════════════════════════

if audio_bytes:

    # Save audio to a temp file so librosa / whisper can read it
    suffix = ".wav"
    if audio_source and audio_source.lower().endswith(".mp3"):
        suffix = ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # ── Audio Player ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step-num">2</span>Audio Playback</div>', unsafe_allow_html=True)
    st.markdown(f"**🎵 Now playing:** `{audio_source}`")
    st.audio(audio_bytes, format="audio/wav")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Analysis columns ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Speech-to-Text (Left Column) ──────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-card" style="height:100%">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="step-num">3</span>Speech-to-Text Transcription</div>', unsafe_allow_html=True)

        transcribe_btn = st.button(
            "📝 Transcribe with Whisper",
            key="btn_transcribe"
        )

        if transcribe_btn:
            with st.spinner("🔄 Loading Whisper model and transcribing..."):
                try:
                    from utils.whisper_transcriber import transcribe_audio
                    result = transcribe_audio(tmp_path, model_size="base")

                    text     = result["text"] or "_[No speech detected]_"
                    lang     = result["language"].upper()
                    segments = result["segments"]

                    st.markdown(f"""
                    <span class="lang-badge">🌐 Language: {lang}</span>
                    <span class="bert-badge">🤖 BERT-ready text</span>
                    <div class="transcription-box">"{text}"</div>
                    """, unsafe_allow_html=True)

                    if segments:
                        with st.expander("📋 View Timed Segments"):
                            for seg in segments:
                                st.markdown(
                                    f"`[{seg['start']:.1f}s – {seg['end']:.1f}s]` {seg['text']}"
                                )

                except Exception as e:
                    st.error(f"Transcription failed: {e}")

        else:
            st.markdown("""
            <div class="info-tip">
                Click the button above to convert the voice recording into text
                using OpenAI Whisper (base model, runs locally — no API key needed).
                The transcription is also used by DistilBERT for deeper analysis.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Prediction (Right Column) ─────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-card" style="height:100%">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="step-num">4</span>AI Voice Detection</div>', unsafe_allow_html=True)

        analyze_btn = st.button(
            "🔍 Analyze Voice",
            type="primary",
            key="btn_analyze",
            disabled=(model is None)
        )

        if model is None:
            st.markdown("""
            <div class="info-tip" style="margin-top:0.5rem">
                ⚠️ Train the model first: <code>python train_model.py</code>
            </div>
            """, unsafe_allow_html=True)

        if analyze_btn and model is not None:
            with st.spinner("🔄 Transcribing → BERT encoding → Audio features → Predicting..."):
                try:
                    result = predict_audio(tmp_path, model_path=MODEL_PATH)

                    label         = result["label"]
                    confidence    = result["confidence"]
                    is_fake       = result["is_fake"]
                    proba_real    = result["proba_real"] * 100
                    proba_fake    = result["proba_fake"] * 100
                    transcription = result.get("transcription", "")

                    # Result card
                    card_class = "result-fake" if is_fake else "result-real"
                    icon       = "🤖" if is_fake else "✅"
                    color      = "#FF4B6E" if is_fake else "#00C896"

                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="result-icon">{icon}</div>
                        <div class="result-label" style="color:{color}">{label}</div>
                        <div class="result-confidence">Confidence: <strong>{confidence:.1f}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show transcription used for BERT
                    if transcription:
                        st.markdown("**📝 Whisper → BERT Input:**")
                        st.markdown(f"""
                        <div class="transcription-box" style="font-size:0.82rem; margin-bottom:0.8rem">
                            {transcription[:300]}{'...' if len(transcription) > 300 else ''}
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability bars
                    st.markdown("**Probability Breakdown:**")
                    col_r, col_f = st.columns(2)
                    with col_r:
                        st.metric("✅ Real", f"{proba_real:.1f}%")
                    with col_f:
                        st.metric("🤖 AI Generated", f"{proba_fake:.1f}%")

                    # Confidence bar figure
                    fig_conf = plot_confidence_meter(confidence, label, is_fake)
                    st.pyplot(fig_conf)

                    # Store result in session state
                    st.session_state["last_result"] = result

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

        elif not analyze_btn:
            st.markdown("""
            <div class="info-tip">
                Click <strong>Analyze Voice</strong> to run the hybrid
                Audio + BERT model on your recording. The model will:
                <ol style="margin-top:0.4rem">
                    <li>Transcribe with Whisper</li>
                    <li>Encode text with DistilBERT</li>
                    <li>Extract acoustic features</li>
                    <li>Predict Real vs AI-Generated</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature Visualizations ────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="step-num">5</span>Audio Visualizations</div>', unsafe_allow_html=True)

    viz_tab1, viz_tab2 = st.tabs(["🔊 Waveform", "🎵 Mel Spectrogram"])

    with viz_tab1:
        with st.spinner("Rendering waveform..."):
            try:
                fig_wave = plot_waveform(tmp_path, label=audio_source)
                st.pyplot(fig_wave)
                st.caption(
                    "The **waveform** shows amplitude (loudness) over time. "
                    "AI-generated voices may appear unnaturally smooth."
                )
            except Exception as e:
                st.error(f"Could not plot waveform: {e}")

    with viz_tab2:
        with st.spinner("Computing Mel spectrogram..."):
            try:
                fig_spec = plot_spectrogram(tmp_path, label=audio_source)
                st.pyplot(fig_spec)
                st.caption(
                    "The **Mel spectrogram** shows frequency content over time "
                    "(brighter = louder). AI voices often show periodic or tiled patterns."
                )
            except Exception as e:
                st.error(f"Could not plot spectrogram: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Cleanup temp file ─────────────────────────────────────────────────────
    try:
        os.unlink(tmp_path)
    except Exception:
        pass  # Best-effort cleanup

else:
    # ── Empty state ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-card" style="text-align:center; padding:3rem 2rem">
        <div style="font-size:4rem; margin-bottom:1rem">🎙️</div>
        <div style="font-size:1.2rem; font-weight:600; color:#e2e8f0; margin-bottom:0.5rem">
            No audio loaded yet
        </div>
        <div style="color:#718096; font-size:0.95rem; max-width:400px; margin:0 auto;">
            Upload a WAV or MP3 file using the panel above, or record directly
            from your microphone to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding:0.5rem 0">
    🎙️ <strong style="color:#63b3ed">AI Voice Authenticity Detector</strong> &nbsp;·&nbsp;
    Whisper STT · DistilBERT · Librosa · Scikit-learn · Streamlit &nbsp;·&nbsp;
    <em>For educational &amp; research purposes</em>
</div>
""", unsafe_allow_html=True)
