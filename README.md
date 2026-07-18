# 🎙️ AI Voice Authenticity Detector

> **Determine whether a voice recording is from a Real Human or AI-Generated.**

A machine learning project that combines **acoustic audio analysis** with **DistilBERT language understanding** and **OpenAI Whisper speech-to-text** to detect synthetic / AI-generated voices with high accuracy.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-Educational-green?style=flat-square)

---

## 📊 Model Performance

Trained and evaluated on **12,006 samples** (6,019 real · 5,987 AI-generated):

| Metric       | Score      |
|--------------|------------|
| ✅ Accuracy  | **98.08%** |
| 🎯 Precision | **97.14%** |
| 🔁 Recall    | **99.08%** |
| 📐 F1 Score  | **98.10%** |

**Confusion Matrix (test set)**

|                   | Predicted Real | Predicted Fake |
|-------------------|:--------------:|:--------------:|
| **Actual Real**   | 1169           | 35             |
| **Actual Fake**   | 11             | 1187           |

> Feature vector: **811-dimensional** hybrid (43 acoustic + 768 DistilBERT).  
> Model: `RandomForestClassifier` · 200 trees · Stratified 80/20 split.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 🎵 Audio Upload | Upload WAV or MP3 voice recordings |
| 🎤 Microphone Recording | Record directly in the browser (optional plugin) |
| 🔊 Audio Playback | Play uploaded audio in the browser |
| 📝 Speech-to-Text | Whisper transcribes speech to text locally (no API key needed) |
| 🤖 Hybrid AI Detection | Acoustic features + DistilBERT NLP → RandomForest prediction |
| 📊 Confidence Score | Shows prediction probability with visual meter |
| 📈 Visualizations | Waveform & Mel Spectrogram charts |
| 🔒 Fully Local | No API keys, no cloud — everything runs on your machine |

---

## 📁 Project Structure

```
ai-voice-detect/
│
├── app.py                    # Main Streamlit dashboard
├── train_model.py            # ML training pipeline (Whisper + BERT + Librosa)
├── predict.py                # Standalone CLI prediction script
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── model/
│   ├── voice_detector.pkl    # Saved trained model (generated after training)
│   ├── metrics.json          # Evaluation metrics (generated after training)
│   └── README.txt
│
├── utils/
│   ├── __init__.py
│   ├── feature_extractor.py  # Librosa audio feature extraction (43-dim)
│   ├── whisper_transcriber.py# OpenAI Whisper STT wrapper
│   └── visualization.py      # Matplotlib plots (waveform, spectrogram)
│
├── dataset/
│   ├── Dataset.csv           # Metadata CSV (Kaggle dataset)
│   ├── Real/Real/            # Real human voice recordings (.wav)
│   └── Fake/Fake/            # AI-generated voice recordings (.wav)
│
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or later ([download](https://www.python.org/downloads/))
- FFmpeg (required by Whisper for audio decoding)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-voice-detect.git
cd ai-voice-detect
```

### 2. Install FFmpeg (Windows)

**Option A – Using winget (easiest):**
```powershell
winget install ffmpeg
```

**Option B – Manual:**
1. Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your Windows PATH environment variable
4. Verify: `ffmpeg -version`

### 3. Create a Virtual Environment

```bash
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 **Note:** The first install may take a few minutes as PyTorch (required by Whisper) is large (~2 GB). For CPU-only:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 5. (Optional) Enable Microphone Recording

```bash
pip install streamlit-mic-recorder
```

---

## 📂 Dataset Setup

The project supports two data sources that are **combined** during training:

### Audio Files (`dataset/`)
```
dataset/
  Real/Real/   ← WAV files of real human voices
  Fake/Fake/   ← WAV files of AI-generated voices
  Dataset.csv  ← Metadata CSV (Kaggle format)
```

### Kaggle CSV Dataset
The `Dataset.csv` file follows the Kaggle AI vs Real voice dataset format with columns:
- `LABEL` — `REAL` or `FAKE`
- Pre-computed audio features (used directly without re-extraction)

> The training pipeline automatically combines both sources for a richer, more balanced dataset.  
> For more data: [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) — organize into `Real/Real/` and `Fake/Fake/`.

---

## 🏋️ Training the Model

```bash
python train_model.py
```

**What the pipeline does:**
1. Loads audio files from `dataset/Real/Real/` and `dataset/Fake/Fake/`
2. Transcribes each file with **OpenAI Whisper** (base model)
3. Encodes transcriptions with **DistilBERT** (768-dim embeddings)
4. Extracts **43 acoustic features** using Librosa (MFCC, Chroma, Spectral, etc.)
5. Combines both into an **811-dimensional** feature vector
6. Also reads and incorporates `Dataset.csv` (Kaggle pre-computed features)
7. Trains a `RandomForestClassifier` (200 trees, balanced class weights)
8. Evaluates on a **stratified 20% held-out test set**
9. Saves model → `model/voice_detector.pkl`
10. Saves metrics → `model/metrics.json`

**Expected results (current trained model):**
```
════════════════════════════════════════════════════════
              📈 MODEL EVALUATION RESULTS
════════════════════════════════════════════════════════
  Accuracy  : 98.08%
  Precision : 97.14%
  Recall    : 99.08%
  F1 Score  : 98.10%

  Dataset   : 12,006 samples (6,019 real · 5,987 fake)
  Features  : 811-dim (43 acoustic + 768 BERT)
════════════════════════════════════════════════════════

💾 Model saved to: model/voice_detector.pkl
```

> ⏱️ Training takes **10–30 minutes** depending on dataset size and CPU speed, as each audio file is processed by Whisper and DistilBERT.

---

## 🚀 Running the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**Workflow:**
1. **Upload** a WAV or MP3 file (or record from microphone)
2. **Play** the audio in the browser
3. Click **Transcribe with Whisper** to convert speech → text
4. Click **Analyze Voice** to run the hybrid model
5. View the **Waveform** and **Mel Spectrogram** visualizations

---

## 🧪 CLI Prediction (Optional)

```bash
python predict.py path/to/your/audio.wav
```

**Example output:**
```
🎙️  Analyzing: your_audio.wav
─────────────────────────────────────────────
  Prediction  : 🤖 AI Generated Voice
  Confidence  : 94.3%
  Real prob   : 5.7%
  Fake prob   : 94.3%
─────────────────────────────────────────────
```

---

## 🧠 Machine Learning Details

### Feature Engineering (811-dimensional hybrid vector)

#### Acoustic Features (43 dims) — extracted by Librosa

| Feature | Dimensions | What it captures |
|---|---|---|
| MFCC Mean | 13 | Timbral texture of the voice |
| MFCC Std Dev | 13 | Variability in timbral texture |
| Chroma | 12 | Pitch class energy distribution |
| Spectral Centroid | 1 | Brightness of the sound |
| Spectral Bandwidth | 1 | Width of the frequency band |
| Zero Crossing Rate | 1 | Signal periodicity |
| RMS Energy | 1 | Overall loudness |
| Spectral Roll-off | 1 | High-frequency energy cut-off |

#### Language Features (768 dims) — extracted by DistilBERT

Whisper first transcribes the audio to text. DistilBERT then encodes the text into a 768-dimensional semantic embedding. AI-generated voices often produce characteristic speech patterns that are detectable at the language level.

### Classifier: RandomForestClassifier
- **200 decision trees** averaged for stability
- **StandardScaler** normalization before classification
- **Stratified 80/20** train/test split
- **Balanced class weights** to handle imbalance
- Trained on **12,006 samples** (combined audio + Kaggle CSV)

---

## 🔮 Future Improvements

- [ ] Real-time streaming prediction from microphone
- [ ] Try deep learning models (CNN/LSTM on raw spectrograms)
- [ ] Support ElevenLabs, VALL-E, and newer TTS detection
- [ ] Expand dataset with ASVspoof 2019 (100k+ files)
- [ ] Export detection report as PDF
- [ ] REST API with FastAPI for production use
- [ ] Docker container for easy deployment
- [ ] Confidence calibration (Platt scaling / isotonic regression)

---

## 📚 References

- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [ASVspoof Challenge](https://www.asvspoof.org/)
- [Scikit-learn RandomForest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📜 License

This project is open-source and free to use for educational and research purposes.

---

_Built as an ML portfolio project — demonstrating audio processing, NLP feature engineering, hybrid classification, and interactive web UI skills._
