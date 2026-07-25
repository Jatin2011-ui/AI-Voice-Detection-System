"""
predict.py
----------
Standalone prediction module for the AI Voice Authenticity Detector.

Pipeline (matches training exactly):
  1. Load audio
  2. Extract 43-dim Librosa audio features
  3. Transcribe with Whisper (base) → text
  4. Extract 768-dim DistilBERT embedding from transcription
  5. Concatenate → 811-dim feature vector
  6. Load trained RandomForest pipeline
  7. Predict class + confidence

Usage:
  a) Imported by app.py: from predict import predict_audio, load_model, MODEL_PATH
  b) CLI:                python predict.py path/to/audio.wav
"""

import os
import sys
import numpy as np
import joblib

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.feature_extractor import extract_features, FEATURE_SIZE
from utils.bert_embedder import get_bert_embedding_safe, BERT_EMBEDDING_SIZE
from utils.whisper_transcriber import transcribe_audio

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "voice_detector.pkl")

# ── Label mapping ──────────────────────────────────────────────────────────────
LABEL_MAP = {
    0: "Real Human Voice",
    1: "AI Generated Voice"
}


def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained scikit-learn pipeline from disk.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Please run 'python train_model.py' first to train the model."
        )
    pipeline = joblib.load(model_path)
    return pipeline


def predict_audio(audio_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Run the full hybrid prediction pipeline on a single audio file.

    Steps:
      1. Extract 43-dim Librosa audio features
      2. Transcribe with Whisper (base) → text
      3. Encode text with DistilBERT → 768-dim vector
      4. Concatenate → 811-dim feature vector
      5. Predict with trained model

    Parameters
    ----------
    audio_path : str
        Path to the .wav or .mp3 file to analyze
    model_path : str
        Path to the trained model .pkl file

    Returns
    -------
    dict with keys:
        "label"       : str   - "Real Human Voice" or "AI Generated Voice"
        "class_id"    : int   - 0 (Real) or 1 (Fake)
        "confidence"  : float - confidence percentage (0–100)
        "proba_real"  : float - probability of being Real (0–1)
        "proba_fake"  : float - probability of being Fake (0–1)
        "is_fake"     : bool  - True if predicted AI-generated
        "transcription": str  - Whisper transcription of the audio
    """

    # ── Validate input ─────────────────────────────────────────────────────────
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".wav", ".mp3", ".flac", ".ogg"):
        raise ValueError(f"Unsupported audio format: {ext}. Use WAV or MP3.")

    # ── Step 1: Audio features (43-dim) ───────────────────────────────────────
    audio_features = extract_features(audio_path, duration=60)  # (43,)

    # ── Step 2: Whisper transcription ─────────────────────────────────────────
    transcription = ""
    try:
        result = transcribe_audio(audio_path, model_size="base")
        transcription = result.get("text", "").strip()
    except Exception:
        transcription = ""  # silent file or Whisper unavailable

    # ── Step 3: BERT embedding (768-dim) ──────────────────────────────────────
    bert_features = get_bert_embedding_safe(transcription)  # (768,)

    # ── Step 4: Concatenate → 811-dim ─────────────────────────────────────────
    X = np.concatenate([audio_features, bert_features]).reshape(1, -1)

    # ── Step 5: Load model + predict ──────────────────────────────────────────
    pipeline = load_model(model_path)

    class_id   = int(pipeline.predict(X)[0])
    proba      = pipeline.predict_proba(X)[0]
    proba_real = float(proba[0])
    proba_fake = float(proba[1])
    confidence = max(proba_real, proba_fake) * 100

    label   = LABEL_MAP[class_id]
    is_fake = (class_id == 1)

    return {
        "label":         label,
        "class_id":      class_id,
        "confidence":    confidence,
        "proba_real":    proba_real,
        "proba_fake":    proba_fake,
        "is_fake":       is_fake,
        "transcription": transcription,
    }


# ── CLI usage ──────────────────────────────────────────────────────────────────

def main():
    """Allow running predict.py from the command line."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        print("Example: python predict.py sample.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"\n  Analyzing: {os.path.basename(audio_path)}")
    print("-" * 45)

    try:
        result = predict_audio(audio_path)
        icon = "AI" if result["is_fake"] else "OK"
        print(f"  Prediction   : [{icon}] {result['label']}")
        print(f"  Confidence   : {result['confidence']:.1f}%")
        print(f"  Real prob    : {result['proba_real'] * 100:.1f}%")
        print(f"  Fake prob    : {result['proba_fake'] * 100:.1f}%")
        print(f"  Transcription: {result['transcription'][:120] or '[silent]'}")
        print("-" * 45)
    except FileNotFoundError as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nPrediction error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
