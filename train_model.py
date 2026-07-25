"""
train_model.py
--------------
Full training pipeline for the AI Voice Authenticity Detector.

Architecture — Hybrid Audio + Language Model:
  1. Whisper (base)  → transcribes each audio file to text
  2. DistilBERT      → converts transcription to a 768-dim semantic embedding
  3. Librosa         → extracts 43 handcrafted audio features
  4. Concatenation   → [43 audio features | 768 BERT features] = 811-dim vector
  5. RandomForest    → trained on the 811-dim vectors

Why BERT?
  AI-generated voices often produce transcriptions with subtly different
  vocabulary, fluency patterns, filler-word distribution, or sentence
  structure. BERT encodes these patterns into a dense vector that
  RandomForest can exploit alongside acoustic features.

Data sources discovered automatically:
  dataset/Real/**/*.wav   → label 0 (Real)
  dataset/Fake/**/*.wav   → label 1 (Fake)
  dataset/KAGGLE/AUDIO/REAL/*.wav  → label 0
  dataset/KAGGLE/AUDIO/FAKE/*.wav  → label 1
  dataset/KAGGLE/DATASET-balanced.csv  (audio-only rows, BERT = zeros)

HOW TO RUN:
  python train_model.py

The script is self-healing:
  - Missing folders are skipped with a warning (no crash)
  - Unreadable audio files are skipped individually
  - Files that fail Whisper transcription use zero BERT vectors
"""

import os
import sys

# ── Fix Windows console Unicode/emoji encoding ─────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.feature_extractor import extract_features, FEATURE_SIZE
from utils.bert_embedder import get_bert_embedding_safe, BERT_EMBEDDING_SIZE
from utils.whisper_transcriber import transcribe_audio

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "voice_detector.pkl")

# Combined feature size: audio (43) + BERT (768)
COMBINED_FEATURE_SIZE = FEATURE_SIZE + BERT_EMBEDDING_SIZE   # 811

# Max audio duration to process per file (seconds)
MAX_DURATION = 60

# Supported audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Keywords that classify folders OR filenames as Real or Fake
REAL_KEYWORDS = {"real", "genuine", "human", "original"}
FAKE_KEYWORDS = {"fake", "ai", "generated", "spoof", "synthetic", "deepfake", "clone"}


# ══════════════════════════════════════════════════════════════════════════════
# Folder label inference
# ══════════════════════════════════════════════════════════════════════════════

def folder_label(folder_name: str):
    """Return 0 (Real), 1 (Fake), or None based on folder name keywords."""
    name_lower = folder_name.lower()
    for kw in REAL_KEYWORDS:
        if kw in name_lower:
            return 0
    for kw in FAKE_KEYWORDS:
        if kw in name_lower:
            return 1
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Discover all labeled audio files
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_audio_files() -> pd.DataFrame:
    """
    Recursively scan dataset/ and collect audio files with inferred labels.
    Skips DEMONSTRATION/ folder (no ground-truth labels).

    Returns DataFrame with columns: [filepath, label, class, source]
    """
    records = []
    seen    = set()

    print("\n[SCAN] Recursively scanning dataset/ for audio files...")

    for root, dirs, files in os.walk(DATASET_DIR):
        rel_root = os.path.relpath(root, DATASET_DIR)

        # Skip demonstration folder
        if "demonstration" in rel_root.lower():
            continue

        # Infer label from folder path components
        path_parts = rel_root.replace("\\", "/").split("/")
        label = None
        for part in reversed(path_parts):
            lbl = folder_label(part)
            if lbl is not None:
                label = lbl
                break

        if label is None:
            continue

        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTS:
                continue

            fpath = os.path.join(root, fname)
            if fpath in seen:
                continue
            seen.add(fpath)

            # ── Filename-stem fallback ────────────────────────────────────────
            # For datasets like real_or_fake_voice/ where files are named
            # original.wav / synthetic_1.mp3 inside numbered folders that
            # carry no recognizable label keyword.
            file_label = label
            if file_label is None:
                stem = os.path.splitext(fname)[0].lower()
                file_label = folder_label(stem)   # reuses keyword check

            if file_label is None:
                continue   # still no label → skip

            source = "KAGGLE_AUDIO" if "kaggle" in rel_root.lower() else "ORIGINAL"

            records.append({
                "filepath": fpath,
                "label":    file_label,
                "class":    "Real" if file_label == 0 else "Fake",
                "source":   source,
            })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["filepath", "label", "class", "source"]
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# KAGGLE pre-extracted CSV (audio features only, BERT features = zeros)
# ══════════════════════════════════════════════════════════════════════════════

KAGGLE_CSV_PATH = os.path.join(DATASET_DIR, "KAGGLE", "DATASET-balanced.csv")


def load_kaggle_csv_features() -> tuple:
    """
    Load KAGGLE CSV and convert to (X, y) arrays compatible with our schema.
    BERT features are padded with zeros (no text available from CSV).

    Returns (X, y) or (None, None) if CSV not found.
    """
    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"  [SKIP] KAGGLE CSV not found: {KAGGLE_CSV_PATH}")
        return None, None

    print(f"\n[STEP] Loading KAGGLE pre-extracted features from CSV...")
    df = pd.read_csv(KAGGLE_CSV_PATH)

    label_map = {"REAL": 0, "FAKE": 1, "real": 0, "fake": 1}
    y_raw = df["LABEL"].map(label_map).values

    # Filter out unmapped labels
    valid_mask = ~pd.isna(y_raw)
    df = df[valid_mask].reset_index(drop=True)
    y = y_raw[valid_mask].astype(np.int32)

    n = len(df)
    # Full combined feature vector (audio + BERT zeros)
    X = np.zeros((n, COMBINED_FEATURE_SIZE), dtype=np.float32)

    # ── MFCC means [0-12] ────────────────────────────────────────────────────
    for i in range(13):
        col = f"mfcc{i+1}"
        if col in df.columns:
            X[:, i] = df[col].fillna(0).values

    # ── MFCC std [13-25] → not in CSV, stay zero ─────────────────────────────

    # ── Chroma[0] mean [26] ──────────────────────────────────────────────────
    if "chroma_stft" in df.columns:
        X[:, 26] = df["chroma_stft"].fillna(0).values

    # ── Spectral Centroid [38] ───────────────────────────────────────────────
    if "spectral_centroid" in df.columns:
        X[:, 38] = df["spectral_centroid"].fillna(0).values

    # ── Spectral Bandwidth [39] ──────────────────────────────────────────────
    if "spectral_bandwidth" in df.columns:
        X[:, 39] = df["spectral_bandwidth"].fillna(0).values

    # ── Zero Crossing Rate [40] ──────────────────────────────────────────────
    if "zero_crossing_rate" in df.columns:
        X[:, 40] = df["zero_crossing_rate"].fillna(0).values

    # ── RMS Energy [41] ──────────────────────────────────────────────────────
    if "rms" in df.columns:
        X[:, 41] = df["rms"].fillna(0).values

    # ── Spectral Rolloff [42] ────────────────────────────────────────────────
    if "rolloff" in df.columns:
        X[:, 42] = df["rolloff"].fillna(0).values

    # BERT features [43:811] → zeros (no transcription available)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    real_count = int(np.sum(y == 0))
    fake_count = int(np.sum(y == 1))
    print(f"  [OK]  KAGGLE CSV: {real_count} Real + {fake_count} Fake = {len(X)} samples "
          f"(BERT features = zeros for CSV rows)")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Transcribe + embed text features
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_and_embed(audio_path: str) -> np.ndarray:
    """
    Transcribe audio with Whisper and extract BERT embedding.

    Returns np.ndarray of shape (BERT_EMBEDDING_SIZE,)
    """
    try:
        result = transcribe_audio(audio_path, model_size="base")
        text = result.get("text", "").strip()
        return get_bert_embedding_safe(text)
    except Exception as e:
        print(f"\n  [BERT] Whisper/BERT failed for {os.path.basename(audio_path)}: {e}")
        return np.zeros(BERT_EMBEDDING_SIZE, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Extract all features (audio + BERT) from audio files
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(df: pd.DataFrame) -> tuple:
    """
    For each audio file in df:
      1. Extract 43-dim Librosa audio features
      2. Transcribe with Whisper → extract 768-dim BERT embedding
      3. Concatenate → 811-dim feature vector

    Returns (X, y)
    """
    X_list, y_list, skipped = [], [], []
    total = len(df)

    if total == 0:
        return (
            np.empty((0, COMBINED_FEATURE_SIZE), dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    print(f"\n[STEP] Extracting Audio + BERT features from {total} files...")
    print("  (Whisper transcribes each file, then DistilBERT encodes the text)\n")

    for i, (_, row) in enumerate(df.iterrows()):
        filepath = row["filepath"]
        label    = row["label"]
        filename = os.path.basename(filepath)
        cls_name = row["class"]

        try:
            t0 = time.time()

            # Audio features (43-dim)
            audio_feat = extract_features(filepath, duration=MAX_DURATION)

            # BERT features (768-dim) via Whisper transcription
            bert_feat = transcribe_and_embed(filepath)

            # Concatenate
            combined = np.concatenate([audio_feat, bert_feat])  # (811,)
            X_list.append(combined)
            y_list.append(label)

            elapsed = time.time() - t0
            done = i + 1
            bar  = "#" * int(done / total * 25) + "-" * (25 - int(done / total * 25))
            print(f"  [{bar}] {done:>3}/{total}  [{cls_name}] {filename:<35} ({elapsed:.1f}s)",
                  end="\r")

        except Exception as e:
            print(f"\n  [SKIP] {filename}: {e}")
            skipped.append(filepath)

    print(f"\n\n[DONE] Extraction done. Processed: {len(X_list)}, Skipped: {len(skipped)}")

    if not X_list:
        return (
            np.empty((0, COMBINED_FEATURE_SIZE), dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Train / Test Split
# ══════════════════════════════════════════════════════════════════════════════

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split with fallback for tiny datasets."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    real_train = int(np.sum(y_train == 0))
    fake_train = int(np.sum(y_train == 1))
    real_test  = int(np.sum(y_test  == 0))
    fake_test  = int(np.sum(y_test  == 1))

    print(f"\n[INFO] Train: {len(X_train)} samples  (Real: {real_train}, Fake: {fake_train})")
    print(f"[INFO] Test : {len(X_test)} samples  (Real: {real_test}, Fake: {fake_test})\n")

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# Model Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, n_samples: int) -> Pipeline:
    """
    Build and train a Scikit-learn Pipeline.
    Uses RandomForest which handles high-dimensional sparse features well.

    Parameters scale automatically with dataset size:
      < 200 samples  → 200 trees
      200–5000       → 400 trees
      > 5000         → 500 trees, max_depth=25
    """
    print("[STEP] Training RandomForestClassifier on hybrid Audio+BERT features...")
    print(f"   Feature dimensions: {X_train.shape[1]} "
          f"({FEATURE_SIZE} audio + {BERT_EMBEDDING_SIZE} BERT)\n")

    if n_samples < 200:
        n_est, max_d = 200, None
    elif n_samples < 5000:
        n_est, max_d = 400, None
    else:
        n_est, max_d = 500, 25

    print(f"   n_estimators={n_est}, max_depth={max_d}, n_jobs=-1 (all cores)\n")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    print("[DONE] Training complete!\n")
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Evaluate on the held-out test set and print full metrics."""
    y_pred = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("=" * 55)
    print("         MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1 Score  : {f1  * 100:.2f}%")
    print()
    print("  Confusion Matrix:")
    print(f"             Predicted Real  Predicted Fake")
    print(f"  Actual Real  [{cm[0][0]:^14}] [{cm[0][1]:^14}]")
    print(f"  Actual Fake  [{cm[1][0]:^14}] [{cm[1][1]:^14}]")
    print()
    print("  Full Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Real (0)", "Fake (1)"]))
    print("=" * 55)

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1_score":         f1,
        "confusion_matrix": cm.tolist(),
        "feature_dims":     int(COMBINED_FEATURE_SIZE),
        "audio_dims":       int(FEATURE_SIZE),
        "bert_dims":        int(BERT_EMBEDDING_SIZE),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Save Model
# ══════════════════════════════════════════════════════════════════════════════

def save_model(pipeline, metrics: dict, dataset_info: dict) -> None:
    """Save the trained pipeline and metrics to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    print(f"\n[SAVED] Model  → {MODEL_PATH}")

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    metrics["dataset_info"] = dataset_info
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] Metrics → {metrics_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  AI Voice Authenticity Detector — Training Pipeline")
    print("  Architecture: Whisper STT + DistilBERT + Audio Features")
    print("=" * 60)

    all_X, all_y = [], []
    dataset_info = {}

    # ── SOURCE A: Raw audio files ──────────────────────────────────────────────
    audio_df = collect_all_audio_files()

    if len(audio_df) > 0:
        real_count = int((audio_df["label"] == 0).sum())
        fake_count = int((audio_df["label"] == 1).sum())
        src_counts = audio_df["source"].value_counts().to_dict()
        print(f"\n[INFO] Audio files found:")
        print(f"   Real : {real_count}")
        print(f"   Fake : {fake_count}")
        print(f"   Total: {len(audio_df)}")
        for src, cnt in src_counts.items():
            print(f"   Source [{src}]: {cnt} files")

        dataset_info["audio_files"] = {
            "total": len(audio_df), "real": real_count, "fake": fake_count
        }

        X_audio, y_audio = extract_all_features(audio_df)

        if len(X_audio) > 0:
            all_X.append(X_audio)
            all_y.append(y_audio)
            print(f"[INFO] Audio+BERT feature matrix shape: {X_audio.shape}")
    else:
        print("\n[WARN] No labeled audio files found in dataset/. Will use CSV only.")
        dataset_info["audio_files"] = {"total": 0, "real": 0, "fake": 0}

    # ── SOURCE B: KAGGLE pre-extracted CSV ────────────────────────────────────
    X_csv, y_csv = load_kaggle_csv_features()

    if X_csv is not None and len(X_csv) > 0:
        all_X.append(X_csv)
        all_y.append(y_csv)
        dataset_info["kaggle_csv"] = {
            "total": len(X_csv),
            "real":  int(np.sum(y_csv == 0)),
            "fake":  int(np.sum(y_csv == 1)),
        }
        print(f"[INFO] KAGGLE CSV feature matrix shape: {X_csv.shape}")
    else:
        dataset_info["kaggle_csv"] = {"total": 0}

    # ── Combine all sources ────────────────────────────────────────────────────
    if not all_X:
        print("\n[ERROR] No training data found at all. Exiting.")
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    total_real = int(np.sum(y == 0))
    total_fake = int(np.sum(y == 1))
    dataset_info["combined"] = {
        "total": len(X), "real": total_real, "fake": total_fake
    }

    print("\n" + "=" * 60)
    print("  COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total samples  : {len(X):,}")
    print(f"  Real (label=0) : {total_real:,}")
    print(f"  Fake (label=1) : {total_fake:,}")
    print(f"  Feature dims   : {X.shape[1]}  ({FEATURE_SIZE} audio + {BERT_EMBEDDING_SIZE} BERT)")
    print("=" * 60)

    if len(X) < 4:
        print("\n[ERROR] Not enough samples to train. Need at least 4.")
        sys.exit(1)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    pipeline = train_model(X_train, y_train, n_samples=len(X_train))

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Save
    save_model(pipeline, metrics, dataset_info)

    print("\n[DONE] All done! Run the app with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()
