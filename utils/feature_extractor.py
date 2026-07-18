"""
utils/feature_extractor.py
--------------------------
Audio feature extraction using Librosa.

Features extracted (total = 43 dimensions):
  [0–12]   MFCC means          (13)
  [13–25]  MFCC std-devs       (13)
  [26–37]  Chroma means        (12)
  [38]     Spectral Centroid   (1)
  [39]     Spectral Bandwidth  (1)
  [40]     Zero Crossing Rate  (1)
  [41]     RMS Energy          (1)
  [42]     Spectral Rolloff    (1)

NOTE: Mel spectrogram summary stats have been intentionally removed so that
the feature vector always has exactly FEATURE_SIZE = 43 dimensions, matching
the KAGGLE CSV layout used during training.
"""

import numpy as np
import librosa


# Total number of audio features (must match KAGGLE CSV loader in train_model.py)
FEATURE_SIZE = 43


def extract_features(audio_path: str, sr: int = 22050, duration: float = None) -> np.ndarray:
    """
    Extract audio features from a given audio file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.wav or .mp3)
    sr : int
        Target sample rate (default: 22050 Hz)
    duration : float, optional
        Maximum duration to load (in seconds). None = load full file.

    Returns
    -------
    np.ndarray
        1-D feature vector of shape (FEATURE_SIZE,) = (43,)
    """

    # Load audio (mono, resampled to sr)
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)

    if len(y) == 0:
        raise ValueError(f"Audio file is empty or could not be loaded: {audio_path}")

    # Pre-allocate fixed-size vector (guarantees consistent shape)
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    # ── [0-12] MFCC means ────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features[0:13] = np.mean(mfcc, axis=1)

    # ── [13-25] MFCC std-devs ────────────────────────────────────────────────
    features[13:26] = np.std(mfcc, axis=1)

    # ── [26-37] Chroma means ─────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features[26:38] = np.mean(chroma, axis=1)

    # ── [38] Spectral Centroid ───────────────────────────────────────────────
    features[38] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # ── [39] Spectral Bandwidth ──────────────────────────────────────────────
    features[39] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # ── [40] Zero Crossing Rate ──────────────────────────────────────────────
    features[40] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # ── [41] RMS Energy ──────────────────────────────────────────────────────
    features[41] = float(np.mean(librosa.feature.rms(y=y)))

    # ── [42] Spectral Rolloff ────────────────────────────────────────────────
    features[42] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # Replace NaN / Inf with 0 (safety net)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_feature_names() -> list:
    """Return human-readable names for each audio feature dimension."""
    names = []
    names += [f"MFCC_mean_{i+1}" for i in range(13)]
    names += [f"MFCC_std_{i+1}"  for i in range(13)]
    names += [f"Chroma_{i+1}"    for i in range(12)]
    names += [
        "Spectral_Centroid",
        "Spectral_Bandwidth",
        "Zero_Crossing_Rate",
        "RMS_Energy",
        "Spectral_Rolloff",
    ]
    return names  # len = 43
