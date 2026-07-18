"""
utils/visualization.py
-----------------------
This module provides audio visualization functions using Matplotlib.

Functions:
  - plot_waveform()     : Display the time-domain amplitude plot
  - plot_spectrogram()  : Display the frequency-domain spectrogram
  - plot_mfcc()         : Display MFCC heatmap
  - plot_confidence()   : Display a confidence gauge bar

All functions return Matplotlib Figure objects so Streamlit can
render them with `st.pyplot(fig)`.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (required for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch


# ── Color palette ─────────────────────────────────────────────────────────────
REAL_COLOR  = "#00C896"   # teal-green  → real/human
FAKE_COLOR  = "#FF4B6E"   # coral-red   → AI generated
BG_COLOR    = "#0E1117"   # dark background matching Streamlit dark theme
TEXT_COLOR  = "#FAFAFA"


def _dark_fig(figsize=(10, 3)):
    """Create a dark-themed Matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    return fig, ax


def plot_waveform(audio_path: str, label: str = "Audio") -> plt.Figure:
    """
    Plot the waveform (amplitude vs. time) of an audio file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file
    label : str
        Title label for the plot

    Returns
    -------
    matplotlib.figure.Figure
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)
    times = np.linspace(0, len(y) / sr, num=len(y))

    fig, ax = _dark_fig(figsize=(10, 3))

    # Gradient-colored waveform using a Line2D trick
    ax.fill_between(times, y, alpha=0.6, color=REAL_COLOR)
    ax.plot(times, y, color=REAL_COLOR, linewidth=0.5, alpha=0.9)

    ax.set_title(f"🔊 Waveform — {label}", fontsize=12, pad=10, color=TEXT_COLOR)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_xlim(0, times[-1])
    ax.axhline(0, color="#555555", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    return fig


def plot_spectrogram(audio_path: str, label: str = "Audio") -> plt.Figure:
    """
    Plot a Mel-frequency spectrogram of an audio file.

    The spectrogram shows how the frequency content changes over time.
    AI voices often show unusually uniform or periodic patterns here.

    Parameters
    ----------
    audio_path : str
        Path to the audio file
    label : str
        Title label

    Returns
    -------
    matplotlib.figure.Figure
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)

    # Compute mel spectrogram and convert to dB scale
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = _dark_fig(figsize=(10, 4))

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=8000,
        ax=ax,
        cmap="magma"     # dark, high-contrast colormap
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="dB")
    ax.set_title(f"🎵 Mel Spectrogram — {label}", fontsize=12, pad=10, color=TEXT_COLOR)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)

    fig.tight_layout()
    return fig


def plot_mfcc(audio_path: str, label: str = "Audio") -> plt.Figure:
    """
    Plot the MFCC heatmap of an audio file.

    MFCCs are a compact representation of the power spectrum of a sound.
    They're among the most important features for voice authenticity detection.

    Parameters
    ----------
    audio_path : str
        Path to the audio file
    label : str
        Title label

    Returns
    -------
    matplotlib.figure.Figure
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    fig, ax = _dark_fig(figsize=(10, 4))

    img = librosa.display.specshow(
        mfcc,
        sr=sr,
        x_axis="time",
        ax=ax,
        cmap="coolwarm"
    )

    fig.colorbar(img, ax=ax, label="MFCC Coefficient Value")
    ax.set_title(f"📊 MFCC Features — {label}", fontsize=12, pad=10, color=TEXT_COLOR)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("MFCC Coefficient", fontsize=9)

    fig.tight_layout()
    return fig


def plot_confidence_meter(confidence: float, label: str, is_fake: bool) -> plt.Figure:
    """
    Plot a visual confidence meter (horizontal bar + annotation).

    Parameters
    ----------
    confidence : float
        Confidence percentage (0–100)
    label : str
        Prediction label ("Real Human Voice" or "AI Generated Voice")
    is_fake : bool
        True if the prediction is AI-generated

    Returns
    -------
    matplotlib.figure.Figure
    """
    color = FAKE_COLOR if is_fake else REAL_COLOR
    icon  = "🤖" if is_fake else "✅"

    fig, ax = plt.subplots(figsize=(8, 1.8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background bar
    ax.barh(0.5, 100, height=0.35, color="#222222", left=0, align="center",
            linewidth=0, zorder=1)

    # Foreground confidence bar
    ax.barh(0.5, confidence, height=0.35, color=color, left=0, align="center",
            linewidth=0, zorder=2, alpha=0.9)

    # Text label
    ax.text(
        50, 0.88,
        f"{icon} {label}  |  Confidence: {confidence:.1f}%",
        ha="center", va="center",
        color=TEXT_COLOR, fontsize=11, fontweight="bold"
    )

    # Confidence number on bar
    ax.text(
        min(confidence, 97), 0.5,
        f"{confidence:.1f}%",
        ha="right", va="center",
        color="white", fontsize=9, fontweight="bold", zorder=3
    )

    fig.tight_layout(pad=0.2)
    return fig
