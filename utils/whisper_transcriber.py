"""
utils/whisper_transcriber.py
-----------------------------
This module uses OpenAI Whisper to transcribe speech from audio files.

Whisper is a state-of-the-art automatic speech recognition (ASR) model.
We use the "base" model which is small enough to run on a CPU in a few seconds.

Model sizes (trade-off: speed vs accuracy):
  tiny   -> fastest, least accurate
  base   -> good balance for demos
  small  -> better accuracy, slower
  medium -> production quality
  large  -> best accuracy, very slow on CPU
"""

import whisper
import os
import tempfile


# Cache the model so it isn't reloaded on every prediction call
_model_cache: dict = {}


def load_whisper_model(model_size: str = "base"):
    """
    Load (and cache) a Whisper model.

    Parameters
    ----------
    model_size : str
        One of: tiny, base, small, medium, large

    Returns
    -------
    whisper.Whisper
        Loaded Whisper model instance
    """
    global _model_cache
    if model_size not in _model_cache:
        print(f"[Whisper] Loading '{model_size}' model (first run may take a moment)...")
        _model_cache[model_size] = whisper.load_model(model_size)
        print(f"[Whisper] Model '{model_size}' loaded successfully.")
    return _model_cache[model_size]


def transcribe_audio(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe an audio file using OpenAI Whisper.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.wav or .mp3)
    model_size : str
        Whisper model size to use (default: "base")

    Returns
    -------
    dict with keys:
        "text"     : str  - Full transcription text
        "language" : str  - Detected language code (e.g. "en")
        "segments" : list - List of timed segments (each with start, end, text)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_whisper_model(model_size)

    # Whisper transcribes the audio and returns a dict
    result = model.transcribe(audio_path, fp16=False)  # fp16=False for CPU compatibility

    return {
        "text":     result.get("text", "").strip(),
        "language": result.get("language", "unknown"),
        "segments": result.get("segments", [])
    }


def transcribe_audio_simple(audio_path: str, model_size: str = "base") -> str:
    """
    Convenience wrapper – returns only the transcription text as a string.

    Parameters
    ----------
    audio_path : str
        Path to the audio file
    model_size : str
        Whisper model size

    Returns
    -------
    str
        Transcribed text
    """
    result = transcribe_audio(audio_path, model_size)
    return result["text"]
