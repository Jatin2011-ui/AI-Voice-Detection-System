"""
utils/bert_embedder.py
----------------------
Extracts BERT embeddings from text (Whisper transcriptions) using
DistilBERT (lightweight, fast, ~67M parameters).

The [CLS] token embedding (768-dim) is used as a fixed-size semantic
fingerprint of the transcribed speech. Combined with audio features,
this gives the model two channels of information:
  1. HOW the voice sounds (audio features)
  2. WHAT the text patterns reveal (BERT embedding)

AI-generated voices often produce transcriptions with subtly different
vocabulary, sentence structure, or punctuation patterns — BERT captures this.

Model: distilbert-base-uncased
  - 40% smaller and 60% faster than BERT-base
  - 97% of BERT's accuracy on GLUE tasks
  - Runs on CPU without any issue
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Lazy-load transformers to avoid startup delay
_tokenizer = None
_model = None

# Fixed BERT embedding size (DistilBERT hidden size)
BERT_EMBEDDING_SIZE = 768


def _load_bert():
    """Lazy-load DistilBERT tokenizer and model."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch

        print("[BERT] Loading DistilBERT model (first run downloads ~250MB)...")
        _tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        _model.eval()  # inference mode
        print("[BERT] DistilBERT loaded successfully.")
    return _tokenizer, _model


def get_bert_embedding(text: str) -> np.ndarray:
    """
    Extract a BERT [CLS] embedding from text.

    Parameters
    ----------
    text : str
        Input text (typically a Whisper transcription)

    Returns
    -------
    np.ndarray
        Shape (768,) float32 embedding vector
    """
    import torch

    tokenizer, model = _load_bert()

    if not text or not text.strip():
        # Return zero vector for empty / silent audio
        return np.zeros(BERT_EMBEDDING_SIZE, dtype=np.float32)

    # Truncate to BERT's max token limit (512)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token is the first token of the last hidden state
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.astype(np.float32)


def get_bert_embedding_safe(text: str) -> np.ndarray:
    """
    Safe wrapper — returns zero vector on any failure.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    np.ndarray
        Shape (768,) float32 — zeros if BERT unavailable or text is empty
    """
    try:
        return get_bert_embedding(text)
    except Exception as e:
        print(f"  [BERT] Warning: embedding failed ({e}), using zeros.")
        return np.zeros(BERT_EMBEDDING_SIZE, dtype=np.float32)
