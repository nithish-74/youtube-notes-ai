import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.getenv("MODEL_NAME", "google/flan-t5-small")
ON_CLOUD = bool(os.getenv("SPACE_ID"))

_tokenizer = None
_model = None


def get_model():
    """Load the summarization model once (lazy — faster app startup on Render)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        _model.eval()
    return _model, _tokenizer, device


def warmup_model():
    """Load model into memory. Call before summarization so UI can show progress."""
    get_model()