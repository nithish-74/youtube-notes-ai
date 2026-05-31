import os
import re
import time

from deep_translator import GoogleTranslator

from Model import ON_CLOUD

# YouTube language codes supported by Google Translate fallback
SUPPORTED_LANGUAGES = {
    "te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "as", "ur",
}

# Cloud: translate fewer/larger chunks so Step 1 doesn't run for 5+ minutes
CLOUD_CHUNK_SIZE = 900
CLOUD_MAX_CHUNKS = 12
LOCAL_CHUNK_SIZE = 400


class TranscriptTranslationError(Exception):
    pass


def translate_to_english(text, source_lang_code, chunk_size=None, on_status=None):
    """Translate non-English text to English (fallback when YouTube won't translate)."""
    if source_lang_code == "en":
        return text

    if source_lang_code not in SUPPORTED_LANGUAGES:
        raise TranscriptTranslationError(
            f"Language '{source_lang_code}' is not supported for translation yet."
        )

    if chunk_size is None:
        chunk_size = CLOUD_CHUNK_SIZE if ON_CLOUD else LOCAL_CHUNK_SIZE

    translator = GoogleTranslator(source=source_lang_code, target="en")
    chunks = [c for c in _split_for_translation(text, chunk_size) if c.strip()]

    if ON_CLOUD and len(chunks) > CLOUD_MAX_CHUNKS:
        chunks = chunks[:CLOUD_MAX_CHUNKS]

    translated_chunks = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if on_status:
            on_status(f"Translating part {i + 1} of {total} to English...")
        if i > 0:
            time.sleep(0.3 if ON_CLOUD else 0.5)
        translated_chunks.append(_translate_chunk(translator, chunk))

    return " ".join(translated_chunks)


def _translate_chunk(translator, chunk, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            return translator.translate(chunk)
        except Exception as e:
            last_error = e
            time.sleep(1 + attempt)
    raise TranscriptTranslationError(
        f"Translation failed after {retries} attempts: {last_error}"
    )


def _split_for_translation(text, chunk_size):
    """Split text into chunks at sentence boundaries for translation."""
    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        while len(sentence) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.append(sentence[:chunk_size].strip())
            sentence = sentence[chunk_size:].strip()

        if len(current) + len(sentence) + 1 <= chunk_size:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks
