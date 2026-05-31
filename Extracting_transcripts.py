import os
import re

import torch
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    NotTranslatable,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
)

from Model import ON_CLOUD, get_model, warmup_model
from Translator import TranscriptTranslationError, translate_to_english
from youtube_fetch import (
    cloud_fetch_error_message,
    fetch_transcript_raw,
    is_network_error,
)

VIDEO_ID_PATTERN = re.compile(
    r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([a-zA-Z0-9_-]{11})"
)

INDIAN_LANGUAGE_CODES = (
    "hi", "te", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "as", "ur",
)

MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "8" if ON_CLOUD else "999"))


def extract_video_id(url):
    """Extract video ID from common YouTube URL formats."""
    match = VIDEO_ID_PATTERN.search(url)
    return match.group(1) if match else None


def _fetch_english_transcript(transcript_list):
    """Return English transcript text, translating from another language if needed."""
    try:
        fetched = transcript_list.find_transcript(["en"]).fetch()
        return fetched, {
            "source_language": "English",
            "source_code": "en",
            "translated": False,
        }
    except NoTranscriptFound:
        pass

    transcripts = list(transcript_list)
    if not transcripts:
        raise NoTranscriptFound(transcript_list.video_id, ["en"], transcript_list)

    def sort_key(transcript):
        # Prefer manual captions, then common Indian languages, then anything else.
        indian_priority = (
            INDIAN_LANGUAGE_CODES.index(transcript.language_code)
            if transcript.language_code in INDIAN_LANGUAGE_CODES
            else len(INDIAN_LANGUAGE_CODES)
        )
        return (transcript.is_generated, indian_priority, transcript.language_code)

    for transcript in sorted(transcripts, key=sort_key):
        if not transcript.is_translatable:
            continue
        try:
            fetched = transcript.translate("en").fetch()
            return fetched, {
                "source_language": transcript.language,
                "source_code": transcript.language_code,
                "translated": True,
            }
        except (NotTranslatable, TranslationLanguageNotAvailable):
            continue

    # YouTube won't translate some auto-generated captions (e.g. Telugu).
    # Fall back to local AI translation.
    source = sorted(transcripts, key=sort_key)[0]
    fetched = source.fetch()
    return fetched, {
        "source_language": source.language,
        "source_code": source.language_code,
        "translated": False,
        "needs_local_translation": True,
    }


def get_transcript(video_id, on_status=None):
    """
    Fetch transcript in English.

    Returns:
        (text, metadata) on success — metadata includes source language info
        str error message on failure
    """
    def status(msg):
        if on_status:
            on_status(msg)

    try:
        fetch_result = fetch_transcript_raw(video_id)

        if fetch_result[0] == "ytdlp":
            text, metadata = fetch_result[1]
        else:
            transcript_list = fetch_result[1]
            fetched, metadata = _fetch_english_transcript(transcript_list)
            text = " ".join(t.text for t in fetched)

        if metadata.get("needs_local_translation"):
            # Translation runs in process_video so UI can show its own step
            return text, metadata

        return text, metadata
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except TranscriptTranslationError as e:
        return f"Error: {e}"
    except NoTranscriptFound:
        return (
            "Error: No English transcript found. "
            "Make sure the video has captions/subtitles enabled "
            "(Telugu, Hindi, etc. are supported and will be translated to English)."
        )
    except TimeoutError as e:
        return f"Error: {cloud_fetch_error_message(e)}"
    except (ConnectionError, TimeoutError) as e:
        return f"Error: {cloud_fetch_error_message(e)}"
    except Exception as e:
        if is_network_error(e):
            return f"Error: {cloud_fetch_error_message(e)}"
        raise


def chunk_text(text, chunk_size=800):
    """Split text into chunks by sentences, hard-splitting oversized sentences."""
    if not text or not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current_chunk = [], ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        while len(sentence) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence[:chunk_size].strip())
            sentence = sentence[chunk_size:].strip()

        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def summarize_chunk(text_chunk):
    """Summarize a single chunk of transcript text in English."""
    model, tokenizer, device = get_model()
    prompt = (
        "Summarize the following text in clear English. "
        "Use concise bullet-style notes:\n"
        f"{text_chunk}"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    # Greedy decode on CPU/cloud — much faster than beam search
    num_beams = 1 if device == "cpu" or ON_CLOUD else 4

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=num_beams,
            length_penalty=1.0,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def process_video(video_url, on_status=None, on_progress=None):
    """
    Process a YouTube video and return structured results.

    Returns:
        dict with keys: success, error (optional), video_id, transcript_meta,
        section_notes, overall_summary, chunk_count
    """
    def status(msg):
        if on_status:
            on_status(msg)

    def progress(pct):
        if on_progress:
            on_progress(pct)

    video_id = extract_video_id(video_url)
    if not video_id:
        return {"success": False, "error": "Invalid YouTube URL. Please check the link and try again."}

    status("Step 1/4: Fetching transcript from YouTube...")
    progress(5)
    transcript_result = get_transcript(video_id, on_status=on_status)

    if isinstance(transcript_result, str):
        return {"success": False, "error": transcript_result.replace("Error: ", "")}

    transcript, transcript_meta = transcript_result
    progress(20)

    if transcript_meta.get("needs_local_translation"):
        status(
            f"Step 2/4: Translating {transcript_meta['source_language']} to English..."
        )
        progress(25)
        try:
            transcript = translate_to_english(
                transcript,
                transcript_meta["source_code"],
                on_status=status,
            )
            transcript_meta["translated"] = True
            transcript_meta["translation_method"] = "local"
            if ON_CLOUD:
                transcript_meta["translation_note"] = (
                    "Cloud mode: translated the opening portion of the video for speed."
                )
        except TranscriptTranslationError as e:
            return {"success": False, "error": str(e)}
        progress(40)

    if not transcript.strip():
        return {"success": False, "error": "Transcript is empty for this video."}

    status("Step 3/4: Loading AI model (first time may take 1–2 min)...")
    progress(45)
    warmup_model()
    progress(60)

    status("Step 4/4: Generating notes...")
    chunks = chunk_text(transcript)
    truncated = False
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
        truncated = True

    notes = []
    for i, chunk in enumerate(chunks):
        status(f"Summarizing section {i + 1} of {len(chunks)}...")
        progress(55 + int(40 * (i + 1) / max(len(chunks), 1)))
        notes.append(summarize_chunk(chunk))

    overall_summary = None
    if len(notes) > 1:
        status("Creating overall summary...")
        combined = "\n".join(notes)
        overall_summary = summarize_chunk(combined)

    progress(100)
    result = {
        "success": True,
        "video_id": video_id,
        "video_url": video_url,
        "transcript_meta": transcript_meta,
        "section_notes": notes,
        "overall_summary": overall_summary,
        "chunk_count": len(chunks),
    }
    if truncated:
        result["warning"] = (
            f"Video was long — only the first {MAX_CHUNKS} sections were summarized "
            "on cloud hosting for speed."
        )
    if transcript_meta.get("translation_note"):
        note = transcript_meta["translation_note"]
        result["warning"] = f"{result.get('warning', '')} {note}".strip()
    return result


def generate_video_notes(video_url):
    print(f"\n[*] Processing video: {video_url}")

    result = process_video(video_url, on_status=lambda msg: print(f"    {msg}"))

    if not result["success"]:
        print(f"Error: {result['error']}")
        return

    meta = result["transcript_meta"]
    if meta["translated"]:
        method = meta.get("translation_method", "youtube")
        method_label = "Google Translate" if method == "local" else "YouTube"
        print(
            f"    Source: {meta['source_language']} ({meta['source_code']}) "
            f"-> translated to English via {method_label}"
        )
    else:
        print(f"    Source: {meta['source_language']}")

    print(f"    -> {result['chunk_count']} chunks created.")

    print("\n" + "=" * 50)
    print("AI GENERATED NOTES (by section)")
    print("=" * 50)
    print("\n".join(f"- {note}" for note in result["section_notes"]))

    if result["overall_summary"]:
        print("\n" + "=" * 50)
        print("OVERALL SUMMARY")
        print("=" * 50)
        print(result["overall_summary"])


if __name__ == "__main__":
    url = input("Paste YouTube URL: ")
    generate_video_notes(url)
