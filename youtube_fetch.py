"""YouTube transcript fetching with retries and yt-dlp fallback for cloud hosts."""

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException, SSLError
from urllib3.util.retry import Retry
from youtube_transcript_api import (
    NoTranscriptFound,
    YouTubeTranscriptApi,
)
from youtube_transcript_api._errors import IpBlocked, RequestBlocked

INDIAN_LANGUAGE_CODES = (
    "hi", "te", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "as", "ur",
)

PREFERRED_LANGS = ["en", *INDIAN_LANGUAGE_CODES]
FETCH_TIMEOUT = int(os.getenv("YOUTUBE_FETCH_TIMEOUT", "25" if os.getenv("SPACE_ID") else "45"))


def _build_session() -> Session:
    session = Session()
    session.headers.update(
        {
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/json,application/xhtml+xml,*/*;q=0.8",
        }
    )
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_api() -> YouTubeTranscriptApi:
    return YouTubeTranscriptApi(http_client=_build_session())


def _retry_call(func, max_attempts=2, base_delay=1):
    last_error = None
    for attempt in range(max_attempts):
        try:
            return func()
        except (SSLError, RequestsConnectionError, TimeoutError, IpBlocked, RequestBlocked) as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(base_delay)
    raise last_error


def _parse_subtitle_payload(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("{"):
        data = json.loads(raw)
        parts = []
        for event in data.get("events", []):
            for seg in event.get("segs", []):
                text = seg.get("utf8", "")
                if text and text != "\n":
                    parts.append(text)
        return " ".join(parts)

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        lines.append(re.sub(r"<[^>]+>", "", line))
    return " ".join(lines)


def _fetch_subtitle_url(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def _get_transcript_via_ytdlp(video_id: str):
    from yt_dlp import YoutubeDL

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 20,
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    def pick_track():
        for lang in PREFERRED_LANGS:
            if lang in manual and manual[lang]:
                return manual[lang][0], lang, manual[lang][0].get("name", lang), False
        for lang in PREFERRED_LANGS:
            if lang in auto and auto[lang]:
                return auto[lang][0], lang, auto[lang][0].get("name", lang), True
        for pool, is_auto in ((manual, False), (auto, True)):
            for lang, tracks in pool.items():
                if tracks:
                    return tracks[0], lang, tracks[0].get("name", lang), is_auto
        return None, None, None, None

    track, lang_code, lang_name, _is_auto = pick_track()
    if not track:
        raise NoTranscriptFound(video_id, ["en"], None)

    sub_url = track.get("url")
    if not sub_url:
        raise NoTranscriptFound(video_id, ["en"], None)

    text = _parse_subtitle_payload(_fetch_subtitle_url(sub_url))
    if not text.strip():
        raise NoTranscriptFound(video_id, ["en"], None)

    metadata = {
        "source_language": lang_name or lang_code,
        "source_code": lang_code,
        "translated": False,
        "needs_local_translation": lang_code != "en",
        "fetch_method": "yt-dlp",
    }
    return text, metadata


def _fetch_transcript_inner(video_id: str):
    on_cloud = bool(os.getenv("SPACE_ID"))

    # On cloud, try yt-dlp first (android client often works when API fails)
    if on_cloud:
        try:
            text, metadata = _get_transcript_via_ytdlp(video_id)
            return "ytdlp", (text, metadata)
        except Exception:
            pass

    api = _get_api()

    def via_api():
        return api.list(video_id)

    try:
        transcript_list = _retry_call(via_api)
        return "api", transcript_list
    except Exception:
        if not on_cloud:
            text, metadata = _get_transcript_via_ytdlp(video_id)
            return "ytdlp", (text, metadata)
        raise


def fetch_transcript_raw(video_id: str):
    """Fetch transcript with a hard timeout so the UI never hangs forever."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch_transcript_inner, video_id)
        try:
            return future.result(timeout=FETCH_TIMEOUT)
        except FuturesTimeout as e:
            raise TimeoutError(
                f"YouTube fetch timed out after {FETCH_TIMEOUT}s. "
                "The cloud server may be blocked by YouTube."
            ) from e


def cloud_fetch_error_message(error: Exception) -> str:
    msg = str(error).lower()
    if "timed out" in msg or "timeout" in msg:
        return (
            "YouTube took too long to respond from the cloud server. "
            "Try a shorter video, retry once, or run locally: `streamlit run app.py`"
        )
    if "ssl" in msg or "eof" in msg or "connection" in msg:
        return (
            "Could not reach YouTube from the cloud server (network/SSL block). "
            "Run locally on your PC for reliable access: `streamlit run app.py`"
        )
    if "blocked" in msg or "429" in msg:
        return (
            "YouTube blocked the request from this server IP. "
            "Run the app locally on your PC."
        )
    return f"Network error while fetching transcript: {error}"


def is_network_error(error: Exception) -> bool:
    if isinstance(error, (SSLError, RequestsConnectionError, TimeoutError, IpBlocked, RequestBlocked)):
        return True
    if isinstance(error, RequestException):
        return True
    msg = str(error).lower()
    return any(k in msg for k in ("ssl", "connection", "timeout", "blocked", "network", "timed out"))
