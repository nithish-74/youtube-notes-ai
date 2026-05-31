"""
Friendly web UI for YouTube video summarization.
Run: streamlit run app.py
"""

import os

import streamlit as st

ON_CLOUD = bool(os.getenv("SPACE_ID"))

st.set_page_config(
    page_title="YouTube Notes AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0f0f14 0%, #1a1a2e 45%, #16213e 100%);
    }

    h1, h2, h3, p, label, .stMarkdown {
        font-family: 'DM Sans', sans-serif !important;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .hero-sub {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }

    .lang-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #c7d2fe;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid rgba(129, 140, 248, 0.4);
    }

    .note-card {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #6366f1;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        color: #e2e8f0;
        line-height: 1.6;
    }

    .summary-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.1));
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 16px;
        padding: 1.5rem;
        color: #f1f5f9;
        line-height: 1.7;
        font-size: 1.05rem;
    }

    div[data-testid="stSidebar"] {
        background: rgba(15, 15, 20, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.06) !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        width: 100%;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_language_badge(meta):
    if meta.get("translated"):
        method = meta.get("translation_method", "youtube")
        via = "Google Translate" if method == "local" else "YouTube"
        return (
            f"{meta['source_language']} ({meta['source_code']}) "
            f"→ English via {via}"
        )
    return meta.get("source_language", "English")


def render_sidebar():
    with st.sidebar:
        st.markdown("### How it works")
        st.markdown(
            """
            1. Paste any YouTube link  
            2. We fetch captions (any language)  
            3. Translate to English if needed  
            4. AI writes clear notes for you  
            """
        )
        st.divider()
        st.markdown("### Supported languages")
        st.caption(
            "English, Hindi, Telugu, Tamil, Kannada, Malayalam, "
            "Marathi, Bengali, Gujarati, Punjabi, and more."
        )
        st.divider()
        st.caption(
            "Runs on the free **flan-t5-small** model. "
            "Set env `MODEL_NAME=google/flan-t5-base` locally for higher quality."
        )
        st.divider()
        st.markdown("### Tips")
        st.info(
            "Cloud hosting can be slow or blocked by YouTube. "
            "First run loads the AI model (~1–2 min). "
            "For best results, run locally: `streamlit run app.py`",
            icon="💡",
        )


def render_results(result):
    meta = result["transcript_meta"]
    st.markdown(
        f'<span class="lang-badge">{format_language_badge(meta)}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Processed {result['chunk_count']} sections")

    if result.get("overall_summary"):
        st.markdown("#### Overall summary")
        st.markdown(
            f'<div class="summary-box">{result["overall_summary"]}</div>',
            unsafe_allow_html=True,
        )
        st.divider()

    st.markdown("#### Section notes")
    for i, note in enumerate(result["section_notes"], start=1):
        st.markdown(
            f'<div class="note-card"><strong>Part {i}</strong><br>{note}</div>',
            unsafe_allow_html=True,
        )

    full_text = ""
    if result.get("overall_summary"):
        full_text += "## Overall Summary\n\n" + result["overall_summary"] + "\n\n"
    full_text += "## Section Notes\n\n"
    for i, note in enumerate(result["section_notes"], start=1):
        full_text += f"### Part {i}\n{note}\n\n"

    st.download_button(
        label="Download notes",
        data=full_text.strip(),
        file_name=f"youtube_notes_{result['video_id']}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def main():
    render_sidebar()

    if ON_CLOUD:
        st.warning(
            "**Cloud demo mode** — YouTube + translation can be slow or fail on free hosting. "
            "For Telugu/Hindi videos, **run locally** for full speed and reliability: "
            "`streamlit run app.py`",
            icon="⚠️",
        )

    col_main, _ = st.columns([2, 1])
    with col_main:
        st.markdown('<p class="hero-title">YouTube Notes AI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">Turn any YouTube video into clear English notes — '
            "Hindi, Telugu, and more.</p>",
            unsafe_allow_html=True,
        )

    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=... or youtu.be/...",
        label_visibility="collapsed",
    )

    summarize = st.button("Generate notes", type="primary", use_container_width=True)

    if summarize:
        if not url or not url.strip():
            st.error("Please paste a YouTube URL first.")
            return

        status_box = st.empty()
        progress = st.progress(0, text="Starting...")
        log_box = st.empty()

        def on_status(msg):
            status_box.info(msg, icon="⏳")
            log_box.caption(msg)

        try:
            from Extracting_transcripts import process_video

            result = process_video(
                url.strip(),
                on_status=on_status,
                on_progress=lambda pct: progress.progress(
                    min(pct, 100) / 100,
                    text="Processing...",
                ),
            )
            progress.progress(1.0, text="Done!")

        except Exception as e:
            from youtube_fetch import cloud_fetch_error_message, is_network_error

            if is_network_error(e):
                st.error(cloud_fetch_error_message(e))
            else:
                st.error(f"Something went wrong: {e}")
            st.info(
                "Tip: YouTube often blocks free cloud servers. "
                "If this keeps failing, run locally with `streamlit run app.py` — it works on your PC."
            )
            return

        status_box.empty()
        progress.empty()
        log_box.empty()

        if not result["success"]:
            st.error(result.get("error", "Unknown error"))
            return

        st.success("Notes ready!")
        if result.get("warning"):
            st.warning(result["warning"])
        st.session_state["last_result"] = result
        render_results(result)

    elif "last_result" in st.session_state:
        st.divider()
        st.caption("Last result")
        render_results(st.session_state["last_result"])


if __name__ == "__main__":
    main()
