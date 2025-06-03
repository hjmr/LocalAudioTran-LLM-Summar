import streamlit as st
import requests
import time
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/frontend.log"), logging.StreamHandler()],
)

logger = logging.getLogger("streamlit")

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None


def copy_to_clipboard(text: str, button_label: str = "📋 Copy to Clipboard", key: str = None) -> None:
    """Create a copy button for text content"""
    if st.button(button_label, key=key):
        try:
            st.code(text, language=None)  # Display as copyable code block
            st.success("✅ Content copied to clipboard!")
        except Exception as e:
            st.error(f"Failed to copy: {e}")


def display_summary(summary: Dict):
    """Display the structured summary with sections"""
    if not summary:
        st.warning("No summary available")
        return

    try:
        # Header with Copy Button
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.subheader("Summary")
        with col2:
            if "full_text" in summary and summary["full_text"]:
                copy_to_clipboard(summary["full_text"], key="copy_summary")

        # Overview Section
        if "overview" in summary and summary["overview"]:
            st.markdown("**概要（Overview）**")
            st.write(summary["overview"])

        # Main Points Section
        if "main_points" in summary and summary["main_points"]:
            st.markdown("**主な議題（Main Points）**")
            for point in summary["main_points"]:
                st.markdown(f"- {point}")

        # Key Insights Section
        if "key_insights" in summary and summary["key_insights"]:
            st.markdown("**重要なポイント（Key Insights）**")
            for insight in summary["key_insights"]:
                st.markdown(f"- {insight}")

        # Action Items / Decisions Section
        if "action_items_decisions" in summary and summary["action_items_decisions"]:
            st.markdown("**アクションアイテム／意思決定（Action Items / Decisions）**")
            for item in summary["action_items_decisions"]:
                st.markdown(f"- {item}")

        # Open Questions / Next Steps Section
        if "open_questions_next_steps" in summary and summary["open_questions_next_steps"]:
            st.markdown("**未解決の問題／次のステップ（Open Questions / Next Steps）**")
            for question in summary["open_questions_next_steps"]:
                st.markdown(f"- {question}")

        # Conclusions Section
        if "conclusions" in summary and summary["conclusions"]:
            st.markdown("**結論（Conclusions）**")
            for conclusion in summary["conclusions"]:
                st.markdown(f"- {conclusion}")

        # Full Text at bottom
        if "full_text" in summary and summary["full_text"]:
            st.divider()
            with st.expander("Show Full Summary Text"):
                st.write(summary["full_text"])

    except Exception as e:
        st.error(f"Error displaying summary: {e}")
        logger.error(f"Error in display_summary: {e}")
        logger.exception("Full traceback:")


def main():
    st.set_page_config(page_title="Audio Transcription & LLM based Summarization", page_icon="🎙️", layout="wide")

    st.title("🎙️ Audio Transcription & Summary")

    # File upload section
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

    # Status containers
    status_container = st.empty()
    progress_container = st.empty()
    error_container = st.empty()
    debug_container = st.empty()

    if uploaded_file:
        logger.info(f"File uploaded: {uploaded_file.name}")

        if st.button("Process Audio"):
            try:
                start_time = time.time()
                logger.info("Starting audio processing")

                # Reset containers
                status_container.empty()
                progress_container.empty()
                error_container.empty()
                debug_container.empty()

                # Show progress
                progress_bar = progress_container.progress(0)
                status_container.info("📤 Uploading audio file...")

                params = {"llm_name": "gemma3:latest", "whisper_model": "medium"}
                files = {"file": uploaded_file}
                logger.info("Sending request to API")

                response = requests.post("http://app:8000/transcribe", params=params, files=files, timeout=7200)

                if response.status_code == 200:
                    data = response.json()
                    logger.info("Received successful response from API")

                    # Debug the response data
                    logger.info(f"Response data: {data}")

                    # Store results in session state
                    st.session_state.transcription = data.get("transcription", "")
                    st.session_state.summary = data.get("summary", {})
                    st.session_state.processing_time = data.get("processing_time", {})

                    # Debug info
                    debug_info = {
                        "Response Keys": list(data.keys()),
                        "Summary Keys": list(data["summary"].keys()) if "summary" in data else None,
                        "Processing Times": data.get("processing_time"),
                        "Transcription Length": len(data.get("transcription", "")),
                        "Summary Content": data.get("summary"),  # Add this to see the actual summary content
                    }

                    logger.info(f"Debug info: {debug_info}")
                    debug_container.json(debug_info)

                    progress_bar.progress(100)
                    status_container.success("✅ Processing Complete!")

                    logger.info(f"Total frontend processing time: {time.time() - start_time:.2f} seconds")
                    st.rerun()
                else:
                    error_msg = f"API Error: {response.status_code}"
                    logger.error(error_msg)
                    error_container.error(error_msg)
                    st.write("Response content:", response.text)
                    progress_bar.progress(0)

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(error_msg)
                logger.exception("Full traceback:")
                error_container.error(error_msg)
                progress_container.progress(0)

    # Display results if available
    if hasattr(st.session_state, "transcription") and hasattr(st.session_state, "summary"):
        tab1, tab2 = st.tabs(["📝 Transcription", "📋 Summary"])

        with tab1:
            # Header with Copy Button for Transcription
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.subheader("Transcription")
            with col2:
                if st.session_state.transcription:
                    copy_to_clipboard(st.session_state.transcription, key="copy_transcription")

            st.text_area("Full Transcription", st.session_state.transcription, height=300)

        with tab2:
            if st.session_state.summary:
                display_summary(st.session_state.summary)
            else:
                st.warning("No summary available")

        # Processing time display
        if hasattr(st.session_state, "processing_time") and st.session_state.processing_time:
            try:
                st.info(
                    "Processing Times:\n"
                    f"Transcription: {st.session_state.processing_time.get('transcription', 0.0):.2f}s\n"
                    f"Summarization: {st.session_state.processing_time.get('summarization', 0.0):.2f}s\n"
                    f"Total: {st.session_state.processing_time.get('total', 0.0):.2f}s"
                )
            except Exception as e:
                logger.error(f"Error displaying processing times: {e}")
                st.warning("Processing time information not available")


if __name__ == "__main__":
    main()
