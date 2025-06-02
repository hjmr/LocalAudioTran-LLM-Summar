from faster_whisper import WhisperModel
import logging
import tempfile
import os
from fastapi import UploadFile
import torch

logger = logging.getLogger("transcription")


class TranscriptionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = None
        self.model = None

    def load_model(self, model_size: str = "large-v3"):
        logger.info(f"Loading Faster-Whisper {model_size} model on {self.device}")
        try:
            self.model = WhisperModel(
                model_size, device=self.device, compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.model_size = model_size
            logger.info(f"Faster-Whisper {self.model_size} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {str(e)}")
            raise

    async def transcribe(self, model_size: str, file: UploadFile) -> str:
        """Transcribe audio file using Faster-Whisper"""
        if self.model_size != model_size or self.model is None:
            logger.info(f"Model size mismatch or model not loaded, loading model: {model_size}")
            self.load_model(model_size)
        try:
            logger.info(f"Starting transcription for {file.filename}")

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp.flush()

                logger.info("Processing audio with Faster-Whisper...")
                segments, info = self.model.transcribe(
                    tmp.name,
                    language="ja",  # Japanese
                    task="transcribe",
                    temperature=0.0,  # Use greedy decoding
                    best_of=1,  # No need for multiple samples with temperature=0
                    beam_size=5,  # Beam search size
                )

                # Clean up the temporary file
                os.unlink(tmp.name)

                transcription = [segment.text for segment in segments if segment.text.strip()]
                transcription = " ".join(transcription).strip()

                # Unload model to free GPU memory
                logger.info("Transcription complete, unloading model...")
                self.unload_model()

                return transcription

        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            # Ensure model is unloaded even if transcription fails
            self.unload_model()
            raise Exception(f"Transcription failed: {str(e)}")

    def unload_model(self):
        """Unload the model to free up GPU memory"""
        try:
            if hasattr(self, "model"):
                del self.model
                self.model = self.model_size = None
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                logger.info("Whisper model unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
