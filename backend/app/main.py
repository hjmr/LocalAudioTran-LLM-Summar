from fastapi import FastAPI, File, UploadFile, HTTPException
from .services.transcription import TranscriptionService
from .services.summarization import SummarizationService
from .utils.logger import setup_logger
import time

logger = setup_logger("api", "api.log")
transcription_logger = setup_logger("transcription", "transcription.log")
summarization_logger = setup_logger("summarization", "summarization.log")

app = FastAPI()

# Initialize services
transcription_service = TranscriptionService()
summarization_service = SummarizationService()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        logger.info(f"Starting processing for file: {file.filename}")

        # 1) Transcribe audio
        logger.info("Beginning transcription...")
        transcription = await transcription_service.transcribe(file)
        transcription_time = time.time()
        logger.info(f"Transcription completed in {transcription_time - start_time:.2f} seconds")

        if not transcription:
            raise HTTPException(status_code=400, detail="Transcription failed")

        logger.info("Beginning summarization pipeline...")
        summary = summarization_service.generate_summary(transcription)
        summarization_time = time.time()
        logger.info(f"Summarization completed in {summarization_time - transcription_time:.2f} seconds")

        return {
            "transcription": transcription,
            "summary": summary,
            "processing_time": {
                "transcription": transcription_time - start_time,
                "summarization": summarization_time - transcription_time,
                "total": time.time() - start_time
            }
        }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        logger.info("Initializing services...")
        summarization_service.load_model()  
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.exception("Full traceback:")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        logger.info("Cleaning up resources...")
        # Any teardown if needed
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        logger.exception("Full traceback:")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if summarization_service.model is None:
            return {"status": "warning", "message": "Model not loaded"}
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))