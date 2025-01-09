from pydantic import BaseModel

class TranscriptionResponse(BaseModel):
    transcription: str
    refined_notes: str 