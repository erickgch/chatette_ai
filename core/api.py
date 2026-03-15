import threading
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from scheduler import start_scheduler
from rag import ask
from ingestion import ingest_all, ingest_notes, ingest_calendar_events, ingest_emails
import tempfile
import os

# ===== Lifespan: runs scheduler on startup =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    print("🚀 Scheduler started in background")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Chatette Personal Assistant API",
    description="Your local AI personal assistant",
    version="2.0.0",
    lifespan=lifespan
)

# ===== Request/Response Models =====
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# ===== Chat Endpoints =====
@app.get("/")
def root():
    return {"status": "Chatette is running!"}

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    """Send a text message and get a text response."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = ask(request.question)
    return AnswerResponse(question=request.question, answer=answer)

# ===== Voice Endpoints =====
@app.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receive audio file, transcribe it and return text."""
    from voice import whisper_model
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    segments, _ = whisper_model.transcribe(tmp_path)
    transcript = " ".join([seg.text for seg in segments]).strip()
    os.unlink(tmp_path)
    return {"transcript": transcript}

@app.post("/voice/speak")
def text_to_speech(request: QuestionRequest):
    """Convert text to speech and return audio file."""
    from voice import speak
    import tempfile
    import soundfile as sf
    import subprocess
    from dotenv import load_dotenv
    load_dotenv()
    PIPER_PATH = os.getenv("PIPER_PATH")
    PIPER_VOICE = os.getenv("PIPER_VOICE")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name
    subprocess.run(
        [PIPER_PATH, "--model", PIPER_VOICE, "--output_file", output_path],
        input=request.question.encode("utf-8"),
        check=True
    )
    return FileResponse(output_path, media_type="audio/wav")

@app.post("/voice/chat")
async def voice_chat(file: UploadFile = File(...)):
    """Full voice pipeline: audio in → text → RAG → audio out."""
    from voice import whisper_model
    from dotenv import load_dotenv
    import subprocess
    load_dotenv()
    PIPER_PATH = os.getenv("PIPER_PATH")
    PIPER_VOICE = os.getenv("PIPER_VOICE")

    # Step 1: Transcribe audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    segments, _ = whisper_model.transcribe(tmp_path)
    transcript = " ".join([seg.text for seg in segments]).strip()
    os.unlink(tmp_path)

    # Step 2: Get answer from RAG
    answer = ask(transcript)

    # Step 3: Convert answer to speech
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name
    subprocess.run(
        [PIPER_PATH, "--model", PIPER_VOICE, "--output_file", output_path],
        input=answer.encode("utf-8"),
        check=True
    )

    return FileResponse(output_path, media_type="audio/wav",
                        headers={"X-Transcript": transcript,
                                 "X-Answer": answer})

# ===== Ingestion Endpoints =====
@app.post("/ingest/all")
def trigger_ingest_all():
    """Manually trigger full ingestion."""
    ingest_all()
    return {"status": "Full ingestion complete"}

@app.post("/ingest/notes")
def trigger_ingest_notes():
    """Manually trigger notes ingestion."""
    ingest_notes()
    return {"status": "Notes ingestion complete"}

@app.post("/ingest/calendar")
def trigger_ingest_calendar():
    """Manually trigger calendar ingestion."""
    ingest_calendar_events()
    return {"status": "Calendar ingestion complete"}

@app.post("/ingest/emails")
def trigger_ingest_emails():
    """Manually trigger email ingestion."""
    ingest_emails()
    return {"status": "Email ingestion complete"}

# ===== Run server =====
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)