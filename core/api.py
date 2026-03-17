import os
import threading
import tempfile
import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from scheduler import start_scheduler
from rag import ask
from ingestion import (
    ingest_all, ingest_notes, ingest_calendar_events,
    ingest_emails, ingest_lists
)

load_dotenv()


# ===================================
# Lifespan
# ===================================

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
    version="4.0.0",
    lifespan=lifespan
)


# ===================================
# Request / Response Models
# ===================================

class QuestionRequest(BaseModel):
    question: str
    mode: str = "auto"

class AnswerResponse(BaseModel):
    question: str
    answer: str
    mode: str

class ContentUpdateRequest(BaseModel):
    content: str

class ListCreateRequest(BaseModel):
    title: str
    items: list[str] = []

class AddItemRequest(BaseModel):
    item: str

class SettingsRequest(BaseModel):
    model_selection: str
    email_days_window: int
    calendar_days_ahead: int
    calendar_days_behind: int


# ===================================
# Status
# ===================================

@app.get("/")
def root():
    return {"status": "Chatette is running!"}

@app.get("/status")
def status():
    """Return server status and current LLM model."""
    use_groq = os.getenv("USE_GROQ", "false").lower() == "true"
    model = os.getenv("GROQ_MODEL") if use_groq else os.getenv("OLLAMA_MODEL")
    return {"status": "running", "model": model}


# ===================================
# Chat
# ===================================

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    """Send a text message and get a response."""
    print(f"📱 Received: question='{request.question}' mode='{request.mode}'")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if request.mode not in ["auto", "personal", "general"]:
        raise HTTPException(status_code=400, detail="Mode must be 'auto', 'personal', or 'general'")
    answer = ask(request.question, mode=request.mode)
    print(f"✅ Answer: '{answer[:80]}...'")
    return AnswerResponse(question=request.question, answer=answer, mode=request.mode)


# ===================================
# Voice
# ===================================

@app.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receive audio file, transcribe and return text."""
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
    PIPER_PATH = os.getenv("PIPER_PATH")
    PIPER_VOICE = os.getenv("PIPER_VOICE")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    segments, _ = whisper_model.transcribe(tmp_path)
    transcript = " ".join([seg.text for seg in segments]).strip()
    os.unlink(tmp_path)

    answer = ask(transcript)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name
    subprocess.run(
        [PIPER_PATH, "--model", PIPER_VOICE, "--output_file", output_path],
        input=answer.encode("utf-8"),
        check=True
    )
    return FileResponse(output_path, media_type="audio/wav",
                        headers={"X-Transcript": transcript, "X-Answer": answer})


# ===================================
# Ingestion
# ===================================

@app.post("/ingest/all")
def trigger_ingest_all():
    ingest_all()
    return {"status": "Full ingestion complete"}

@app.post("/ingest/notes")
def trigger_ingest_notes():
    ingest_notes()
    return {"status": "Notes ingestion complete"}

@app.post("/ingest/calendar")
def trigger_ingest_calendar():
    ingest_calendar_events()
    return {"status": "Calendar ingestion complete"}

@app.post("/ingest/emails")
def trigger_ingest_emails():
    ingest_emails()
    return {"status": "Email ingestion complete"}

@app.post("/ingest/lists")
def trigger_ingest_lists():
    ingest_lists()
    return {"status": "Lists ingestion complete"}


# ===================================
# Reminders
# ===================================

@app.get("/reminders")
def get_reminders():
    """Get all reminders as a list of lines."""
    from note_manager import get_reminders_as_lines
    return {"reminders": get_reminders_as_lines()}

@app.delete("/reminders/{index}")
def delete_reminder_line(index: int):
    """Delete a reminder by line index."""
    from note_manager import delete_reminder_by_index
    result = delete_reminder_by_index(index)
    ingest_notes()
    return {"status": result}


# ===================================
# Personal Notes
# ===================================

@app.get("/personal-notes")
def get_personal_notes():
    """Get personal notes content."""
    from note_manager import get_all_personal_notes
    return {"content": get_all_personal_notes()}

@app.put("/personal-notes")
def update_personal_notes_endpoint(request: ContentUpdateRequest):
    """Update personal notes content."""
    from note_manager import update_personal_notes
    update_personal_notes(request.content)
    ingest_notes()
    return {"status": "Personal notes updated"}

@app.delete("/personal-notes")
def delete_personal_notes_endpoint():
    """Clear all personal notes."""
    from note_manager import delete_personal_notes
    delete_personal_notes()
    ingest_notes()
    return {"status": "Personal notes cleared"}


# ===================================
# Drafts
# ===================================

@app.get("/drafts")
def get_drafts():
    """List all saved drafts."""
    from note_manager import get_all_drafts
    return {"drafts": get_all_drafts()}

@app.get("/drafts/{filename}")
def get_draft(filename: str):
    """Get content of a specific draft."""
    from note_manager import get_draft_content
    return {"content": get_draft_content(filename)}

@app.put("/drafts/{filename}")
def update_draft_endpoint(filename: str, request: ContentUpdateRequest):
    """Update a draft's content."""
    from note_manager import update_draft
    success = update_draft(filename, request.content)
    if not success:
        raise HTTPException(status_code=404, detail="Draft not found")
    ingest_notes()
    return {"status": "Draft updated"}

@app.delete("/drafts/{filename}")
def delete_draft_endpoint(filename: str):
    """Delete a draft."""
    from note_manager import delete_draft
    success = delete_draft(filename)
    if not success:
        raise HTTPException(status_code=404, detail="Draft not found")
    ingest_notes()
    return {"status": "Draft deleted"}


# ===================================
# Lists
# ===================================

@app.get("/lists")
def get_lists():
    """List all lists."""
    from note_manager import get_all_lists
    return {"lists": get_all_lists()}

@app.get("/lists/{filename}")
def get_list(filename: str):
    """Get a list's items as structured data."""
    from note_manager import get_list_items, get_list_content
    return {
        "content": get_list_content(filename),
        "items": get_list_items(filename)
    }

@app.post("/lists")
def create_list_endpoint(request: ListCreateRequest):
    """Create a new list."""
    from note_manager import create_list
    filename = create_list(request.title, request.items)
    ingest_lists()
    return {"filename": filename, "status": "List created"}

@app.put("/lists/{filename}")
def update_list_endpoint(filename: str, request: ContentUpdateRequest):
    """Update a list's full content."""
    from note_manager import update_list
    success = update_list(filename, request.content)
    if not success:
        raise HTTPException(status_code=404, detail="List not found")
    ingest_lists()
    return {"status": "List updated"}

@app.post("/lists/{filename}/items")
def add_list_item(filename: str, request: AddItemRequest):
    """Add an item to a list."""
    from note_manager import add_item_to_list
    success = add_item_to_list(filename, request.item)
    if not success:
        raise HTTPException(status_code=404, detail="List not found")
    ingest_lists()
    return {"status": "Item added"}

@app.put("/lists/{filename}/toggle/{item_index}")
def toggle_list_item_endpoint(filename: str, item_index: int):
    """Toggle a checkbox item."""
    from note_manager import toggle_list_item
    success = toggle_list_item(filename, item_index)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    ingest_lists()
    return {"status": "Item toggled"}

@app.delete("/lists/{filename}")
def delete_list_endpoint(filename: str):
    """Delete a list."""
    from note_manager import delete_list
    success = delete_list(filename)
    if not success:
        raise HTTPException(status_code=404, detail="List not found")
    ingest_lists()
    return {"status": "List deleted"}

@app.delete("/lists/{filename}/items/{line_index}")
def delete_list_item_endpoint(filename: str, line_index: int):
    """Delete a specific item from a list."""
    from note_manager import delete_list_item
    success = delete_list_item(filename, line_index)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    ingest_lists()
    return {"status": "Item deleted"}


# ===================================
# Settings
# ===================================

@app.get("/settings")
def get_settings():
    """Return current settings."""
    from settings_manager import read_settings
    return read_settings()


@app.post("/settings")
def save_settings(request: SettingsRequest):
    """Save settings to .env and restart api.py."""
    from settings_manager import write_settings
    success = write_settings({
        "model_selection": request.model_selection,
        "email_days_window": request.email_days_window,
        "calendar_days_ahead": request.calendar_days_ahead,
        "calendar_days_behind": request.calendar_days_behind,
    })
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save settings")

    def _restart():
        import time, sys
        time.sleep(1.5)
        print("🔄 Restarting Chatette with new settings...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_restart, daemon=True).start()
    return {"status": "saved", "restarting": True}


# ===================================
# Notifications Cache
# ===================================

@app.get("/notifications/cache")
def get_notifications_cache():
    """Return the pre-generated notifications cache file."""
    import json
    from pathlib import Path

    cache_path = Path(os.getenv("NOTES_PATH", ".")) / "notifications_cache.json"

    if not cache_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Notifications cache not yet generated. Try again shortly."
        )

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    return cache


# ===================================
# Run server
# ===================================

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)