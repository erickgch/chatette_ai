import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from scheduler import start_scheduler
from rag import ask
from ingestion import ingest_all, ingest_notes, ingest_calendar_events, ingest_emails

# ===== Lifespan: runs scheduler on startup =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    print("🚀 Scheduler started in background")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Personal Assistant API",
    description="Your local AI personal assistant",
    version="1.0.0",
    lifespan=lifespan
)

# ===== Request/Response Models =====
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# ===== Endpoints =====
@app.get("/")
def root():
    return {"status": "Personal Assistant is running!"}

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    """Send a question and get an answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = ask(request.question)
    return AnswerResponse(question=request.question, answer=answer)

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