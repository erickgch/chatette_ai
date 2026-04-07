import asyncio
import os
import re
import threading
import tempfile
import subprocess
import uvicorn
import scipy.signal as signal_proc  # type: ignore[import-untyped]
import scipy.io.wavfile as wav_io  # type: ignore[import-untyped]
import numpy as np
from voice import whisper_model
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from settings_manager import ChatetteSettings
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path

# Load .env before importing any module that reads env vars at import time (e.g. chromadb telemetry)
load_dotenv(Path(__file__).parent.parent / ".env")

# Disable ChromaDB telemetry before any chromadb import
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Header
from scheduler import start_scheduler
from rag_lw import ask, handle_reminder, handle_calendar_event, handle_delete, handle_delete_event, handle_draft, handle_personal_note, handle_weather, clear_device_context, clear_device_pending, device_has_pending, handle_set_alarm, get_last_alarm
import bulb_controller
from chatette_tv import channel_registry as _channel_registry
from chatette_tv.cast_router import router as cast_router
from ingestion import (
    ingest_all, ingest_notes, ingest_calendar_events,
    ingest_emails, ingest_lists
)

# Patch chromadb Posthog telemetry AFTER all chromadb-touching imports are done.
# Chromadb ignores ANONYMIZED_TELEMETRY in some versions — this silences the noisy
# "capture() takes 1 positional argument but 3 were given" warnings at startup.
try:
    import chromadb.telemetry.product.posthog as _posthog
    _posthog.Posthog.capture = lambda self, *args, **kwargs: None
except Exception:
    pass


# ===================================
# Lifespan
# ===================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    print("🚀 Scheduler started in background")
    _channel_registry.init()
    yield
    print("🛑 Shutting down...")


app = FastAPI(
    title="Chatette Personal Assistant API",
    description="Your local AI personal assistant",
    version="4.0.0",
    lifespan=lifespan
)
app.include_router(cast_router, prefix="/cast")


# ===================================
# Request / Response Models
# ===================================

class QuestionRequest(BaseModel):
    question: str
    mode: str = "auto"
    lang: str = "en"

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
    model_config = ConfigDict(protected_namespaces=())

    model_selection: str
    email_days_window: int
    calendar_days_ahead: int
    calendar_days_behind: int

class ReminderResponse(BaseModel):
    index: int
    text: str
    due: str | None
    created: str

class RemindersListResponse(BaseModel):
    reminders: list[ReminderResponse]

class FileMeta(BaseModel):
    filename: str
    modified: str

class ListsResponse(BaseModel):
    lists: list[FileMeta]

class ListItemResponse(BaseModel):
    index: int
    text: str
    checked: bool

class ListDetailResponse(BaseModel):
    content: str
    items: list[ListItemResponse]

class DraftsResponse(BaseModel):
    drafts: list[FileMeta]

class ContentResponse(BaseModel):
    content: str

class SyncFilePayload(BaseModel):
    content: str = ""
    modified: str = ""

class SyncListPayload(BaseModel):
    filename: str
    content: str
    modified: str = ""

class SyncPushResponse(BaseModel):
    updated: list[str]
    conflict_files: list[str] = []
    synced_at: str

# SyncPullResponse: model defined but not yet wired to /sync/pull —
# wiring it requires restructuring build_pull_response() and updating
# sync_service.dart to match the new reminders shape.
class SyncPullResponse(BaseModel):
    reminders: list[ReminderResponse]
    conflict_files: list[str] = []
    timestamp: str

class CommandRequest(BaseModel):
    command: str   # e.g. "reminder", "calendar", "draft", "journal", "weather"
    payload: str = ""  # user-typed text after the command prefix
    lang: str = "en"


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


@app.post("/clear_context")
def clear_context(x_device: str = Header("app", alias="X-Device")):
    """Reset the in-memory conversation context for this device."""
    clear_device_context(x_device)
    return {"cleared": True}


@app.post("/clear_pending")
def clear_pending(x_device: str = Header("app", alias="X-Device")):
    """Discard pending confirmation state for this device."""
    clear_device_pending(x_device)
    return {"cleared": True}


# ===================================
# Chat
# ===================================

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest, x_device: str = Header("app", alias="X-Device")):
    """Send a text message and get a response."""
    print(f"📱 Received: question='{request.question}' mode='{request.mode}' device='{x_device}'")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if request.mode not in ["auto", "personal", "general"]:
        raise HTTPException(status_code=400, detail="Mode must be 'auto', 'personal', or 'general'")
    try:
        answer = ask(request.question, mode=request.mode, lang=request.lang, device_id=x_device)
        print(f"✅ Answer: '{answer[:80]}...'")
        return AnswerResponse(question=request.question, answer=answer, mode=request.mode)
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
            raise HTTPException(status_code=429, detail="rate_limit_reached")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================
# Command (hardcoded shortcuts — no intent classification)
# ===================================

@app.post("/command", response_model=AnswerResponse)
def command(request: CommandRequest):
    """
    Execute a hardcoded command directly, bypassing LLM intent classification.
    Used by app shortcut buttons to avoid token cost and latency of classification.
    """
    cmd = request.command.strip().lower()
    payload = request.payload.strip()
    lang = request.lang

    # Build a synthetic question from command + payload for context
    question = payload if payload else cmd

    try:
        if cmd == "reminder":
            extracted = {"items": [payload]} if payload else {}
            answer = handle_reminder(question, extracted, lang)

        elif cmd == "calendar":
            extracted = {"title": payload, "description": ""} if payload else {}
            answer = handle_calendar_event(question, extracted, lang)

        elif cmd == "draft":
            extracted = {"type": "", "purpose": payload} if payload else {}
            answer = handle_draft(question, extracted, lang)

        elif cmd == "journal":
            extracted = {"items": [payload]} if payload else {}
            answer = handle_personal_note(question, extracted, lang)

        elif cmd == "weather":
            # Parse free-form payload — separate city from timeframe keywords
            timeframe = "today"
            city = payload.strip()
            # Strip leading prepositions ("weather in London" → "London")
            city = re.sub(r"^(in|for|at|near)\s+", "", city, flags=re.IGNORECASE).strip()
            timeframe_keywords = {
                "tomorrow": ["tomorrow", "morgen", "mañana"],
                "now":      ["now", "current", "currently", "jetzt", "aktuell", "ahora"],
                "week":     ["this week", "week", "woche", "semana", "7 days"],
            }
            for tf, keywords in timeframe_keywords.items():
                for kw in keywords:
                    if kw in city.lower():
                        timeframe = tf
                        city = re.sub(rf"\b{re.escape(kw)}\b", "", city, flags=re.IGNORECASE).strip(" ,")
                        break
            extracted = {"city": city.strip(), "timeframe": timeframe}
            answer = handle_weather(question, extracted, lang)

        elif cmd == "delete_reminder":
            extracted = {"text": payload}
            answer = handle_delete(question, extracted, lang)

        elif cmd == "delete_event":
            extracted = {"title": payload}
            answer = handle_delete_event(question, extracted, lang)

        elif cmd == "list":
            from rag_lw import handle_create_list
            answer = handle_create_list(question, {}, lang)

        elif cmd == "delete_list":
            from rag_lw import handle_delete_list
            extracted = {"list_name": payload}
            answer = handle_delete_list(question, extracted, lang)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {cmd}")

        print(f"✅ Command '{cmd}' → '{answer[:80]}...'")
        return AnswerResponse(question=question, answer=answer, mode="command")

    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
            raise HTTPException(status_code=429, detail="rate_limit_reached")
        raise HTTPException(status_code=500, detail=str(e))


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


_CONFIRM_NEGATIVES = {"no", "nein", "nope", "cancel", "cancelar", "abbrechen"}

# Last spoken language per device — keeps TTS consistent across confirm/cancel button presses
_device_last_lang: dict[str, str] = {}

_PIPER_VOICE_MAP = {
    "en": None,  # resolved at runtime from PIPER_VOICE env var
    "de": "piper/de_DE-ramona-low.onnx",
    "es": "piper/es_AR-daniela-high.onnx",
}

_processing_lock = asyncio.Lock()

_EMPTY_TRANSCRIPT_MSG = {
    "en": "I didn't get that. Can you repeat that?",
    "de": "Das habe ich nicht verstanden. Kannst du das wiederholen?",
    "es": "No entendí. ¿Puedes repetirlo?",
}


@app.post("/voice/chat")
async def voice_chat(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    x_intent: str = Header(None, alias="X-Intent"),
    x_device: str = Header("app", alias="X-Device"),
    x_timer_seconds: str = Header(None, alias="X-Timer-Seconds"),
    x_alarm_time:    str = Header(None, alias="X-Alarm-Time"),
    x_skip_whisper:  str = Header(None, alias="X-Skip-Whisper"),
):
    if _processing_lock.locked():
        raise HTTPException(status_code=503, headers={"X-Chatette-Busy": "true"},
                            detail="Chatette is busy — please try again in a moment.")

    async with _processing_lock:
        return await _voice_chat_inner(
            file, lang, x_intent, x_device, x_timer_seconds, x_alarm_time,
            x_skip_whisper=x_skip_whisper)


async def _voice_chat_inner(file, lang, x_intent, x_device,
                             x_timer_seconds=None, x_alarm_time=None,
                             x_skip_whisper=None):
    PIPER_PATH = os.getenv("PIPER_PATH")

    # Save uploaded audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if x_skip_whisper == "true":
        # Button-driven intent — transcript will be set by intent-override logic below.
        # Skip Whisper entirely to save latency.
        transcript = ""
        spoken_lang = lang
        os.unlink(tmp_path)
    else:
        whisper_lang = lang if lang in ("en", "de", "es") else None
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=whisper_lang,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        seg_list = list(segments)
        os.unlink(tmp_path)
        # Filter out hallucinated segments (high no_speech_prob)
        real_segs = [s for s in seg_list if s.no_speech_prob < 0.6]
        transcript = " ".join(s.text for s in real_segs).strip()
        spoken_lang = (
            info.language
            if (transcript and info.language in _PIPER_VOICE_MAP)
            else lang
        )

    # Determine TTS language:
    # - Verbal requests: trust Whisper detection; store it for later.
    # - Silent button presses (confirm/cancel): use stored conversational language,
    #   but allow an explicit LANG button override (non-'en' client lang that differs from stored).
    if x_intent in ("confirm", "cancel", "bulb_on", "bulb_off"):
        stored = _device_last_lang.get(x_device)
        if stored and lang != 'en' and lang != stored and lang in _PIPER_VOICE_MAP:
            # LANG button was pressed — explicit override
            tts_lang = lang
            _device_last_lang[x_device] = lang
        else:
            tts_lang = stored or spoken_lang
    else:
        tts_lang = spoken_lang
        _device_last_lang[x_device] = spoken_lang

    piper_voice = _PIPER_VOICE_MAP.get(tts_lang) or os.getenv("PIPER_VOICE")

    # Button-driven intent overrides — bypass transcription entirely.
    if x_intent == "confirm":
        normalized = transcript.lower().strip().rstrip(".,!?")
        if normalized not in _CONFIRM_NEGATIVES:
            transcript = "yes"
    elif x_intent == "cancel":
        transcript = "no"
    elif x_intent == "bulb_on":
        transcript = "turn on the lights"
    elif x_intent == "bulb_off":
        transcript = "turn off the lights"
    elif x_intent == "get_weather" and not transcript:
        _weather_defaults = {
            "en": "what's the weather today?",
            "de": "wie ist das Wetter heute?",
            "es": "qué tiempo hace hoy?",
        }
        transcript = _weather_defaults.get(lang, _weather_defaults["en"])
    elif x_intent == "view_agenda" and not transcript:
        _agenda_defaults = {
            "en": "give me a summary of my agenda, reminders, and recent emails",
            "de": "gib mir eine Übersicht meiner Termine, Erinnerungen und letzten E-Mails",
            "es": "dame un resumen de mi agenda, recordatorios y correos recientes",
        }
        transcript = _agenda_defaults.get(lang, _agenda_defaults["en"])
    elif x_intent == "view_emails" and not transcript:
        _email_defaults = {
            "en": "what emails have I received recently?",
            "de": "welche E-Mails habe ich zuletzt erhalten?",
            "es": "qué correos he recibido recientemente?",
        }
        transcript = _email_defaults.get(lang, _email_defaults["en"])

    if not transcript:
        answer = _EMPTY_TRANSCRIPT_MSG.get(tts_lang, _EMPTY_TRANSCRIPT_MSG["en"])
    elif x_timer_seconds:
        # Key-input countdown timer — bypass LLM, call handler directly
        answer = handle_set_alarm("", {"seconds": int(x_timer_seconds)}, tts_lang)
    elif x_alarm_time:
        # Key-input wall-clock alarm — bypass LLM, call handler directly
        h, m = map(int, x_alarm_time.split(":"))
        answer = handle_set_alarm("", {"hour": h, "minute": m}, tts_lang)
    else:
        # Get answer — use intent as mode hint if provided
        mode = "auto"
        if x_intent and x_intent not in ("general", "free"):
            mode = x_intent
        if mode in ("bulb_on", "bulb_off"):
            mode = "control_bulbs"
        answer = ask(transcript, mode=mode, lang=tts_lang, device_id=x_device)

    # Strip emojis before TTS (they appear in text but should not be spoken)
    import unicodedata
    def _strip_emojis(text: str) -> str:
        return "".join(
            c for c in text
            if not (unicodedata.category(c) in ("So", "Mn")
                    or 0x1F300 <= ord(c) <= 0x1FAFF
                    or 0x2600  <= ord(c) <= 0x27BF
                    or 0xFE00  <= ord(c) <= 0xFE0F)
        ).strip()
    tts_text = _strip_emojis(answer)

    # Generate TTS — piper on Windows exits with non-zero due to a stack-guard
    # firing during cleanup, but the output WAV is fully written before that.
    # Use check=False and verify the file instead.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name
    subprocess.run(
        [PIPER_PATH, "--model", piper_voice, "--length-scale", "1.2", "--output_file", output_path],
        input=tts_text.encode("utf-8"),
        check=False,
    )
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
        raise RuntimeError(f"Piper failed to produce audio for: {tts_text[:60]}")

    # Resample to 48000Hz for Pi
    rate, data = wav_io.read(output_path)
    if rate != 48000:
        up = 48000 // np.gcd(48000, rate)
        down = rate // np.gcd(48000, rate)
        data = signal_proc.resample_poly(data, up, down).astype(np.int16)
        wav_io.write(output_path, 48000, data)

    
    def _safe_header(value: str) -> str:
        if not value:
            return ""
        # Remove CR/LF (breaks HTTP header framing)
        value = value.replace("\r", " ").replace("\n", " ")
        # Remove control characters (0x00-0x1F except tab, and DEL 0x7F)
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1F\x7F]", "", value)
        # HTTP headers are Latin-1 bytes — drop chars outside that range (> U+00FF)
        # but preserve accented letters, umlauts, etc. that are within Latin-1
        value = value.encode("latin-1", errors="ignore").decode("latin-1")
        return value[:2000]

    alarm_data = get_last_alarm()
    resp_headers = {
        "X-Transcript": _safe_header(transcript),
        "X-Answer": _safe_header(answer),
        "X-Awaiting-Confirm": "1" if device_has_pending(x_device) else "0",
    }
    if alarm_data.get('seconds'):
        resp_headers["X-Timer-Seconds"] = str(alarm_data['seconds'])
    if alarm_data.get('alarm_time'):
        h, m = alarm_data['alarm_time']
        resp_headers["X-Alarm-Time"] = f"{h:02d}:{m:02d}"
    return FileResponse(output_path, media_type="audio/wav", headers=resp_headers)

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

@app.get("/reminders", response_model=RemindersListResponse)
def get_reminders():
    """Get all reminders as structured objects."""
    from note_manager import get_reminders_list
    return {"reminders": [
        {"index": i, **r.model_dump()}
        for i, r in enumerate(get_reminders_list())
    ]}

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

@app.get("/personal-notes", response_model=ContentResponse)
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

@app.get("/drafts", response_model=DraftsResponse)
def get_drafts():
    """List all saved drafts."""
    from note_manager import get_all_drafts
    return {"drafts": get_all_drafts()}

@app.get("/drafts/{filename}", response_model=ContentResponse)
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

@app.get("/lists", response_model=ListsResponse)
def get_lists():
    """List all lists."""
    from note_manager import get_all_lists
    return {"lists": get_all_lists()}

@app.get("/lists/{filename}", response_model=ListDetailResponse)
def get_list(filename: str):
    """Get a list's items as structured data."""
    from note_manager import get_list_items, get_list_content
    return {
        "content": get_list_content(filename),
        "items": [i.model_dump() for i in get_list_items(filename)]
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

@app.get("/settings", response_model=ChatetteSettings)
def get_settings():
    """Return current settings."""
    from settings_manager import read_settings
    return read_settings()


@app.post("/settings")
def save_settings(request: SettingsRequest):
    """Save settings to .env and restart api.py."""
    from settings_manager import read_settings, write_settings, ChatetteSettings
    current = read_settings()
    updated = ChatetteSettings(
        server_url=current.server_url,
        model_selection=request.model_selection,
        use_groq=request.model_selection != "local",
        groq_model=request.model_selection if request.model_selection != "local" else current.groq_model,
        email_days_window=request.email_days_window,
        calendar_days_ahead=request.calendar_days_ahead,
        calendar_days_behind=request.calendar_days_behind,
    )
    success = write_settings(updated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save settings")

    from rag_lw import reload_llm
    reload_llm()
    return {"status": "saved"}


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
# Sync
# ===================================

class SyncPushRequest(BaseModel):
    reminders: SyncFilePayload = SyncFilePayload()
    personal_notes: SyncFilePayload = SyncFilePayload()
    lists: list[SyncListPayload] = []


@app.get("/sync/pull")
def sync_pull():
    """Return current state of all data for phone to cache."""
    from sync_manager import build_pull_response
    return build_pull_response()


@app.post("/sync/push", response_model=SyncPushResponse)
def sync_push(request: SyncPushRequest):
    """Receive phone changes and apply to PC files."""
    from sync_manager import apply_push
    from ingestion import ingest_notes, ingest_lists
    result = apply_push({
        "reminders": request.reminders.model_dump(),
        "personal_notes": request.personal_notes.model_dump(),
        "lists": [item.model_dump() for item in request.lists],
    })
    # Re-ingest anything that changed
    if any("list" in u for u in result["updated"]):
        ingest_lists()
    if any(u in ["reminders", "personal_notes"] for u in result["updated"]):
        ingest_notes()
    return result


# ===================================
# Bulb Control
# ===================================

class BulbBrightnessRequest(BaseModel):
    level: int = 50

class BulbColorRequest(BaseModel):
    hue: int = 240

class BulbTemperatureRequest(BaseModel):
    kelvin: int = 2700


@app.post("/bulb/on")
def bulb_on():
    try:
        bulb_controller.turn_on()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/bulb/off")
def bulb_off():
    try:
        bulb_controller.turn_off()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/bulb/brightness")
def bulb_brightness(req: BulbBrightnessRequest):
    try:
        bulb_controller.set_brightness(req.level)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/bulb/color")
def bulb_color(req: BulbColorRequest):
    try:
        bulb_controller.set_color(req.hue, 100)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/bulb/temperature")
def bulb_temperature(req: BulbTemperatureRequest):
    try:
        bulb_controller.set_color_temperature(req.kelvin)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/bulb/status")
def bulb_status():
    try:
        return bulb_controller.get_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ===================================
# Run server
# ===================================

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)