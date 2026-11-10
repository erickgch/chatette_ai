import asyncio
import os
import re
import sys
import threading
import tempfile
import subprocess
import uvicorn
from datetime import datetime
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

# ===================================
# Session log
# ===================================

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_SESSION_LOG_PATH = _LOG_DIR / f"chatette_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_session_log_file = open(_SESSION_LOG_PATH, "w", encoding="utf-8", buffering=1)


class _Tee:
    """Write to both the original stream and the session log file."""
    def __init__(self, original):
        self._orig = original

    def write(self, data):
        self._orig.write(data)
        try:
            _session_log_file.write(data)
        except Exception:
            pass

    def flush(self):
        self._orig.flush()
        try:
            _session_log_file.flush()
        except Exception:
            pass

    def __getattr__(self, attr):
        return getattr(self._orig, attr)


sys.stdout = _Tee(sys.stdout)
sys.stderr = _Tee(sys.stderr)

# Load .env before importing any module that reads env vars at import time (e.g. chromadb telemetry)
load_dotenv(Path(__file__).parent.parent / ".env")

# Disable ChromaDB telemetry before any chromadb import
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Header, Request
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

def _start_notes_watcher():
    """Watch data/docs/ for .txt changes and re-ingest automatically."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    notes_dir = Path(__file__).parent.parent / "data" / "docs"
    _debounce_timer: dict = {}

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith(".txt"):
                self._trigger()

        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".txt"):
                self._trigger()

        def _trigger(self):
            if "t" in _debounce_timer:
                _debounce_timer["t"].cancel()
            import threading as _t
            timer = _t.Timer(2.0, _do_ingest)
            _debounce_timer["t"] = timer
            timer.start()

    def _do_ingest():
        try:
            ingest_notes()
            print("📝 Notes auto-ingested (file change detected)")
        except Exception as e:
            print(f"⚠️ Auto-ingest failed: {e}")

    observer = Observer()
    observer.schedule(_Handler(), str(notes_dir), recursive=False)
    observer.daemon = True
    observer.start()
    print("👁️ Notes watcher started")


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()
    print("🚀 Scheduler started in background")
    _channel_registry.init()
    _start_notes_watcher()
    yield
    print("🛑 Shutting down...")


# ===================================
# API key — every request must carry X-API-Key (except "/", the reachability ping)
# ===================================
# Without it, any device on the network (or a web page open in the browser) could read
# notes and emails, write to the calendar, change settings, switch lights and TV.
# Fails closed: no key configured → every request is refused.

import secrets
API_KEY = os.getenv("CHATETTE_API_KEY", "")
_OPEN_PATHS = {"/"}

if not API_KEY:
    print("❌ CHATETTE_API_KEY is not set in .env — all API requests will be refused")


def require_api_key(request: Request):
    if request.url.path in _OPEN_PATHS:
        return
    sent = request.headers.get("X-API-Key", "")
    if not API_KEY or not secrets.compare_digest(sent.encode(), API_KEY.encode()):
        print(f"🔒 Refused {request.method} {request.url.path} from "
              f"{request.client.host if request.client else '?'} — {'wrong' if sent else 'no'} API key")
        raise HTTPException(status_code=401, detail="invalid_api_key")


from fastapi import Depends
app = FastAPI(
    title="Chatette Personal Assistant API",
    description="Your local AI personal assistant",
    version="4.0.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
    # The interactive docs would list every endpoint without needing the key
    docs_url=None, redoc_url=None, openapi_url=None,
)
app.include_router(cast_router, prefix="/cast")

# Allow the browser UI (chatette_web/index.html, opened from disk) to call the API.
# A page opened from disk has the origin "null"; no other web page may call the API.
# expose_headers lets the page read the transcript/answer headers of /voice/chat.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-Answer", "X-Awaiting-Confirm",
                    "X-Timer-Seconds", "X-Alarm-Time", "X-Chatette-Busy"],
)


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
    thinking_enabled: bool = False
    # None = not sent (e.g. the phone app, which doesn't know this setting) → keep current value
    auto_rag_selector: bool | None = None

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
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    return {"status": "running", "model": model}


@app.post("/logs/generate")
def generate_log():
    """Flush and return info about the current session log."""
    try:
        _session_log_file.flush()
        size = _SESSION_LOG_PATH.stat().st_size
        return {
            "status": "ok",
            "log_file": _SESSION_LOG_PATH.name,
            "path": str(_SESSION_LOG_PATH),
            "size_bytes": size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
# Client identification (for the logs)
# ===================================

_DEVICE_NAMES = {"web": "Web app", "app": "Phone app", "s_phone": "FeTAp phone"}


def _short_user_agent(ua: str) -> str:
    """Condense a User-Agent header into e.g. 'Firefox on Windows' or 'Flutter/Dart'."""
    if not ua:
        return ""
    if ua.startswith("Dart/"):
        return "Flutter/Dart"
    if ua.startswith("python-requests"):
        return "Python requests"
    browser = next((name for key, name in (("Edg/", "Edge"), ("OPR/", "Opera"), ("Firefox/", "Firefox"),
                                           ("Chrome/", "Chrome"), ("Safari/", "Safari")) if key in ua), "")
    system = next((name for key, name in (("Android", "Android"), ("iPhone", "iOS"), ("iPad", "iOS"),
                                          ("Windows", "Windows"), ("Mac OS", "macOS"), ("Linux", "Linux"))
                   if key in ua), "")
    if browser:
        return f"{browser} on {system}" if system else browser
    return ua.split(" ")[0][:40]


def _client_label(http: Request, device: str | None) -> str:
    """Who sent this request: friendly name, device id, IP address and client software."""
    ip = http.client.host if http.client else "?"
    agent = _short_user_agent(http.headers.get("user-agent", ""))
    if not device and agent == "Flutter/Dart":
        name = "Phone app"            # the app's shortcut buttons send no X-Device header
    else:
        name = _DEVICE_NAMES.get(device or "", device or "unknown client")
    return f"{name} [{device or '-'}] from {ip}" + (f", {agent}" if agent else "")


# ===================================
# Chat
# ===================================

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest, http: Request, x_device: str = Header("app", alias="X-Device")):
    """Send a text message and get a response."""
    print(f"📱 Received from {_client_label(http, x_device)}: question='{request.question}' mode='{request.mode}'")
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
def command(request: CommandRequest, http: Request, x_device: str = Header(None, alias="X-Device")):
    """
    Execute a hardcoded command directly, bypassing LLM intent classification.
    Used by app shortcut buttons to avoid token cost and latency of classification.
    """
    cmd = request.command.strip().lower()
    payload = request.payload.strip()
    # Answer in the language the user typed in (the app sends its UI language)
    from rag_lw import _detect_language
    lang = _detect_language(payload, request.lang) if payload else request.lang
    print(f"📱 Command from {_client_label(http, x_device)}: command='{cmd}' payload='{payload}'")

    # Build a synthetic question from command + payload for context
    question = payload if payload else cmd

    # Same device as the chat: a confirmation question ("Save this event?") must be pending
    # on the device that will answer "yes" (the app's buttons send no header → "app").
    from rag_lw import _current_device, _remember
    device_token = _current_device.set(x_device or "app")
    try:
        if cmd == "reminder":
            extracted = {"items": [payload]} if payload else {}
            answer = handle_reminder(question, extracted, lang)

        elif cmd == "calendar":
            # No title hint: the payload holds date and time too, and the model copies a hint verbatim
            answer = handle_calendar_event(question, {}, lang)

        elif cmd == "draft":
            extracted = {"type": "", "purpose": payload} if payload else {}
            answer = handle_draft(question, extracted, lang)

        elif cmd == "journal":
            extracted = {"items": [payload]} if payload else {}
            answer = handle_personal_note(question, extracted, lang)

        elif cmd == "weather":
            # Free-form payload, e.g. "for the next 3 days in Hamburg" → city + timeframe
            from weather import parse_weather_request
            city, timeframe = parse_weather_request(payload)
            extracted = {"city": city, "timeframe": timeframe}
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
        _remember(question, answer)   # so a follow-up ("Ja", "make it 11") has the context
        return AnswerResponse(question=question, answer=answer, mode="command")

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
            raise HTTPException(status_code=429, detail="rate_limit_reached")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _current_device.reset(device_token)


# ===================================
# Voice
# ===================================

@app.post("/voice/transcribe")
async def transcribe_audio(http: Request, file: UploadFile = File(...),
                           x_device: str = Header(None, alias="X-Device")):
    """Receive audio file, transcribe and return text."""
    print(f"🎤 Transcription request from {_client_label(http, x_device)}")
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
    "de": "piper/de_DE-kerstin-low.onnx",
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
    http: Request,
    file: UploadFile = File(...),
    lang: str = Form("en"),
    x_intent: str = Header(None, alias="X-Intent"),
    x_device: str = Header("app", alias="X-Device"),
    x_timer_seconds: str = Header(None, alias="X-Timer-Seconds"),
    x_alarm_time:    str = Header(None, alias="X-Alarm-Time"),
    x_skip_whisper:  str = Header(None, alias="X-Skip-Whisper"),
):
    print(f"🎤 Voice message from {_client_label(http, x_device)}"
          + (f" (button intent: {x_intent})" if x_intent else ""))
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
        answer = ask(transcript, mode=mode, lang=tts_lang, device_id=x_device, voice=True)

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
        [PIPER_PATH, "--model", piper_voice, "--length-scale", "1.35", "--output_file", output_path],
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


# One note at a time — used by the note-list editors. A note's header (creation time, and
# whether it is timestamped or timeless) is set once when it is added and never changes;
# it is the note's id. Editing only changes the text.

class PersonalNote(BaseModel):
    id: str
    created: str
    timeless: bool
    text: str

class PersonalNotesListResponse(BaseModel):
    notes: list[PersonalNote]
    legacy_text: str = ""          # text without headers (old format), if any is left

class PersonalNoteCreateRequest(BaseModel):
    text: str
    timestamped: bool = True       # False → a timeless note (lasting fact)

class PersonalNoteEditRequest(BaseModel):
    text: str


@app.get("/personal-notes/items", response_model=PersonalNotesListResponse)
def list_personal_notes_endpoint():
    """All personal notes, oldest first."""
    from note_manager import _read_personal_notes
    preamble, notes = _read_personal_notes()
    return {"notes": notes, "legacy_text": preamble}

@app.post("/personal-notes/items", response_model=PersonalNote)
def add_personal_note_endpoint(request: PersonalNoteCreateRequest):
    """Add a note; its header is fixed from now on."""
    from note_manager import add_personal_note
    note = add_personal_note(request.text, timestamped=request.timestamped)
    if not note:
        raise HTTPException(status_code=400, detail="Note text cannot be empty")
    ingest_notes()
    return note

@app.put("/personal-notes/items/{note_id}")
def edit_personal_note_endpoint(note_id: str, request: PersonalNoteEditRequest):
    """Change a note's text (its header stays the same)."""
    from note_manager import edit_personal_note
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Note text cannot be empty — delete the note instead")
    if not edit_personal_note(note_id, request.text):
        raise HTTPException(status_code=404, detail="Note not found")
    ingest_notes()
    return {"status": "Note updated"}

@app.delete("/personal-notes/items/{note_id}")
def delete_personal_note_endpoint(note_id: str):
    """Delete one note."""
    from note_manager import delete_personal_note
    if not delete_personal_note(note_id):
        raise HTTPException(status_code=404, detail="Note not found")
    ingest_notes()
    return {"status": "Note deleted"}


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
        email_days_window=request.email_days_window,
        calendar_days_ahead=request.calendar_days_ahead,
        calendar_days_behind=request.calendar_days_behind,
        thinking_enabled=request.thinking_enabled,
        auto_rag_selector=(current.auto_rag_selector if request.auto_rag_selector is None
                           else request.auto_rag_selector),
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

    cache_path = Path(os.getenv("DOCS_PATH", ".")) / "notifications_cache.json"

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