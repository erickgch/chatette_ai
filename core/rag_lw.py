import os
import json
import re
import requests
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from ingestion import ingest_notes, ingest_calendar_events, ingest_lists
import bulb_controller
from chatette_tv import cast_manager as _cast_manager, channel_registry as _channel_registry, youtube_search as _yt_search
from note_manager import (
    create_reminder, delete_reminder_by_line, get_all_reminders,
    get_reminders_list, delete_all_reminders,
    save_personal_note, save_draft,
    create_list, add_item_to_list, find_list_by_name,
    get_all_lists, delete_list, get_list_items, delete_list_item
)
from datetime import datetime, timedelta, timezone
from google_integration import CalendarEvent, create_calendar_event, delete_calendar_event, get_upcoming_events
from dt_utils import _local_tz, _parse_and_fix_dt
from weather import (
    get_current_weather, get_today_forecast, get_weekly_forecast,
    format_weather_context, geocode
)
from places import search_places
from pydantic import BaseModel, field_validator, ValidationError, ConfigDict

load_dotenv()
os.chdir(Path(__file__).parent.parent)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
USER_NAME = os.getenv("USER_NAME")
USER_PROFILE = os.getenv("USER_PROFILE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "false").lower() == "true"

# ── Calendar window — read from .env, set via app Settings ──────────────────
CALENDAR_DAYS_AHEAD = int(os.getenv("CALENDAR_DAYS_AHEAD", "14"))
CALENDAR_DAYS_BEHIND = int(os.getenv("CALENDAR_DAYS_BEHIND", "1"))
# Separate wider window for event deletion searches (default 60 days)
DELETE_EVENT_DAYS_AHEAD = int(os.getenv("DELETE_EVENT_DAYS_AHEAD", "90"))

# Initialize embeddings (always local)
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=EMBEDDING_MODEL
)

# Connect to existing ChromaDB
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# Initialize LLM — Groq or Ollama
if USE_GROQ:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL
    )
    print(f"LLM loaded: {GROQ_MODEL} via Groq ☁️")
else:
    llm = OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL
    )
    print(f"LLM loaded: {OLLAMA_MODEL} via Ollama 🖥️")


def reload_llm():
    """Hot-reload the LLM and calendar constants from .env (called after settings save)."""
    global llm, OLLAMA_MODEL, THINKING_ENABLED, CALENDAR_DAYS_AHEAD, CALENDAR_DAYS_BEHIND, AUTO_RAG_SELECTOR
    load_dotenv(override=True)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    THINKING_ENABLED = os.getenv("THINKING_ENABLED", "false").lower() == "true"
    AUTO_RAG_SELECTOR = os.getenv("AUTO_RAG_SELECTOR", "false").lower() == "true"
    print(f"Auto RAG selector: {'on' if AUTO_RAG_SELECTOR else 'off'}")
    CALENDAR_DAYS_AHEAD = int(os.getenv("CALENDAR_DAYS_AHEAD", "14"))
    CALENDAR_DAYS_BEHIND = int(os.getenv("CALENDAR_DAYS_BEHIND", "1"))
    llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    print(f"LLM reloaded: {OLLAMA_MODEL} via Ollama 🖥️")


# Model families that can "think" (reason before answering). Only these get Ollama's
# `think` flag — Ollama rejects it for models without thinking support (e.g. qwen2.5).
_THINKING_MODELS = ("qwen3", "deepseek-r1", "gpt-oss", "magistral")
_OLLAMA_TIMEOUT = 300   # seconds — long drafts on a cold model can take a while


def _ollama_generate(prompt: str) -> str:
    """Call Ollama's /api/generate directly.

    Why not OllamaLLM: Qwen3 thinks before every answer unless Ollama's `think` flag is
    false. The old "/no_think" prompt suffix is ignored by current Ollama versions, and
    langchain-ollama 0.2.0 can't pass the flag — so every call silently generated ~150
    hidden reasoning tokens (~3 s) that were stripped from the answer afterwards.
    """
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if any(name in OLLAMA_MODEL.lower() for name in _THINKING_MODELS):
        body["think"] = THINKING_ENABLED
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    resp = requests.post(url, json=body, timeout=_OLLAMA_TIMEOUT)
    if resp.status_code == 400 and "think" in body and "think" in resp.text.lower():
        # This model doesn't support the flag after all — retry without it
        body.pop("think")
        resp = requests.post(url, json=body, timeout=_OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "")


def llm_invoke(prompt: str) -> str:
    """Invoke LLM and always return a plain string."""
    try:
        _is_qwen = "qwen3" in OLLAMA_MODEL.lower()

        if USE_GROQ:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            text = _ollama_generate(prompt)

        # Strip any <think>...</think> blocks (thinking mode, or models that inline it)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Strip markdown bold/italic added by Qwen
        if _is_qwen:
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)

        return text
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str or "too many" in error_str or "quota" in error_str:
            raise RateLimitError("Groq rate limit reached")
        raise


class RateLimitError(Exception):
    pass


# ===================================
# TRANSLATION HELPER
# ===================================

_LANG_NAMES = {"en": "English", "de": "German", "es": "Spanish"}


def _t(en: str, de: str, es: str, lang: str) -> str:
    """Return the right translation based on language code."""
    if lang == "de":
        return de
    if lang == "es":
        return es
    return en


_TZ_ABBREVS = {
    "Central European Standard Time": "CET",
    "Central European Summer Time": "CEST",
    "Greenwich Mean Time": "GMT",
    "British Summer Time": "BST",
    "Eastern Standard Time": "EST",
    "Eastern Daylight Time": "EDT",
    "Pacific Standard Time": "PST",
    "Pacific Daylight Time": "PDT",
    "Coordinated Universal Time": "UTC",
}


_MONTHS = {
    'en': ['January','February','March','April','May','June',
           'July','August','September','October','November','December'],
    'de': ['Januar','Februar','März','April','Mai','Juni',
           'Juli','August','September','Oktober','November','Dezember'],
    'es': ['enero','febrero','marzo','abril','mayo','junio',
           'julio','agosto','septiembre','octubre','noviembre','diciembre'],
}

def _ordinal_en(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _fmt_due(date_str: str | None, lang: str = "en") -> str:
    """Convert YYYY-MM-DD → spoken date string for TTS (e.g. 'the 30th of March, 2026')."""
    if not date_str:
        return ""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        month = _MONTHS.get(lang, _MONTHS['en'])[d.month - 1]
        if lang == "de":
            return f"dem {d.day}. {month} {d.year}"
        elif lang == "es":
            return f"el {d.day} de {month} de {d.year}"
        else:
            return f"the {_ordinal_en(d.day)} of {month}, {d.year}"
    except Exception:
        return date_str


def _fmt_event_dt(iso: str | None, lang: str = "en") -> str:
    """Convert ISO datetime → spoken date+time string for TTS."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
        month = _MONTHS.get(lang, _MONTHS['en'])[dt.month - 1]
        time_part = dt.strftime("%H:%M")
        if lang == "de":
            return f"{dt.day}. {month} {dt.year} um {time_part} Uhr"
        elif lang == "es":
            return f"el {dt.day} de {month} de {dt.year} a las {time_part}h"
        else:
            return f"{month} {_ordinal_en(dt.day)}, {dt.year} at {time_part}"
    except Exception:
        return iso


# ===================================
# PERSONALITY
# ===================================

CHATETTE_PERSONA = """You are Chatette — a small robotic cat and personal assistant to {user_name}.
About {user_name}: {user_profile}

Your personality:
- Warm, friendly and competent.
- Keep answers concise and straight to the point, but stay friendly.
- You have your own voice. Not a generic AI bot. Not a corporate assistant.
- Diligent, accurate, but friendly. You answer first, add warmth second.
- Never over-explain or pad responses with unnecessary pleasantries.
- Never apologize unnecessarily.
- LANGUAGE: always answer in the language of the user's message (German message → German answer, Spanish → Spanish, English → English), even if the data or context you were given is in another language. Only if the message's language is unclear, use {lang_name}."""

CHATETTE_RULES = """
STRICT RULES:
- Only answer what was specifically asked. Nothing more. Do not make additional comments.
- Keep answers to 1-2 sentences maximum.
- Always respond in the same language of the QUESTION, not the context.
- If the question refers to a previous exchange, use it to answer correctly. But your notes and the context are the source of truth: if your previous answer contradicts them, trust the notes and correct yourself.
- When the question is about {user_name} personally (family, relationships, health, history, preferences, plans), look for the answer in the personal notes in the context as well — not only in the profile.
- Never invent personal information that is not in the profile or the personal notes.
- Each personal note starts with a header. "[YYYY-MM-DD HH:MM]" is when {user_name} wrote that note — use it for "when" questions (e.g. "In June 2026 you wrote that you wanted to…"). "[timeless · created …]" marks a lasting fact about him: never attach a date to it; the creation date is only its ID.
- Never refer to {user_name} in third person — use 'you' and 'your'.
- {user_name}'s notes are written by him in first person: "I"/"my" in a note means {user_name}. When answering, turn them into "you"/"your" (German "du"/"dein", Spanish "tú"/"tu") — never answer as "I" as if you were {user_name}. Answer the question in your own words; don't copy note text.
- When corrected on a factual error, say so briefly and drop it. Never propose an alternative wrong answer as a recovery.
- Never claim to control hardware or services not explicitly listed in your tools. You can control smart bulbs. That is all.
- Use only the parts of the context that answer the question. Never mention unrelated calendar events, reminders, emails or notes.
- For any date, weekday or "how long until/ago", copy the numbers from the "Date facts" — never calculate them yourself.
- When asked about yourself (your plans, feelings, preferences), answer honestly as a software assistant. You do not have a physical life, personal plans, or a schedule of your own."""


# Custom RAG prompt
prompt_template = PromptTemplate(
    template=CHATETTE_PERSONA + CHATETTE_RULES + """

If you don't find the answer in the context, say so briefly.
Use ONLY context that directly answers the question asked. Ignore everything else in the context.

Examples of the answer FORMAT. Words in <angle brackets> are placeholders for values that
would come from the real context — they are NOT facts about {user_name}; never use them.

Context: Today is <weekday, date>. Calendar event: <appointment> on <day> at <time>.
{user_name}: Do I have anything on <day>?
Chatette: Yes — <appointment> at <time>.

Context: Note: I spent a week in <city> in <month> and loved the food.
{user_name}: Have I been to <country>?
Chatette: Yes — you spent a week in <city> in <month>.

Context: Note: I am allergic to <thing>.
{user_name}: Do I have any allergies?
Chatette: Yes — you're allergic to <thing>.

Context: (no relevant information found)
{user_name}: What's my blood type?
Chatette: I don't have that in my notes.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question", "user_name", "user_profile"]
)

_current_device: ContextVar[str] = ContextVar("device_id", default="app")
_current_voice: ContextVar[bool] = ContextVar("voice", default=False)
_current_question: ContextVar[str] = ContextVar("question", default="")

# ── Auto RAG selector ───────────────────────────────────────────────────────────────
# On: chat modes are ignored. For general-conversation questions a separate small yes/no
# model call ("is this about the user's personal life?") decides: yes → always search
# personal data; no → the usual relevance cut-off. Off: auto/personal/general modes as
# before. Set via Settings (AUTO_RAG_SELECTOR in .env). Tested in
# experiments/agentic_rag (variant E): 28/30 personal, 30/30 general correct.
AUTO_RAG_SELECTOR = os.getenv("AUTO_RAG_SELECTOR", "false").lower() == "true"

_ABOUT_USER_PROMPT = """Decide whether answering this question requires personal information about the user
({user}) — his own life, body, health, family, friends, people he knows, home, plans,
schedule, history, preferences, notes or emails.

- true: the question is about people or facts of {user}'s personal life, so the answer
  depends on his own data.
- false: the answer is general/world knowledge — even if phrased personally, when the
  personal wording only wraps a general fact ("do I know the capital of France?", "am I aware
  that ice is cold?", "should I know…") — or it is about public figures or well-known things,
  even ones known by a first name (e.g. Claude).

Examples of true: "am I vegetarian?", "do I have a car?". Examples of false: "do I know who
wrote Hamlet?", "who is Claude?".

Question: "{q}"
Reply with only: true or false"""


def _ask_about_user(question: str) -> bool:
    """The yes/no call of the auto RAG selector (one small model call, ~0.1 s)."""
    try:
        reply = llm_invoke(_ABOUT_USER_PROMPT.format(user=USER_NAME, q=question)).strip().lower()
        return reply.startswith("true")
    except RateLimitError:
        raise
    except Exception as e:
        print(f"⚠️ about_user call failed ({e}) — using the relevance cut-off")
        return False

_pending_state: dict[str, "PendingState"] = {}

# ── Short-term memory ───────────────────────────────────────────────────────────────
# Per device, the last SHORT_TERM_TURNS exchanges (question + answer), so follow-up
# questions work ("do I have a brother?" → "where does he live?"). Kept in memory only:
# forgotten after SHORT_TERM_TTL_SECONDS of silence on that device, and on restart.
# (Replaces the former 20-turn summarizer, which never triggered in practice and pasted
# unchecked "learned facts" — one of them wrong — into every prompt.)
SHORT_TERM_TURNS = 3
SHORT_TERM_TTL_SECONDS = 3600
SHORT_TERM_MAX_CHARS = 300          # per question / answer in the prompt
_conversation_contexts: dict[str, dict] = {}   # device → {"turns": [...], "last_activity": ts}


def _pending() -> "PendingState":
    device = _current_device.get()
    if device not in _pending_state:
        _pending_state[device] = PendingState()
    return _pending_state[device]


def _ctx() -> dict:
    """This device's short-term memory; emptied first if it has been silent too long."""
    device = _current_device.get()
    ctx = _conversation_contexts.setdefault(device, {"turns": [], "last_activity": 0.0})
    if ctx["turns"] and time.time() - ctx["last_activity"] > SHORT_TERM_TTL_SECONDS:
        print(f"🧹 Short-term memory of '{device}' forgotten after "
              f"{(time.time() - ctx['last_activity']) / 60:.0f} min of silence")
        ctx["turns"] = []
    return ctx


def _remember(question: str, answer: str) -> None:
    """Store an exchange in this device's short-term memory (keeps the last SHORT_TERM_TURNS)."""
    ctx = _ctx()
    ctx["turns"] = (ctx["turns"] + [{"question": question, "answer": answer}])[-SHORT_TERM_TURNS:]
    ctx["last_activity"] = time.time()


def clear_device_context(device_id: str) -> None:
    if device_id in _conversation_contexts:
        _conversation_contexts[device_id]["turns"] = []
    if device_id in _pending_state:
        _pending_state[device_id].clear()


def clear_device_pending(device_id: str) -> None:
    if device_id in _pending_state:
        _pending_state[device_id].clear()


def device_has_pending(device_id: str) -> bool:
    return device_id in _pending_state and _pending_state[device_id].action is not None



# ===================================
# PYDANTIC MODELS
# ===================================

VALID_INTENTS = [
    "create_reminder",
    "delete_reminder",
    "create_event",
    "delete_event",
    "save_personal_note",
    "create_draft",
    "create_list",
    "add_to_list",
    "remove_from_list",
    "delete_list",
    "view_reminders",
    "view_events",
    "view_agenda",
    "view_emails",
    "get_weather",
    "find_places",
    "web_search",
    "about_chatette",
    "control_bulbs",
    "set_alarm",
    "cast_youtube",
    "cast_channel",
    "cast_tv_power",
    "cast_volume",
    "cast_stop",
    "general",
]

VALID_CONFIDENCES = {"high", "medium", "low"}


class ClassificationResult(BaseModel):
    intent: str
    confidence: str
    extracted: dict

    @field_validator("intent")
    @classmethod
    def intent_must_be_valid(cls, v: str) -> str:
        if v not in VALID_INTENTS:
            return "general"
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: str) -> str:
        if v not in VALID_CONFIDENCES:
            return "medium"
        return v


class PendingState(BaseModel):
    """
    Mutable module-level state for the two-step confirmation flow.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    items: list | None = None
    action: str | None = None
    conflict: str | None = None
    line_to_delete: str | None = None
    event_data: CalendarEvent | dict | None = None
    due: str | None = None
    lang: str = "en"

    def clear(self) -> None:
        """Reset all fields to their defaults."""
        self.text = None
        self.items = None
        self.action = None
        self.conflict = None
        self.line_to_delete = None
        self.event_data = None
        self.due = None
        self.lang = "en"


# Per-device pending state — accessed via _pending() helper



class CalendarEventData(BaseModel):
    """Validated calendar event extracted from LLM output."""
    title: str
    start: str
    end: str
    description: str = ""
    attendees: list = []

    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Event title must not be empty")
        return v.strip()

    @field_validator("start", mode="before")
    @classmethod
    def validate_start(cls, v) -> str:
        return _parse_and_fix_dt(v)

    @field_validator("end", mode="before")
    @classmethod
    def validate_end(cls, v, info) -> str:
        if not v:
            start_str = info.data.get("start")
            if start_str:
                try:
                    start_dt = datetime.fromisoformat(start_str)
                    return _parse_and_fix_dt(None, fallback=start_dt + timedelta(hours=1))
                except Exception:
                    pass
        return _parse_and_fix_dt(v)


# ===================================
# LLM INTENT CLASSIFIER
# ===================================

def classify_intent(question: str, lang: str = "en") -> ClassificationResult:
    """
    Classify the user's intent and extract relevant data in one LLM call.
    Returns a validated ClassificationResult.
    """
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    prompt = f"""Today is {current_date}.
{previous_context}
You are an intent classifier for a personal AI assistant called Chatette.

User message: "{question}"

Classify this message into exactly one intent and extract relevant data.

Valid intents:
- create_reminder: user wants to save a reminder (e.g. "remind me to...", "write down...", "note that...")
- delete_reminder: user wants to delete a reminder (e.g. "delete the reminder...", "forget about...")
- create_event: user wants to add a calendar event (e.g. "add to my calendar", "schedule a meeting...")
- delete_event: user wants to remove a calendar event (e.g. "cancel my appointment", "remove the event...", "can you delete the...")
- save_personal_note: user wants to save to personal notes/diary (e.g. "add to my diary", "save to personal notes...")
- create_draft: user wants to compose a document/email/letter (e.g. "compose a...", "write a draft...", "help me write...")
- create_list: user wants to create a new list (e.g. "create a shopping list", "make a list...")
- add_to_list: user wants to add an item to an existing list (e.g. "add milk to my shopping list")
- remove_from_list: user wants to remove an item from a list (e.g. "remove milk from the list")
- delete_list: user wants to delete an entire list (e.g. "delete my shopping list")
- view_reminders: user wants to see their saved reminders or to-do items (e.g. "show my reminders", "what do I have to do?", "what did I write down?")
- view_events: user wants to see their calendar events or appointments (e.g. "do I have anything this week?", "what's on my calendar?", "any appointments coming up?", "what are my plans for tomorrow?"). Do NOT use for questions about past activities, training history, or personal habits — those are general.
- view_agenda: user wants a combined overview of reminders AND events — a quick picture of what is coming up (e.g. "what do I have going on?", "give me my agenda", "what's coming up this week?", "anything important soon?"). Do NOT use for personal fact questions like "am I married?", "what's my height?", "where was I born?" — those are general.
- view_emails: user wants to see recent emails (e.g. "any new emails?", "check my inbox", "what emails did I get?", "did I get any emails?")
- get_weather: user wants weather info (e.g. "what's the weather?", "will it rain today?", "forecast for Berlin")
- find_places: user wants to find nearby places, venues or businesses (e.g. "good Chinese restaurant nearby", "find me a pharmacy", "any sushi places close by?", "recommend a cafe near me")
- web_search: the user EXPLICITLY asks to search the web or internet ("search the web for…", "google…", "look it up online", "such im Internet nach…", "busca en internet…"). Only when they explicitly ask for a web search — an ordinary question about news or facts is general.
- about_chatette: user is asking about Chatette (e.g. "who are you?", "what can you do?")
- control_bulbs: user wants to control a smart light bulb (e.g. "turn on the lights", "dim to 50%", "set color to blue", "red lights on", "turn off the lamp")
- set_alarm: user wants to set a countdown timer ("set a timer for 10 minutes", "remind me in 30 seconds") OR an alarm at a specific time ("set an alarm for 7 AM", "wake me up at 6:30", "alarm at 20:00")
- cast_youtube: user explicitly asks to play or cast something to the TV right now ("play Bonobo on YouTube", "put on some jazz on the TV", "cast this to the TV", "play Falling Down on YouTube"). Do NOT use this for wishes or statements like "I want to watch X" or "I'd like to see X" — those are general conversation.
- cast_channel: user wants to watch a live TV channel ("I want to watch ARD", "put on ZDF", "turn on Euronews", "open Arte", "put on Milenio", "TV5 Monde", "NHK World")
- cast_tv_power: user wants to turn the TV on or off ("turn on the TV", "turn off the TV", "switch the TV off")
- cast_volume: user wants to change TV volume ("volume up", "volume down", "set volume to 40", "louder", "quieter", "mute")
- cast_stop: user wants to stop TV playback ("stop the TV", "pause the TV", "stop casting")
- general: anything else — general questions, conversation, advice, or personal fact questions about the user (e.g. "am I married?", "where was I born?", "what's my height?", "do I have allergies?", "have I been to France?")

IMPORTANT: Only choose an action intent (create_reminder, create_event, save_personal_note, create_list, ...) when the user explicitly asks you to DO that action ("remind me", "add", "schedule", "save", "write down", "put in my calendar"). Merely mentioning a future event, appointment or plan is NOT a request to save it. If the user shares something and asks a question, wants advice, tips, ideas or an opinion, choose general.

For the extracted field, include only what is relevant.
IMPORTANT: If the user mentions multiple items of the same type, extract ALL of them as an array.

- create_reminder: {{"items": ["reminder text with any date/time kept inside", ...]}} — ALWAYS use items array; KEEP all date/time expressions inside each item (e.g. "Pay photographer tomorrow", never strip dates out)
- delete_reminder: {{"text": "the reminder description"}}
- create_event: {{"events": [{{"title": "...", "description": "..."}}, ...]}} — ALWAYS use events array, even for one event
- delete_event: {{"title": "event title"}}
- save_personal_note: {{"items": ["note 1", "note 2", ...]}} — ALWAYS use items array
- create_draft: {{"type": "email/letter/message", "purpose": "what it's about"}}
- create_list: {{"title": "list name", "items": ["item1", "item2"]}}
- add_to_list: {{"items": ["item1", "item2", ...], "list_name": "target list"}} — ALWAYS use items array
- remove_from_list: {{"item": "item name", "list_name": "target list"}}
- delete_list: {{"list_name": "list to delete"}}
- get_weather: {{"city": "city name or empty for home", "timeframe": "now|today|tomorrow|week|N days (e.g. \"3 days\")"}}
- find_places: {{"place_type": "restaurant|cafe|bar|pharmacy|supermarket|museum|gym|...", "cuisine": "chinese|italian|japanese|... or empty"}}
- view_emails: {{"max_results": 5}} — optional, default 5
- web_search: {{"query": "what to search for"}}
- control_bulbs: {{}}
- set_alarm: for a countdown use {{"seconds": <total seconds>}}; for a specific time use {{"hour": <0-23>, "minute": <0-59>}}
- cast_youtube: {{"query": "search query string"}}
- cast_channel: {{"channel": "ard"|"zdf"|"euronews"|"arte_fr"|"arte_de"|"milenio"|"tv5monde"|"nhk"}}
- cast_tv_power: {{"action": "on"|"off"}}
- cast_volume: {{"level": 0-100}} for absolute, or {{"delta": -100 to 100}} for relative (e.g. "louder" → +10, "quieter" → -10)
- cast_stop: {{}}
- view_reminders / view_events / view_agenda / about_chatette / general: {{}}

Examples of multi-item extraction:
User: "remind me to buy flowers, call mom and visit uncle"
→ {{"items": ["Buy flowers", "Call mom", "Visit uncle"]}}

User: "save reminder due tomorrow 'transfer 120 euros to photographer'"
→ {{"items": ["Transfer 120 euros to photographer tomorrow"]}}

User: "remind me on Friday to send the invoice"
→ {{"items": ["Send the invoice on Friday"]}}

User: "add milk, eggs and bread to my shopping list"
→ {{"items": ["Milk", "Eggs", "Bread"], "list_name": "shopping"}}

User: "schedule a dentist appointment on Monday and a haircut on Wednesday"
→ {{"events": [{{"title": "Dentist appointment", "description": ""}}, {{"title": "Haircut", "description": ""}}]}}

User: "have I been training for the marathon?" (previous topic was marathon date)
→ {{"intent": "general", "confidence": "high", "extracted": {{}}}}

User: "am I married?" or "where was I born?" or "do I have allergies?"
→ {{"intent": "general", "confidence": "high", "extracted": {{}}}}

User: "I'm having surgery in two days. Do you have any tips?" or "my flight is on Monday, what should I pack?"
→ {{"intent": "general", "confidence": "high", "extracted": {{}}}}  (asking for advice — nothing to save)

User: "I have a dentist appointment on Friday, put it in my calendar"
→ {{"intent": "create_event", "confidence": "high", "extracted": {{"events": [{{"title": "Dentist appointment", "description": ""}}]}}}}

Reply with ONLY a valid JSON object, nothing else:
{{
  "intent": "intent_name",
  "confidence": "high|medium|low",
  "extracted": {{...}}
}}"""

    try:
        response = llm_invoke(prompt)
        # Strip <think>...</think> blocks produced by reasoning models (e.g. Qwen3)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response[response.find("{"):response.rfind("}")+1]
        raw = json.loads(response)

        result = ClassificationResult(
            intent=raw.get("intent", "general"),
            confidence=raw.get("confidence", "medium"),
            extracted=raw.get("extracted", {})
        )
        print(f"🧠 Intent: {result.intent} [{result.confidence}] — {result.extracted}")
        return result

    except (json.JSONDecodeError, ValidationError, Exception) as e:
        print(f"⚠️ Intent classification failed: {e} — falling back to general")
        return ClassificationResult(intent="general", confidence="low", extracted={})


# ===================================
# CONVERSATION CONTEXT HELPER
# ===================================

def _normalize_question(text: str | None) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", (text or "").lower()).split())


def _shorten(text: str, limit: int = SHORT_TERM_MAX_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _build_conversation_context() -> str:
    """The recent conversation (short-term memory) for the prompt, oldest first.

    A wrong earlier answer must not outvote the user's notes: the small model tends to
    repeat whatever it said before (e.g. "no siblings" even with "brother Tom" in the
    context). So the exchanges are labelled as possibly wrong, and when the user repeats
    a question — usually a sign the answer was wrong — that earlier exchange is left out.
    """
    turns = _ctx()["turns"]
    current = _normalize_question(_current_question.get())
    kept = [t for t in turns if _normalize_question(t["question"]) != current]
    if len(kept) < len(turns):
        print("↪️ Same question asked again — its previous answer left out of the prompt")
    if not kept:
        return ""
    lines = "\n".join(f"User: {_shorten(t['question'])}\nChatette: {_shorten(t['answer'])}" for t in kept)
    return (
        "Recent conversation, oldest first (only for follow-up questions; it may contain "
        "mistakes — if it contradicts the notes or context below, trust the notes and context):\n"
        f"{lines}\n\n"
    )


def _persona_prompt(lang: str = "en") -> str:
    """Return the base persona prompt with user info filled in (+ computed date facts
    for dates in the profile, e.g. days until the next birthday)."""
    return CHATETTE_PERSONA.format(
        user_name=USER_NAME,
        user_profile=USER_PROFILE,
        lang_name=_LANG_NAMES.get(lang, "English")
    ) + _date_facts(USER_PROFILE or "")


# ===================================
# HANDLERS
# ===================================

def _normalise_items(extracted: dict, key: str = "items", fallback_key: str = "text") -> list:
    """Normalize extracted data to always return a list of strings."""
    items = extracted.get(key)
    if items and isinstance(items, list):
        return [str(i).strip() for i in items if str(i).strip()]
    text = extracted.get(fallback_key, "")
    return [text.strip()] if text.strip() else []


def handle_reminder(question: str, extracted: dict, lang: str = "en") -> str:
    """Extract one or more reminders, check conflicts, confirm as a batch."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    all_reminders = get_all_reminders()
    previous_context = _build_conversation_context()

    items = _normalise_items(extracted, key="items", fallback_key="text")

    if not items:
        extraction_prompt = f"""Today is {current_date}.
{previous_context}
The user wants to save reminders: "{question}"

Extract ALL reminder items mentioned. Return ONLY a JSON array of strings.
- Remove trigger phrases like "write down", "remind me", "make a note", "save a reminder"
- Keep ALL date/time expressions in the text (e.g. "tomorrow", "due Friday", "on April 3rd", "next week")
- If a date appears before the task (e.g. "due tomorrow to wash clothes"), move it after: "Wash clothes tomorrow"
- One item per reminder, no explanations

Examples:
"remind me to buy flowers, call mom and visit uncle"
→ ["Buy flowers", "Call mom", "Visit uncle"]

"write down dentist next Friday"
→ ["Dentist next Friday"]

"save a reminder due tomorrow to wash white clothes and pick up the cake"
→ ["Wash white clothes tomorrow", "Pick up the cake tomorrow"]

Output (JSON array only):"""
        try:
            response = llm_invoke(extraction_prompt).strip()
            response = response[response.find("["):response.rfind("]")+1]
            items = json.loads(response)
            items = [str(i).strip() for i in items if str(i).strip()]
        except Exception:
            items = [question.strip()]

    if not items:
        return _t(
            "I couldn't figure out what to save. Can you rephrase?",
            "Ich konnte nicht verstehen, was gespeichert werden soll. Kannst du es umformulieren?",
            "No entendí qué guardar. ¿Puedes reformularlo?",
            lang
        )

    # Extract due dates separately from reminder text
    due_prompt = f"""Today is {current_date}.
For each reminder, split the task text from its due date.

Reminders: {json.dumps(items)}

Return a JSON array with one object per reminder:
- "text": the task only, no date/time phrases
- "due": ISO date string "YYYY-MM-DD" if a specific date is mentioned, otherwise null

Examples:
["Dentist next Friday"] → [{{"text": "Dentist", "due": "2026-03-27"}}]
["Buy milk", "Call mom on April 3rd"] → [{{"text": "Buy milk", "due": null}}, {{"text": "Call mom", "due": "2026-04-03"}}]

JSON array only:"""
    try:
        due_response = llm_invoke(due_prompt).strip()
        due_response = due_response[due_response.find("["):due_response.rfind("]")+1]
        due_items = json.loads(due_response)
        due_items = [
            {"text": str(d.get("text", items[i])).strip() or items[i], "due": d.get("due")}
            for i, d in enumerate(due_items)
            if i < len(items)
        ]
        # Pad if LLM returned fewer items than expected
        for i in range(len(due_items), len(items)):
            due_items.append({"text": items[i], "due": None})
    except Exception:
        due_items = [{"text": item, "due": None} for item in items]

    if len(due_items) == 1:
        reminder_text = due_items[0]["text"]
        due_date = due_items[0]["due"]
        if all_reminders != "No reminders found.":
            conflict_prompt = f"""Check this list of reminders for duplicates or conflicts.

New reminder: "{reminder_text}"

Existing reminders:
{all_reminders}

Rules:
- Only flag CONFLICT or DUPLICATE for reminders with a specific date or time
- Shopping items, tasks, general notes: always CLEAR
- DUPLICATE only if the exact same reminder already exists

Reply with one of:
- "DUPLICATE: <exact line>"
- "CONFLICT: <exact line>"
- "CLEAR"
"""
            conflict_check = llm_invoke(conflict_prompt).strip()
            print(f"Conflict check: '{conflict_check}'")

            if conflict_check.startswith("DUPLICATE:"):
                existing_line = conflict_check.replace("DUPLICATE:", "").strip()
                _pending().text = reminder_text
                _pending().items = None
                _pending().action = "save"
                _pending().conflict = "duplicate"
                _pending().line_to_delete = existing_line
                _pending().due = due_date
                _pending().lang = lang
                return _t(
                    f"Heads up — you already have something similar: '{existing_line}'. Want to replace it with '{reminder_text}'?",
                    f"Hinweis — du hast bereits etwas Ähnliches: '{existing_line}'. Ersetzen mit '{reminder_text}'?",
                    f"Aviso — ya tienes algo similar: '{existing_line}'. ¿Reemplazar con '{reminder_text}'?",
                    lang
                )
            elif conflict_check.startswith("CONFLICT:"):
                existing_line = conflict_check.replace("CONFLICT:", "").strip()
                _pending().text = reminder_text
                _pending().items = None
                _pending().action = "save"
                _pending().conflict = "conflict"
                _pending().line_to_delete = existing_line
                _pending().due = due_date
                _pending().lang = lang
                return _t(
                    f"Quick one — you already have '{existing_line}' around that time. Still want to save '{reminder_text}'?",
                    f"Kurze Frage — du hast bereits '{existing_line}' zu dieser Zeit. Trotzdem '{reminder_text}' speichern?",
                    f"Un momento — ya tienes '{existing_line}' a esa hora. ¿Guardar '{reminder_text}' igualmente?",
                    lang
                )

        _pending().text = reminder_text
        _pending().items = None
        _pending().action = "save"
        _pending().conflict = None
        _pending().line_to_delete = None
        _pending().due = due_date
        _pending().lang = lang
        due_hint = f" (due {_fmt_due(due_date, lang)})" if due_date else ""
        return _t(
            f"Just to confirm — save this to reminders: '{reminder_text}'{due_hint}?",
            f"Zur Bestätigung — in die Erinnerungen speichern: '{reminder_text}'{due_hint}?",
            f"Para confirmar — ¿guardar esto en recordatorios: '{reminder_text}'{due_hint}?",
            lang
        )

    preview = "\n".join(
        f"  • {d['text']}" + (f" (due {_fmt_due(d['due'], lang)})" if d.get("due") else "")
        for d in due_items
    )
    _pending().text = None
    _pending().items = due_items
    _pending().action = "save_batch"
    _pending().conflict = None
    _pending().line_to_delete = None
    _pending().due = None
    _pending().lang = lang
    return _t(
        f"Save these {len(due_items)} reminders?\n{preview}",
        f"Diese {len(due_items)} Erinnerungen speichern?\n{preview}",
        f"¿Guardar estos {len(due_items)} recordatorios?\n{preview}",
        lang
    )


def handle_delete(question: str, extracted: dict, lang: str = "en") -> str:
    """Find matching reminder and ask confirmation."""
    items = get_reminders_list()
    if not items:
        return _t(
            "Nothing on your reminder list.",
            "Keine Erinnerungen vorhanden.",
            "No tienes recordatorios.",
            lang
        )

    # "delete all reminders" / "alle Erinnerungen löschen" / "borra todos los recordatorios"
    if re.search(r"\b(all|every|everything|alle|todos|todas)\b", question.lower()):
        _pending().text = None
        _pending().action = "delete_all"
        _pending().conflict = None
        _pending().line_to_delete = None
        _pending().lang = lang
        return _t(
            f"Delete all {len(items)} reminder{'s' if len(items) != 1 else ''}?",
            f"Alle {len(items)} Erinnerungen löschen?",
            f"¿Eliminar los {len(items)} recordatorios?",
            lang
        )

    hint = (extracted.get("text") or "").strip().lower()

    # 1. Deterministic match on the classifier hint (substring, then fuzzy)
    matched = None
    if hint:
        subs = [i for i in items if hint in i.text.lower() or i.text.lower() in hint]
        if len(subs) == 1:
            matched = subs[0]
        else:
            import difflib
            scored = sorted(
                ((difflib.SequenceMatcher(None, hint, i.text.lower()).ratio(), i) for i in items),
                key=lambda s: s[0], reverse=True
            )
            if scored[0][0] >= 0.6:
                matched = scored[0][1]

    # 2. Fall back to the LLM — ask for a number, not free text
    if matched is None:
        previous_context = _build_conversation_context()
        numbered = "\n".join(
            f"{n}. {i.text}" + (f" (due: {i.due})" if i.due else "")
            for n, i in enumerate(items, 1)
        )
        match_prompt = f"""Find the reminder the user wants to delete.
{previous_context}
The user said: "{question}"
Hint from classifier: "{hint}"

Reminders:
{numbered}

- Find the best match, even if wording differs
- If the user says "that" or "it", use the previous exchange
- Return ONLY the number of the matching reminder
- If nothing matches, return 0"""

        answer = llm_invoke(match_prompt).strip()
        print(f"LLM matched: '{answer}'")
        m = re.search(r"\d+", answer)
        idx = int(m.group()) if m else 0
        if 1 <= idx <= len(items):
            matched = items[idx - 1]

    if matched is None:
        return _t(
            "Nothing matching that in your reminders. Can you be more specific?",
            "Keine passende Erinnerung gefunden. Kannst du genauer sein?",
            "No encontré un recordatorio que coincida. ¿Puedes ser más específico?",
            lang
        )

    matched_line = matched.text
    _pending().text = None
    _pending().action = "delete"
    _pending().conflict = None
    _pending().line_to_delete = matched_line
    _pending().lang = lang
    return _t(
        f"Delete this one: '{matched_line}'?",
        f"Diese Erinnerung löschen: '{matched_line}'?",
        f"¿Eliminar este recordatorio: '{matched_line}'?",
        lang
    )


def handle_delete_event(question: str, extracted: dict, lang: str = "en") -> str:
    """Find matching calendar event and ask confirmation."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    try:
        # Uses DELETE_EVENT_DAYS_AHEAD (default 60) and CALENDAR_DAYS_BEHIND from .env
        events = get_upcoming_events(
            days_ahead=DELETE_EVENT_DAYS_AHEAD,
            days_behind=CALENDAR_DAYS_BEHIND
        )
    except Exception as e:
        print(f"Could not fetch events: {e}")
        return _t(
            "I couldn't connect to your Google Calendar. Is the server online?",
            "Ich konnte keine Verbindung zu deinem Google Kalender herstellen.",
            "No pude conectarme a tu Google Calendar. ¿Está el servidor activo?",
            lang
        )

    if not events:
        return _t(
            "No upcoming events found in your calendar.",
            "Keine bevorstehenden Termine in deinem Kalender gefunden.",
            "No encontré eventos próximos en tu calendario.",
            lang
        )

    events_summary = "\n".join([
        f"- ID: {e.id} | {e.title} on {e.start}"
        for e in events
    ])

    hint = extracted.get("title", "")

    match_prompt = f"""Today is {current_date}.
{previous_context}
The user wants to delete a calendar event: "{question}"
Hint from classifier: "{hint}"

Upcoming events:
{events_summary}

- Find the event that best matches
- If the user says "that" or "it", use the previous exchange
- Return ONLY the event ID and title in this exact format: ID|Title
- If nothing matches clearly, return NO_MATCH

Your answer:"""

    matched = llm_invoke(match_prompt).strip().strip('"').strip("'")
    print(f"LLM matched event: '{matched}'")

    if "NO_MATCH" in matched or not matched or "|" not in matched:
        return _t(
            "Couldn't find that event. Can you be more specific?",
            "Ich konnte diesen Termin nicht finden. Kannst du genauer sein?",
            "No encontré ese evento. ¿Puedes ser más específico?",
            lang
        )

    parts = matched.split("|", 1)
    event_id = parts[0].strip()
    event_title = parts[1].strip() if len(parts) > 1 else event_id

    _pending().text = event_title
    _pending().action = "delete_event"
    _pending().conflict = None
    _pending().line_to_delete = event_id
    _pending().event_data = {"id": event_id, "title": event_title}
    _pending().lang = lang
    return _t(
        f"Delete '{event_title}' from your Google Calendar?",
        f"'{event_title}' aus deinem Google Kalender löschen?",
        f"¿Eliminar '{event_title}' de tu Google Calendar?",
        lang
    )


def handle_personal_note(question: str, extracted: dict, lang: str = "en") -> str:
    """Extract one or more personal notes and confirm as batch."""
    previous_context = _build_conversation_context()

    items = _normalise_items(extracted, key="items", fallback_key="text")

    if not items:
        extraction_prompt = f"""The user said: "{question}"
{previous_context}

Extract ALL note items mentioned. Return ONLY a JSON array of strings.
- Remove trigger phrases like "add to my personal notes", "add to my diary"
- If multiple notes are mentioned, return all of them
- Return ONLY the note texts

Output (JSON array only):"""
        try:
            response = llm_invoke(extraction_prompt).strip()
            response = response[response.find("["):response.rfind("]")+1]
            items = json.loads(response)
            items = [str(i).strip() for i in items if str(i).strip()]
        except Exception:
            items = [question.strip()]

    if not items:
        return _t(
            "I couldn't figure out what to save. Can you rephrase?",
            "Ich konnte nicht verstehen, was gespeichert werden soll.",
            "No entendí qué guardar. ¿Puedes reformularlo?",
            lang
        )

    if len(items) == 1:
        note_text = items[0]
        _pending().text = note_text
        _pending().items = None
        _pending().action = "personal_note"
        _pending().conflict = None
        _pending().line_to_delete = None
        _pending().event_data = None
        _pending().lang = lang
        return _t(
            f"Add this to your personal notes: '{note_text}'?",
            f"Das zu deinen persönlichen Notizen hinzufügen: '{note_text}'?",
            f"¿Añadir esto a tus notas personales: '{note_text}'?",
            lang
        )

    preview = "\n".join(f"  • {item}" for item in items)
    _pending().text = None
    _pending().items = items
    _pending().action = "personal_note_batch"
    _pending().conflict = None
    _pending().line_to_delete = None
    _pending().event_data = None
    _pending().lang = lang
    return _t(
        f"Add these {len(items)} notes to your personal notes?\n{preview}",
        f"Diese {len(items)} Einträge zu deinen persönlichen Notizen hinzufügen?\n{preview}",
        f"¿Añadir estos {len(items)} elementos a tus notas personales?\n{preview}",
        lang
    )


# ===================================
# DRAFTS — guided flow
# ===================================
# A chat draft is a short conversation; its state lives in _pending():
#   draft_offer   → Chatette restated the request and asked whether she may ask
#                   a few questions / add suggestions, or just write it.
#   draft_clarify → she asked clarifying questions and waits for the answers.
#   draft         → she showed a draft and waits for save / discard / changes.
# Quick requests, the Draft shortcut (/command) and voice go straight to "draft".
# event_data holds the request, so every step writes from the user's own words.

DRAFT_STAGES = ("draft_offer", "draft_clarify", "draft")

_QUOTED_FILENAME_RE = re.compile(r"""["'“‘]([^"'“”‘’\n]+?\.(?:txt|md|docx?|rtf))["'”’]""", re.IGNORECASE)
_BARE_FILENAME_RE = re.compile(r"\b([\w\-.]+\.(?:txt|md|docx?|rtf))\b", re.IGNORECASE)

# Explicit "don't ask me anything" wording — the only thing that skips the questions step.
_QUICK_DRAFT_RE = re.compile(
    r"\b(quick(ly)?|just (write|draft|give)|no questions|without (any )?questions|straight away|right away|"
    r"kurz(e|en|er)?|schnell|einfach (nur )?schreiben|ohne (rück)?fragen|"
    r"rápid[oa]|rapid[oa]|sin preguntas|directamente)\b",
    re.IGNORECASE)

_DRAFT_CANCEL = {"cancel", "stop", "never mind", "nevermind", "forget it", "abbrechen", "vergiss es",
                 "cancelar", "olvídalo", "olvidalo"}
_DRAFT_YES = {"yes", "yep", "yeah", "sure", "ok", "okay", "please", "go ahead", "ja", "klar", "gerne",
              "sí", "si", "claro", "vale", "dale"}
_DRAFT_NO = {"no", "nope", "nein", "no thanks", "nein danke", "no gracias"}


def _parse_json(text: str, opener: str = "{", closer: str = "}"):
    start, end = text.find(opener), text.rfind(closer)
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _requested_filename(text: str) -> str | None:
    """Filename the user asked for, e.g. "name it 'christmas-party-idea.txt'"."""
    m = _QUOTED_FILENAME_RE.search(text) or _BARE_FILENAME_RE.search(text)
    return m.group(1).strip() if m else None


def _short_reply(text: str) -> str:
    """Lower-cased reply without punctuation, for matching one-word answers."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _draft_state() -> dict:
    data = _pending().event_data
    return data if isinstance(data, dict) else {}


def _set_draft_stage(stage: str, state: dict, lang: str, content: str | None = None) -> None:
    _pending().action = stage
    _pending().text = content
    _pending().items = None
    _pending().conflict = None
    _pending().line_to_delete = state.get("title")
    _pending().event_data = state
    _pending().lang = lang


def _analyze_draft_request(request: str, previous_context: str, lang: str) -> dict:
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    prompt = f"""Today is {current_date}.
{previous_context}
The user asked their personal assistant for a draft:
\"\"\"{request}\"\"\"

Analyze the request. Return ONLY a JSON object with:
- "type": the kind of document (e.g. "email", "letter", "message", "plan", "document")
- "title": a short title for it, at most 6 words
- "summary": 1-2 sentences in {_LANG_NAMES.get(lang, "English")}, speaking to the user as "you", restating what they want with every concrete detail and constraint they gave (who, where, when, numbers, restrictions, tone). Do not add anything they did not say.

Output:"""
    info = _parse_json(llm_invoke(prompt)) or {}
    return {
        "type": str(info.get("type") or "document"),
        "title": str(info.get("title") or "draft"),
        "summary": str(info.get("summary") or "").strip(),
        "quick": bool(_QUICK_DRAFT_RE.search(request)),
    }


def _write_draft(state: dict, lang: str) -> str:
    """Write the draft from the user's request (+ clarifying Q&A if any)."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    lang_name = _LANG_NAMES.get(lang, "English")
    qa = ""
    if state.get("questions"):
        qa = ("\nYou asked the user:\n" + "\n".join(f"- {q}" for q in state["questions"]) +
              f"\nThe user answered:\n\"\"\"{state.get('answers', '')}\"\"\"\n")
    if state.get("suggestions"):
        fill = ("- Where details are still open, fill them with concrete, creative suggestions of your own "
                "that fit every constraint above.")
    else:
        fill = ("- Do not add ideas the user did not ask for. Use [placeholder] only for details that are "
                "genuinely needed and missing.")
    prompt = f"""Today is {current_date}.
{state.get("context", "")}
You are drafting a {state.get("type", "document")} on behalf of {USER_NAME}, the user. {USER_NAME} is the author/sender — never the recipient. When a sign-off is needed, sign as {USER_NAME}; address recipients the way the user described them.

The user's request:
\"\"\"{state["request"]}\"\"\"
{qa}
Rules:
- Follow every detail and constraint in the request{" and the answers" if qa else ""} exactly, and never contradict them (if they rule something out, leave it out; if the space or budget is small, keep every suggestion realistic for it).
- Parts of the request that are instructions to you (e.g. "quick", "no questions", "name the file …", "imagine you are …") are not content — never repeat them in the document.
- Do not invent facts about the user's situation (who it is for, company, budget, venue, date) that they did not give. Today's date is only for reference — do not use it as the date of the event or document unless the user said so.
{fill}
- Write entirely in {lang_name}, well structured and to the point.
- Output only the document itself — no introduction, closing remarks or file name.

The {state.get("type", "document")}:"""
    return llm_invoke(prompt).strip()


def _draft_questions(state: dict, lang: str) -> list[str]:
    prompt = f"""The user asked their personal assistant for a draft:
\"\"\"{state["request"]}\"\"\"

Before writing it, list the 3 to 6 questions whose answers would make this {state.get("type", "document")} noticeably better.
- Only ask about things the user has NOT already said.
- Keep each question short and specific.
- Write them in {_LANG_NAMES.get(lang, "English")}.
Return ONLY a JSON array of strings.

Output:"""
    raw = llm_invoke(prompt)
    questions = _parse_json(raw, "[", "]")
    if not isinstance(questions, list) or not questions:
        # Fallback: one question per non-empty line, numbering stripped
        questions = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip() for l in raw.splitlines()]
    return [str(q).strip() for q in questions if str(q).strip()][:6]


def _classify_draft_reply(stage: str, reply: str) -> str:
    """Interpret the user's reply to the offer ("quick"/"suggest"/"cancel")
    or to the save question ("save"/"discard"/"revise")."""
    short = _short_reply(reply)
    if short in _DRAFT_CANCEL:
        return "cancel" if stage == "draft_offer" else "discard"
    if short in _DRAFT_YES:
        return "suggest" if stage == "draft_offer" else "save"
    if short in _DRAFT_NO:
        return "quick" if stage == "draft_offer" else "discard"

    if stage == "draft_offer":
        asked = "whether she may ask a few questions and add her own suggestions before writing the draft, or just write it now"
        options = ('"suggest" — the user agrees to questions/suggestions (or adds more details)\n'
                   '"quick" — the user declines and just wants the draft now\n'
                   '"cancel" — the user no longer wants the draft at all')
        fallback = "suggest"
    else:
        asked = "whether to save the draft she just showed"
        options = ('"save" — the user wants it saved as it is\n'
                   '"discard" — the user does not want it saved\n'
                   '"revise" — the user asks for changes to the draft')
        fallback = "revise"
    prompt = f"""The assistant asked the user {asked}.
The user replied: "{reply}"

Which one is it?
{options}
Return ONLY a JSON object like {{"decision": "..."}}.

Output:"""
    decision = (_parse_json(llm_invoke(prompt)) or {}).get("decision", "")
    valid = ("suggest", "quick", "cancel") if stage == "draft_offer" else ("save", "discard", "revise")
    return decision if decision in valid else fallback


def _present_draft(state: dict, content: str, lang: str, updated: bool = False) -> str:
    _set_draft_stage("draft", state, lang, content)
    if updated:
        intro = _t("Here's the updated version:", "Hier ist die überarbeitete Version:", "Aquí está la versión actualizada:", lang)
    else:
        intro = _t("Got it — here's the draft:", "Alles klar — hier ist der Entwurf:", "Entendido — aquí está el borrador:", lang)
    name = state.get("filename")
    if name:
        ask_save = _t(
            f"Want me to save it as '{name}'? You can also tell me what to change.",
            f"Soll ich es als '{name}' speichern? Du kannst mir auch sagen, was ich ändern soll.",
            f"¿Lo guardo como '{name}'? También puedes decirme qué cambiar.",
            lang)
    else:
        ask_save = _t(
            "Want me to write this down? You can also tell me what to change.",
            "Soll ich das speichern? Du kannst mir auch sagen, was ich ändern soll.",
            "¿Quieres que lo guarde? También puedes decirme qué cambiar.",
            lang)
    return f"{intro}\n\n{content}\n\n{ask_save}"


def handle_draft(question: str, extracted: dict, lang: str = "en", guided: bool = False) -> str:
    """Start a draft. guided=True (typed chat) runs the offer → questions → draft flow;
    otherwise (quick request, Draft shortcut, voice) the draft is written right away."""
    request = question.strip() or extracted.get("purpose", "")
    previous_context = _build_conversation_context()
    info = _analyze_draft_request(request, previous_context, lang)
    state = {
        "request": request,
        "context": previous_context,
        "type": info["type"],
        "title": info["title"],
        "filename": _requested_filename(request),
        "questions": [],
        "answers": "",
        "suggestions": False,
    }
    if state["filename"]:
        state["filename"] = os.path.splitext(state["filename"])[0] + ".txt"

    if not guided or info["quick"] or not info["summary"]:
        return _present_draft(state, _write_draft(state, lang), lang)

    _set_draft_stage("draft_offer", state, lang)
    return info["summary"] + "\n\n" + _t(
        "Before I write it — may I ask a few questions and add some suggestions of my own? "
        "Or should I just write it with what you've given me?",
        "Bevor ich loslege — darf ich ein paar Fragen stellen und eigene Vorschläge einbringen? "
        "Oder soll ich es einfach mit deinen Angaben schreiben?",
        "Antes de escribirlo — ¿puedo hacerte algunas preguntas y añadir sugerencias propias? "
        "¿O lo escribo directamente con lo que me has dado?",
        lang)


def handle_draft_followup(reply: str, lang: str = "en") -> str:
    """Continue a draft conversation started by handle_draft()."""
    stage = _pending().action
    state = _draft_state()
    lang = _pending().lang or lang
    if not state.get("request"):
        _pending().clear()
        return _t("I lost track of that draft — could you ask again?",
                  "Ich habe den Entwurf verloren — kannst du nochmal fragen?",
                  "Perdí el borrador — ¿puedes pedírmelo de nuevo?", lang)

    dropped = _t("No problem — I've dropped the draft.",
                 "Kein Problem — ich verwerfe den Entwurf.",
                 "Sin problema — descarto el borrador.", lang)

    if stage == "draft_offer":
        decision = _classify_draft_reply(stage, reply)
        if decision == "cancel":
            _pending().clear()
            return dropped
        if decision == "quick":
            return _present_draft(state, _write_draft(state, lang), lang)
        # Accepted — keep any extra details they added in the same breath.
        if len(_short_reply(reply).split()) > 3:
            state["request"] += "\n" + reply.strip()
        state["questions"] = _draft_questions(state, lang)
        state["suggestions"] = True
        if not state["questions"]:
            return _present_draft(state, _write_draft(state, lang), lang)
        _set_draft_stage("draft_clarify", state, lang)
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(state["questions"], 1))
        return _t(
            f"Great — a few questions first:\n\n{numbered}\n\nAnswer whichever you like; I'll fill in the rest with my own suggestions.",
            f"Super — zuerst ein paar Fragen:\n\n{numbered}\n\nBeantworte, was du möchtest; den Rest ergänze ich mit eigenen Vorschlägen.",
            f"Genial — primero unas preguntas:\n\n{numbered}\n\nResponde lo que quieras; el resto lo completo con mis propias sugerencias.",
            lang)

    if stage == "draft_clarify":
        if _short_reply(reply) in _DRAFT_CANCEL:
            _pending().clear()
            return dropped
        state["answers"] = reply.strip()
        return _present_draft(state, _write_draft(state, lang), lang)

    # stage == "draft": save, discard or revise
    content = _pending().text or ""
    decision = _classify_draft_reply(stage, reply)
    if decision == "save":
        filename = save_draft(state.get("title") or "draft", content, filename=state.get("filename"))
        ingest_notes()
        _pending().clear()
        if not filename:
            return _t("Something went wrong saving the draft — try again?",
                      "Beim Speichern ist etwas schiefgelaufen — nochmal versuchen?",
                      "Algo salió mal al guardar el borrador — ¿intentamos de nuevo?", lang)
        return _t(
            f"Saved as '{filename}' in your drafts. You're all set.",
            f"Als '{filename}' in deinen Entwürfen gespeichert.",
            f"Guardado como '{filename}' en tus borradores.",
            lang)
    if decision == "discard":
        _pending().clear()
        return dropped

    revise_prompt = f"""You wrote this {state.get("type", "document")} on behalf of {USER_NAME} (the author/sender, never the recipient):
\"\"\"{content}\"\"\"

Their original request was:
\"\"\"{state["request"]}\"\"\"

They now ask for these changes:
\"\"\"{reply}\"\"\"

Rewrite the whole {state.get("type", "document")} with the changes applied. Keep everything else that still fits, and keep respecting every constraint from the original request unless the changes override it.
Write entirely in {_LANG_NAMES.get(lang, "English")}. Output only the document itself.

The {state.get("type", "document")}:"""
    return _present_draft(state, llm_invoke(revise_prompt).strip(), lang, updated=True)


def handle_create_list(question: str, extracted: dict, lang: str = "en") -> str:
    """Extract list title and items, confirm creation."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    title = extracted.get("title", "")
    items = extracted.get("items", [])

    if not title:
        extraction_prompt = f"""Today is {current_date}.
{previous_context}
The user said: "{question}"

Extract list title and items.
Return ONLY a JSON object:
- title: list name
- items: array of items (empty if none mentioned)

Output:"""
        response = llm_invoke(extraction_prompt).strip()
        response = response[response.find("{"):response.rfind("}")+1]
        try:
            list_data = json.loads(response)
            title = list_data.get("title", "New List")
            items = list_data.get("items", [])
        except json.JSONDecodeError:
            title = "New List"
            items = []

    items_preview = ", ".join(items) if items else "empty for now"

    _pending().text = title
    _pending().action = "create_list"
    _pending().conflict = None
    _pending().line_to_delete = None
    _pending().event_data = {"title": title, "items": items}
    _pending().lang = lang
    return _t(
        f"Create a list called '{title}' — {items_preview}?",
        f"Eine Liste namens '{title}' erstellen — {items_preview}?",
        f"¿Crear una lista llamada '{title}' — {items_preview}?",
        lang
    )


def handle_add_to_list(question: str, extracted: dict, lang: str = "en") -> str:
    """Extract one or more items and target list, confirm adding."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return _t(
            "No lists yet — want to create one first?",
            "Noch keine Listen — möchtest du zuerst eine erstellen?",
            "Sin listas aún — ¿quieres crear una primero?",
            lang
        )

    lists_summary = "\n".join([f"- {l['filename']}" for l in all_lists])

    items = _normalise_items(extracted, key="items", fallback_key="item")
    list_name = extracted.get("list_name", "")

    if not items or not list_name:
        extraction_prompt = f"""The user said: "{question}"
{previous_context}
Available lists:
{lists_summary}

Extract ALL items to add and the target list.
Return ONLY a JSON object:
- items: array of item names (even if just one)
- list_name: target list name or filename

Output:"""
        response = llm_invoke(extraction_prompt).strip()
        response = response[response.find("{"):response.rfind("}")+1]
        try:
            data = json.loads(response)
            raw = data.get("items", [])
            if isinstance(raw, list):
                items = [str(i).strip() for i in raw if str(i).strip()]
            else:
                single = data.get("item", "")
                items = [single] if single else []
            list_name = data.get("list_name", list_name)
        except json.JSONDecodeError:
            return _t(
                "Didn't quite catch that — can you be more specific?",
                "Das habe ich nicht ganz verstanden — kannst du genauer sein?",
                "No entendí bien — ¿puedes ser más específico?",
                lang
            )

    if not items:
        return _t(
            "What would you like to add to the list?",
            "Was möchtest du zur Liste hinzufügen?",
            "¿Qué quieres añadir a la lista?",
            lang
        )

    filename = find_list_by_name(list_name)
    if not filename:
        return _t(
            f"No list matching '{list_name}' on file.",
            f"Keine Liste mit dem Namen '{list_name}' gefunden.",
            f"No encontré una lista llamada '{list_name}'.",
            lang
        )

    if len(items) == 1:
        item = items[0]
        _pending().text = item
        _pending().items = None
        _pending().action = "add_to_list"
        _pending().conflict = None
        _pending().line_to_delete = filename
        _pending().event_data = {"item": item, "filename": filename}
        _pending().lang = lang
        return _t(
            f"Add '{item}' to '{filename}'?",
            f"'{item}' zur Liste '{filename}' hinzufügen?",
            f"¿Añadir '{item}' a '{filename}'?",
            lang
        )

    preview = "\n".join(f"  • {i}" for i in items)
    _pending().text = None
    _pending().items = items
    _pending().action = "add_to_list_batch"
    _pending().conflict = None
    _pending().line_to_delete = filename
    _pending().event_data = {"items": items, "filename": filename}
    _pending().lang = lang
    return _t(
        f"Add these {len(items)} items to '{filename}'?\n{preview}",
        f"Diese {len(items)} Einträge zu '{filename}' hinzufügen?\n{preview}",
        f"¿Añadir estos {len(items)} elementos a '{filename}'?\n{preview}",
        lang
    )


def handle_remove_from_list(question: str, extracted: dict, lang: str = "en") -> str:
    """Find and remove a specific item from a list."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return _t("No lists on file.", "Keine Listen vorhanden.", "Sin listas en archivo.", lang)

    lists_with_items = []
    for l in all_lists:
        items = get_list_items(l['filename'])
        if items:
            item_names = [i.text for i in items]
            lists_with_items.append(f"{l['filename']}: {', '.join(item_names)}")

    if not lists_with_items:
        return _t("Your lists are all empty.", "Deine Listen sind alle leer.", "Tus listas están vacías.", lang)

    lists_summary = "\n".join(lists_with_items)

    item = extracted.get("item", "")
    list_name = extracted.get("list_name", "")

    if not item:
        extraction_prompt = f"""The user said: "{question}"
{previous_context}
Lists and their items:
{lists_summary}

Extract what item to remove and from which list.
Return ONLY a JSON object:
- item: the item text to remove
- filename: the exact list filename

Output:"""
        response = llm_invoke(extraction_prompt).strip()
        response = response[response.find("{"):response.rfind("}")+1]
        try:
            data = json.loads(response)
            item = data.get("item", "")
            list_name = data.get("filename", list_name)
        except json.JSONDecodeError:
            return _t(
                "Didn't catch that — which item and which list?",
                "Das habe ich nicht verstanden — welches Element und welche Liste?",
                "No entendí — ¿qué elemento y de qué lista?",
                lang
            )

    if not item:
        return _t(
            "Can you be more specific about what to remove and from which list?",
            "Kannst du genauer angeben, was entfernt werden soll und aus welcher Liste?",
            "¿Puedes ser más específico sobre qué eliminar y de qué lista?",
            lang
        )

    filename = find_list_by_name(list_name) if list_name else None

    matched = None
    if filename:
        list_items = get_list_items(filename)
        found = next(
            (i for i in list_items if item.lower() in i.text.lower() or
             i.text.lower() in item.lower()),
            None
        )
        if found:
            matched = {"text": found.text, "index": found.index, "filename": filename}
    else:
        for l in all_lists:
            list_items = get_list_items(l['filename'])
            found = next(
                (i for i in list_items if item.lower() in i.text.lower() or
                 i.text.lower() in item.lower()),
                None
            )
            if found:
                filename = l['filename']
                matched = {"text": found.text, "index": found.index, "filename": filename}
                break

    if not matched:
        return _t(
            f"Couldn't find '{item}' in any of your lists.",
            f"'{item}' wurde in keiner Liste gefunden.",
            f"No encontré '{item}' en ninguna lista.",
            lang
        )

    _pending().text = matched['text']
    _pending().action = "remove_from_list"
    _pending().conflict = None
    _pending().line_to_delete = filename
    _pending().event_data = {"filename": filename, "line_index": matched['index']}
    _pending().lang = lang
    return _t(
        f"Remove '{matched['text']}' from '{filename}'?",
        f"'{matched['text']}' aus '{filename}' entfernen?",
        f"¿Eliminar '{matched['text']}' de '{filename}'?",
        lang
    )


def handle_delete_list(question: str, extracted: dict, lang: str = "en") -> str:
    """Find and confirm deletion of a list."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return _t(
            "No lists to delete — all clear.",
            "Keine Listen zum Löschen vorhanden.",
            "Sin listas para eliminar.",
            lang
        )

    lists_summary = "\n".join([f"- {l['filename']}" for l in all_lists])
    hint = extracted.get("list_name", "")

    match_prompt = f"""The user wants to delete a list: "{question}"
{previous_context}
Hint from classifier: "{hint}"
Available lists:
{lists_summary}

Return ONLY the exact filename to delete.
If nothing matches, return NO_MATCH."""

    matched = llm_invoke(match_prompt).strip().strip('"').strip("'")

    if "NO_MATCH" in matched or not matched:
        return _t(
            "Nothing matching that in your lists — can you be more specific?",
            "Keine passende Liste gefunden — kannst du genauer sein?",
            "No encontré esa lista — ¿puedes ser más específico?",
            lang
        )

    _pending().text = matched
    _pending().action = "delete_list"
    _pending().conflict = None
    _pending().line_to_delete = matched
    _pending().event_data = None
    _pending().lang = lang
    return _t(
        f"About to delete '{matched}' — this can't be undone. You sure?",
        f"'{matched}' wird gelöscht — das kann nicht rückgängig gemacht werden. Sicher?",
        f"Se eliminará '{matched}' — esto no se puede deshacer. ¿Seguro?",
        lang
    )


# ── Weekday dates — worked out in code ─────────────────────────────────────────────────
# The model gets weekday maths wrong ("Dinner on Friday", said on a Friday, became last
# Wednesday). The prompt gets a table of the coming days, and a single weekday in the message
# ("Friday", "Freitag", "viernes") is resolved here and overrides the model's date.

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3, "freitag": 4, "samstag": 5,
    "sonnabend": 5, "sonntag": 6,
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2, "jueves": 3, "viernes": 4,
    "sábado": 5, "sabado": 5, "domingo": 6,
}
_WEEKDAY_RE = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")s?\b", re.IGNORECASE)
_EXPLICIT_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}(st|nd|rd|th)\b",
                               re.IGNORECASE)


def _has_explicit_date(text: str) -> bool:
    """A numeric date ("02.11.", "3rd") or a day with a month name ("6 November", "May 3")."""
    if _EXPLICIT_DATE_RE.search(text):
        return True
    months = "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True))
    return bool(re.search(rf"\b\d{{1,2}}\.?\s+(of\s+|de\s+)?({months})\b|\b({months})\.?\s+\d{{1,2}}\b",
                          text, re.IGNORECASE))


def _upcoming_days_table(now: datetime, days: int = 14) -> str:
    rows = [f"- {(now + timedelta(days=i)).strftime('%A %Y-%m-%d')}"
            + (" (today)" if i == 0 else " (tomorrow)" if i == 1 else "") for i in range(days)]
    return "Upcoming days (use these dates — never calculate weekdays yourself):\n" + "\n".join(rows)


def _resolve_weekday(question: str, start_iso: str | None, now: datetime) -> str | None:
    """If the message names exactly one weekday and no explicit date, return `start_iso` moved
    to the next such weekday (today counts if its time hasn't passed yet). Else None."""
    if not start_iso or _has_explicit_date(question):
        return None
    days = {_WEEKDAYS[m.lower()] for m in _WEEKDAY_RE.findall(question)}
    if len(days) != 1:
        return None
    try:
        start = datetime.fromisoformat(start_iso)
    except ValueError:
        return None
    ahead = (days.pop() - now.weekday()) % 7
    target = now.date() + timedelta(days=ahead)
    if ahead == 0 and start.time() <= now.time():
        target += timedelta(days=7)
    return start.replace(year=target.year, month=target.month, day=target.day).isoformat()


def handle_calendar_event(question: str, extracted: dict, lang: str = "en") -> str:
    """Extract event details and confirm creation."""
    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    extraction_prompt = f"""Today is {current_date}.
{_upcoming_days_table(now)}
{previous_context}The user said: "{question}"
Hint from classifier — title: "{extracted.get('title', '')}", description: "{extracted.get('description', '')}"

Extract calendar event details. Return ONLY a JSON object:
- title: the event name in the user's own words and language — leave out the date, the time
  and the words that introduce them ("am", "um", "on", "at", "el", "a las")
- start: ISO 8601 datetime
- end: ISO 8601 datetime (null if not specified)
- description: extra details (empty if none)
- attendees: list of email addresses (empty list if none)

Return ONLY the JSON.
"""
    response = llm_invoke(extraction_prompt).strip()
    response = response[response.find("{"):response.rfind("}")+1]

    try:
        event_data = json.loads(response)
    except json.JSONDecodeError:
        return _t(
            "Couldn't parse the event details — try again?",
            "Die Termindetails konnten nicht verarbeitet werden — nochmal versuchen?",
            "No pude procesar los detalles del evento — ¿intentamos de nuevo?",
            lang
        )

    print(f"📅 Extracted event: {event_data}")

    resolved = _resolve_weekday(question, event_data.get("start"), now)
    if resolved and resolved != event_data.get("start"):
        print(f"📅 Weekday date fixed in code: {event_data['start']} → {resolved}")
        if event_data.get("end"):
            try:   # keep the duration
                shift = datetime.fromisoformat(resolved) - datetime.fromisoformat(event_data["start"])
                event_data["end"] = (datetime.fromisoformat(event_data["end"]) + shift).isoformat()
            except ValueError:
                event_data["end"] = None
        event_data["start"] = resolved

    try:
        validated = CalendarEventData(
            title=event_data.get("title", ""),
            start=event_data.get("start"),
            end=event_data.get("end"),
            description=event_data.get("description", ""),
            attendees=event_data.get("attendees", [])
        )
    except ValidationError as e:
        print(f"⚠️ Calendar event validation failed: {e}")
        return _t(
            "The event details didn't look right — could you rephrase with a clear date and time?",
            "Die Termindetails sahen nicht korrekt aus — kannst du es mit Datum und Uhrzeit nochmal versuchen?",
            "Los detalles del evento no eran correctos — ¿puedes intentarlo de nuevo con fecha y hora?",
            lang
        )

    event_dict = validated.model_dump()
    _pending().text = f"{validated.title} on {validated.start}"  # internal ISO kept
    _pending().action = "calendar"
    _pending().conflict = None
    _pending().line_to_delete = None
    _pending().event_data = event_dict
    _pending().lang = lang

    confirmation = _t(
        f"Lock in '{validated.title}' on {_fmt_event_dt(validated.start, 'en')}?",
        f"'{validated.title}' am {_fmt_event_dt(validated.start, 'de')} in den Kalender eintragen?",
        f"¿Agendar '{validated.title}' el {_fmt_event_dt(validated.start, 'es')}?",
        lang
    )
    if validated.attendees:
        confirmation += " " + _t(
            f"I'll invite {', '.join(validated.attendees)} too.",
            f"Ich lade auch {', '.join(validated.attendees)} ein.",
            f"También invitaré a {', '.join(validated.attendees)}.",
            lang
        )
    return confirmation


def handle_view_reminders(question: str, lang: str = "en") -> str:
    """Return all current reminders."""
    reminders = get_all_reminders()
    if reminders == "No reminders found.":
        return _t(
            "All clear — nothing on your reminder list.",
            "Alles klar — keine Erinnerungen vorhanden.",
            "Todo despejado — no tienes recordatorios.",
            lang
        )

    format_prompt = f"""{_persona_prompt(lang)}
Today is: {datetime.now().strftime("%A, %B %d, %Y")}
The user asked: "{question}"

Their reminders:
{reminders}

Present these in a friendly, natural way — warm but concise.
Respond in {_LANG_NAMES.get(lang, 'English')}."""

    return llm_invoke(format_prompt).strip()


# ===================================
# DATE FACTS — date maths done in code, handed to the LLM
# ===================================
# Small models guess badly ("your birthday on March 3rd is in 8 months" — it was 10).
# _date_facts() finds dates in profile/chunk text and states the relative offsets.

_MONTH_NUMBERS = {
    "january": 1, "jan": 1, "januar": 1, "enero": 1, "february": 2, "feb": 2, "februar": 2, "febrero": 2,
    "march": 3, "mar": 3, "märz": 3, "marzo": 3, "april": 4, "apr": 4, "abril": 4,
    "may": 5, "mai": 5, "mayo": 5, "june": 6, "jun": 6, "juni": 6, "junio": 6,
    "july": 7, "jul": 7, "juli": 7, "julio": 7, "august": 8, "aug": 8, "agosto": 8,
    "september": 9, "sep": 9, "sept": 9, "septiembre": 9, "october": 10, "oct": 10, "oktober": 10,
    "okt": 10, "octubre": 10, "november": 11, "nov": 11, "noviembre": 11,
    "december": 12, "dec": 12, "dezember": 12, "dez": 12, "diciembre": 12,
}
_M = "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True))
_DATE_PATTERNS = [
    re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\.?\s+(?:of\s+|de\s+)?(?P<m>{_M})\b\.?,?\s*(?:de\s+)?(?P<y>\d{{4}})?", re.I),
    re.compile(rf"\b(?P<m>{_M})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(?P<y>\d{{4}})?", re.I),
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    re.compile(r"\b(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4})\b"),
]
_BIRTH_WORDS = re.compile(r"born|birthday|geburtstag|geboren|cumpleaños|nací|naci", re.I)


def _rel_days(delta: int) -> str:
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    return f"in {delta} days" if delta > 0 else f"{-delta} days ago"


def _date_facts(*texts: str, max_facts: int = 12) -> str:
    """Return a prompt block of precomputed date facts for dates found in `texts`."""
    from datetime import date as _date
    today = _date.today()
    facts: dict[tuple, tuple[int, str]] = {}   # (month, day, year) -> (priority, line)
    for text in texts:
        if not text:
            continue
        for pattern in _DATE_PATTERNS:
            for m in pattern.finditer(text):
                month = m.group("m")
                month = int(month) if month.isdigit() else _MONTH_NUMBERS.get(month.lower())
                try:
                    day, year = int(m.group("d")), int(m.group("y")) if m.group("y") else None
                    _date(year or 2000, month, day)       # validate (2000 is a leap year)
                except (TypeError, ValueError):
                    continue
                key = (month, day, year)
                if key in facts:
                    continue
                label = " ".join(m.group(0).split())
                is_birth = bool(_BIRTH_WORDS.search(text[max(0, m.start() - 40):m.start()]))
                if year is None and not is_birth:
                    # No year given: a recent date most likely meant this year
                    # ("the concert is on the 4th of October", written in September).
                    try:
                        this_year = _date(today.year, month, day)
                    except ValueError:
                        this_year = None
                    if this_year and -60 <= (this_year - today).days < 0:
                        facts[key] = (3, f"- \"{label}\" = {this_year:%A, %B %d, %Y}, "
                                         f"{_rel_days((this_year - today).days)} (already in the past).")
                        continue
                if is_birth or year is None:
                    # Recurring date (birthday / date without a year): next occurrence
                    nxt = None
                    for y in (today.year, today.year + 1, today.year + 2):
                        try:
                            cand = _date(y, month, day)
                        except ValueError:            # 29 February in a non-leap year
                            continue
                        if cand >= today:
                            nxt = cand
                            break
                    if not nxt:
                        continue
                    days = _rel_days((nxt - today).days)
                    if is_birth:
                        line = (f"- Birthday: {nxt:%B} {day}" + (f" (born {year})" if year else "") +
                                f". Next birthday: {nxt:%A, %B %d, %Y}, {days}" +
                                (f"; {USER_NAME or 'the user'} turns {nxt.year - year}." if year else "."))
                    else:
                        line = f"- \"{label}\" = {nxt:%A, %B %d, %Y}, {days}."
                    facts[key] = (0 if is_birth else 2, line)
                else:
                    when = _date(year, month, day)
                    delta = (when - today).days
                    if not -60 <= delta <= 400:       # far past/future dates are just noise
                        continue
                    past = " (already in the past)" if delta < 0 else ""
                    facts[key] = (1 if delta >= 0 else 3,
                                  f"- \"{label}\" = {when:%A, %B %d, %Y}, {_rel_days(delta)}{past}.")
    if not facts:
        return ""
    lines = [line for _, line in sorted(facts.values(), key=lambda f: f[0])][:max_facts]
    return (f"\nDate facts (computed for you — today is {today:%A, %B %d, %Y}). Use these numbers "
            f"exactly and never calculate dates yourself. If a note says something \"will\" happen "
            f"but its date is in the past, it already happened — use the past tense:\n"
            + "\n".join(lines) + "\n")


def _relative_event_label(start_iso: str) -> str:
    """Convert an ISO start string to a human-readable label with a relative-date hint.
    Example: 'Wednesday, June 4 at 8:00 PM (tomorrow)'
    Python computes the relative offset so the LLM never has to do date arithmetic.
    """
    try:
        from datetime import date as _date
        dt = datetime.fromisoformat(start_iso)
        today = _date.today()
        delta = (dt.date() - today).days
        if delta == 0:
            rel = "today"
        elif delta == 1:
            rel = "tomorrow"
        elif delta == -1:
            rel = "yesterday"
        elif delta > 1:
            rel = f"in {delta} days"
        else:
            rel = f"{abs(delta)} days ago"
        has_time = dt.hour != 0 or dt.minute != 0
        date_part = dt.strftime("%A, %B %d")
        if has_time:
            h = dt.hour % 12 or 12
            ampm = "AM" if dt.hour < 12 else "PM"
            time_part = f" at {h}:{dt.minute:02d} {ampm}"
        else:
            time_part = ""
        return f"{date_part}{time_part} ({rel})"
    except Exception:
        return start_iso


def handle_view_events(question: str, lang: str = "en") -> str:
    """Return upcoming calendar events — live Google Calendar first, ChromaDB fallback."""
    current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    events_context = None
    live_failed = False

    # Always try live Google Calendar first
    try:
        events = get_upcoming_events(
            days_ahead=CALENDAR_DAYS_AHEAD,
            days_behind=CALENDAR_DAYS_BEHIND
        )
        if not events:
            return _t(
                "Nothing on your calendar for the next two weeks.",
                "Keine Termine in den nächsten zwei Wochen.",
                "Nada en tu calendario para las próximas dos semanas.",
                lang
            )
        lines = [f"- {e.title}: {_relative_event_label(e.start)}" for e in events]
        events_context = f"Today is: {current_date}\n\nUpcoming events:\n" + "\n".join(lines)
    except Exception as e:
        print(f"Live calendar fetch failed: {e} — falling back to ChromaDB")
        live_failed = True

    if live_failed:
        try:
            results = vectorstore.similarity_search(
                question, k=5,
                filter={"collection": "calendar"}
            )
            if results:
                context = "\n\n".join([doc.page_content for doc in results])
                events_context = f"Today is: {current_date}\n\n[Cached calendar data]\n\n{context}"
            else:
                return _t(
                    "I couldn't reach your Google Calendar right now — is the server online?",
                    "Ich konnte deinen Google Kalender nicht erreichen — ist der Server online?",
                    "No pude acceder a tu Google Calendar — ¿está el servidor activo?",
                    lang
                )
        except Exception as e2:
            print(f"ChromaDB calendar fallback also failed: {e2}")
            return _t(
                "I couldn't reach your Google Calendar right now — is the server online?",
                "Ich konnte deinen Google Kalender nicht erreichen — ist der Server online?",
                "No pude acceder a tu Google Calendar — ¿está el servidor activo?",
                lang
            )

    format_prompt = f"""{_persona_prompt(lang)}

The user asked: "{question}"

{events_context}

Answer the question asked; only list events that are relevant to it.
If the question is not about these events at all, answer it without mentioning them.
Be friendly and natural. Respond in {_LANG_NAMES.get(lang, 'English')}.
Keep it concise — 1 to 4 events max, summarize the rest if there are more."""

    return llm_invoke(format_prompt).strip()


def handle_view_agenda(question: str, lang: str = "en") -> str:
    """Combined agenda: reminders + upcoming events in one LLM call."""
    current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # ── Reminders ──
    reminders = get_all_reminders()
    reminders_section = reminders if reminders != "No reminders found." else None

    # ── Events: live Google Calendar first, ChromaDB fallback ──
    events_section = None
    try:
        events = get_upcoming_events(
            days_ahead=CALENDAR_DAYS_AHEAD,
            days_behind=CALENDAR_DAYS_BEHIND
        )
        if events:
            events_section = "\n".join(
                f"- {e.title}: {_relative_event_label(e.start)}" for e in events
            )
    except Exception as e:
        print(f"Agenda live calendar fetch failed: {e} — falling back to ChromaDB")
        try:
            results = vectorstore.similarity_search(
                question, k=5,
                filter={"collection": "calendar"}
            )
            if results:
                events_section = "[Cached] " + "\n\n".join(
                    [doc.page_content for doc in results]
                )
        except Exception as e2:
            print(f"Agenda ChromaDB fallback also failed: {e2}")

    # ── Build combined context ──
    sections = []
    if reminders_section:
        sections.append(f"Reminders:\n{reminders_section}")
    if events_section:
        sections.append(f"Upcoming events:\n{events_section}")

    if not sections:
        return _t(
            "All clear — nothing on your reminders or calendar.",
            "Alles frei — keine Erinnerungen und keine Termine.",
            "Todo despejado — sin recordatorios ni eventos.",
            lang
        )

    combined = "\n\n".join(sections)

    prompt = f"""{_persona_prompt(lang)}
Today is {current_date}.

The user asked: "{question}"

Here is their current agenda:
{combined}

Give a brief, natural overview — what's coming up, what to keep in mind.
Highlight anything time-sensitive or important.
Respond in {_LANG_NAMES.get(lang, 'English')}.
Keep it concise — a few sentences at most."""

    return llm_invoke(prompt).strip()


# ── Chatette TV handlers ──────────────────────────────────────────────────────

_CHANNEL_DISPLAY = {
    "ard": "ARD", "zdf": "ZDF", "euronews": "Euronews",
    "arte_fr": "Arte (FR)", "arte_de": "Arte (DE)",
    "milenio": "Milenio", "tv5monde": "TV5 Monde", "nhk": "NHK World",
}


def handle_cast_tv_power(_question: str, extracted: dict, lang: str = "en") -> str:
    action = extracted.get("action", "on")
    ok = _cast_manager.power_on() if action == "on" else _cast_manager.power_off()
    if not ok:
        return _t("I couldn't reach the TV.", "Ich konnte den Fernseher nicht erreichen.", "No pude conectar con el televisor.", lang)
    if action == "on":
        return _t("TV is on.", "Fernseher eingeschaltet.", "Televisor encendido.", lang)
    return _t("TV is off.", "Fernseher ausgeschaltet.", "Televisor apagado.", lang)


def handle_cast_volume(_question: str, extracted: dict, lang: str = "en") -> str:
    level = extracted.get("level")
    delta = extracted.get("delta")
    if level is not None:
        new = _cast_manager.set_volume(int(level))
    elif delta is not None:
        new = _cast_manager.volume_delta(int(delta))
    else:
        # Infer from question keywords
        q = _question.lower()
        if any(w in q for w in ("louder", "up", "lauter", "hoch", "más alto", "subir")):
            new = _cast_manager.volume_delta(10)
        elif any(w in q for w in ("quieter", "down", "leiser", "runter", "bajar", "menos")):
            new = _cast_manager.volume_delta(-10)
        else:
            new = _cast_manager.volume_delta(10)
    if new < 0:
        return _t("I couldn't reach the TV.", "Ich konnte den Fernseher nicht erreichen.", "No pude conectar con el televisor.", lang)
    return _t(f"Volume set to {new}%.", f"Lautstärke auf {new}% gesetzt.", f"Volumen al {new}%.", lang)


def handle_cast_youtube(_question: str, extracted: dict, lang: str = "en") -> str:
    if not _yt_search.is_available():
        if not _yt_search.YOUTUBE_API_KEY:
            return _t("YouTube search is not configured.", "YouTube-Suche ist nicht konfiguriert.", "La búsqueda de YouTube no está configurada.", lang)
        return _t("YouTube search is unavailable for today — the daily quota has been reached.", "Die YouTube-Suche ist heute nicht verfügbar — das Tageslimit wurde erreicht.", "La búsqueda de YouTube no está disponible hoy — se alcanzó el límite diario.", lang)
    query = extracted.get("query", _question)
    result = _yt_search.search_videos(query, max_results=1)
    if result is None or not result["results"]:
        if _yt_search._quota_exceeded:
            return _t("YouTube search is unavailable for today — the daily quota has been reached.", "Die YouTube-Suche ist heute nicht verfügbar — das Tageslimit wurde erreicht.", "La búsqueda de YouTube no está disponible hoy — se alcanzó el límite diario.", lang)
        return _t("I couldn't find that on YouTube.", "Ich konnte das auf YouTube nicht finden.", "No encontré eso en YouTube.", lang)
    video_id = result["results"][0]["video_id"]
    title    = result["results"][0]["title"]
    ok = _cast_manager.play_youtube(video_id)
    if not ok:
        return _t("I couldn't reach the TV.", "Ich konnte den Fernseher nicht erreichen.", "No pude conectar con el televisor.", lang)
    return _t(f"Casting '{title}'.", f"Ich spiele '{title}'.", f"Reproduciendo '{title}'.", lang)


def handle_cast_channel(_question: str, extracted: dict, lang: str = "en") -> str:
    channel = extracted.get("channel", "")
    url = _channel_registry.get_url(channel)
    if not url:
        return _t(f"I don't have a stream for {channel}.", f"Ich habe keinen Stream für {channel}.", f"No tengo stream para {channel}.", lang)
    name = _CHANNEL_DISPLAY.get(channel, channel)
    ok = _cast_manager.play_hls(url, title=name)
    if not ok:
        return _t("I couldn't reach the TV.", "Ich konnte den Fernseher nicht erreichen.", "No pude conectar con el televisor.", lang)
    return _t(f"Opening {name}.", f"Ich öffne {name}.", f"Abriendo {name}.", lang)


def handle_cast_stop(_question: str, lang: str = "en") -> str:
    ok = _cast_manager.stop()
    if not ok:
        return _t("I couldn't reach the TV.", "Ich konnte den Fernseher nicht erreichen.", "No pude conectar con el televisor.", lang)
    return _t("Playback stopped.", "Wiedergabe gestoppt.", "Reproducción detenida.", lang)


# ── Timer / alarm thread-local (read by api.py after ask()) ──────────────────
_tl = threading.local()

def _clear_alarm_state():
    _tl.timer_seconds = None
    _tl.alarm_time    = None

def get_last_alarm():
    return {
        'seconds':    getattr(_tl, 'timer_seconds', None),
        'alarm_time': getattr(_tl, 'alarm_time',    None),
    }


def handle_set_alarm(_question: str, extracted: dict, lang: str = "en") -> str:
    """Handle set_alarm intent — covers both countdowns and wall-clock alarms."""
    _clear_alarm_state()

    seconds    = extracted.get('seconds')
    hour       = extracted.get('hour')
    minute     = extracted.get('minute', 0)

    if seconds is not None:
        # Countdown timer
        secs = int(seconds)
        if secs <= 0:
            return _t("I didn't catch the duration — how long should the timer run?",
                      "Ich habe die Dauer nicht verstanden — wie lange soll der Timer laufen?",
                      "No escuché la duración — ¿cuánto tiempo debe durar el temporizador?",
                      lang)
        _tl.timer_seconds = secs
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        parts = []
        if hh: parts.append({"en": f"{hh}h", "de": f"{hh} Std.", "es": f"{hh}h"}[lang])
        if mm: parts.append({"en": f"{mm} minutes", "de": f"{mm} Minuten", "es": f"{mm} minutos"}[lang])
        if ss: parts.append({"en": f"{ss} seconds", "de": f"{ss} Sekunden", "es": f"{ss} segundos"}[lang])
        time_str = " ".join(parts) or "0s"
        return _t(f"Timer set for {time_str} from now.",
                  f"Timer auf {time_str} ab jetzt gestellt.",
                  f"Temporizador de {time_str} a partir de ahora.",
                  lang)

    elif hour is not None:
        # Wall-clock alarm
        hh, mm = int(hour), int(minute)
        if hh > 23 or mm > 59:
            return _t("That doesn't look like a valid time.",
                      "Das scheint keine gültige Uhrzeit zu sein.",
                      "Eso no parece una hora válida.",
                      lang)
        _tl.alarm_time = (hh, mm)
        time_str = f"{hh:02d}:{mm:02d}"
        return _t(f"Alarm set at {time_str}.",
                  f"Wecker gestellt auf {time_str} Uhr.",
                  f"Alarma programada a las {time_str}.",
                  lang)

    else:
        return _t("I didn't catch the time — please try again.",
                  "Ich habe die Zeit nicht verstanden — bitte versuche es erneut.",
                  "No escuché la hora — por favor inténtalo de nuevo.",
                  lang)


def handle_view_emails(question: str, extracted: dict, lang: str = "en") -> str:
    """Return recent emails — RAG first, live Gmail fallback if empty."""
    current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    max_results = int(extracted.get("max_results", 5))

    # Try RAG first (fast, no API call)
    rag_docs = []
    try:
        results = vectorstore.similarity_search(
            question, k=max_results,
            filter={"collection": "emails"}
        )
        rag_docs = results
    except Exception as e:
        print(f"Email RAG retrieval failed: {e}")

    if rag_docs:
        context = "\n\n".join([doc.page_content for doc in rag_docs])
        emails_context = f"Today is: {current_date}\n\n{context}"
    else:
        print("Email RAG empty — fetching live emails")
        try:
            from google_integration import get_recent_emails
            emails = get_recent_emails(max_results=max_results)
        except Exception as e:
            print(f"Live email fetch failed: {e}")
            return _t(
                "I couldn't reach your Gmail right now — is the server online?",
                "Ich konnte dein Gmail gerade nicht erreichen — ist der Server online?",
                "No pude acceder a tu Gmail ahora — ¿está el servidor en línea?",
                lang
            )

        if not emails:
            return _t(
                "No recent emails found.",
                "Keine neuen E-Mails gefunden.",
                "No se encontraron correos recientes.",
                lang
            )

        lines = [
            f"- From: {e.from_} | Subject: {e.subject} | {e.date}"
            for e in emails
        ]
        emails_context = f"Today is: {current_date}\n\nRecent emails:\n" + "\n".join(lines)

    is_voice = _current_voice.get()

    if is_voice:
        format_instructions = f"""Summarize the emails out loud in Chatette's warm, conversational voice — as if speaking to someone, not writing a list.
Always mention emails about: purchases or orders, appointments or reservations, personal messages from real people, contracts or legal notices, job interviews or work offers, travel (flights, hotels, bookings). Skip newsletters, promotions, automated notifications, and social media alerts unless nothing else is present.
Keep it to 2-4 natural sentences. No bullet points or dashes.
If no important emails were found, say so in one friendly sentence.
Respond in {_LANG_NAMES.get(lang, 'English')}.

Example good response: "You got a couple of things worth noting — Anna sent you a message about weekend plans, and Dr. Müller's office confirmed your dental appointment for Thursday at 10."
Example bad response (do NOT do this): "- From: LinkedIn | 5 new job suggestions. - From: Amazon | Today's deals." """
    else:
        format_instructions = f"""Summarize the emails below. Always mention emails about: purchases or orders, appointments or reservations, personal messages from real people, contracts or legal notices, job interviews or work offers, travel (flights, hotels, bookings). Skip newsletters, promotions, automated notifications, and social media alerts unless nothing else is present.

Format: one bullet per email worth mentioning — sender, subject, and one sentence about the content if relevant.
If no important emails were found, say so briefly.
Respond in {_LANG_NAMES.get(lang, 'English')}.

Examples:
- From: Amazon | Order #123 shipped — your headphones are on the way, arriving Friday.
- From: Dr. Müller's office | Appointment confirmation — dental check-up Thursday at 10am.
- From: Anna | Weekend plans — she's asking if you're free Saturday.
- From: Lufthansa | Your flight to Lisbon — booking confirmed, departs Oct 12 at 7:15am."""

    format_prompt = f"""{_persona_prompt(lang)}

The user asked: "{question}"

{emails_context}

{format_instructions}"""

    return llm_invoke(format_prompt).strip()


def handle_places(question: str, extracted: dict, lang: str = "en") -> str:
    """Search OSM for nearby places and return a natural recommendation."""
    place_type = extracted.get("place_type", "restaurant").strip().lower()
    cuisine = extracted.get("cuisine", "").strip().lower() or None

    try:
        places = search_places(place_type, cuisine=cuisine)
    except Exception as e:
        print(f"⚠️ Places search error: {e}")
        return _t(
            "I couldn't search for places right now — try again in a moment.",
            "Ich konnte gerade keine Orte suchen — versuch es gleich nochmal.",
            "No pude buscar lugares en este momento — inténtalo de nuevo.",
            lang
        )

    if not places:
        query_desc = f"{cuisine} {place_type}" if cuisine else place_type
        return _t(
            f"I couldn't find any {query_desc} nearby.",
            f"Ich konnte keine {query_desc} in der Nähe finden.",
            f"No encontré ningún {query_desc} cerca.",
            lang
        )

    # Format places for the LLM
    places_text = ""
    for i, p in enumerate(places[:3], 1):
        dist = p["distance_m"]
        dist_str = f"{dist}m" if dist < 1000 else f"{dist/1000:.1f}km"
        line = f"{i}. {p['name']}"
        if p["address"]:
            line += f" — {p['address']}"
        line += f" ({dist_str} away)"
        if p["cuisine"]:
            line += f" | cuisine: {p['cuisine']}"
        if p["opening_hours"]:
            line += f" | hours: {p['opening_hours']}"
        places_text += line + "\n"

    places_prompt = f"""{_persona_prompt(lang)}

The user asked: "{question}"

Here are the nearby places I found via OpenStreetMap:
{places_text.strip()}

Give 2-3 brief recommendations in Chatette's warm, conversational voice.
For each place: mention the name, why it's a good match, and where it is.
Keep the total response to 3-5 sentences. Do not mention OpenStreetMap.
Respond in {_LANG_NAMES.get(lang, 'English')}."""

    return llm_invoke(places_prompt).strip()


def handle_weather(question: str, extracted: dict, lang: str = "en") -> str:
    """Fetch weather and return a natural Chatette-style response with time-of-day temperatures and qualitative wind descriptions."""
    city = extracted.get("city", "").strip()
    timeframe = str(extracted.get("timeframe", "today")).lower()

    # "3 days", "next 3 days", "3 Tage", "3 días" → first N days of the weekly forecast
    days_match = re.search(r"(\d+)\s*(?:days?|tagen?|d[ií]as)", timeframe)
    num_days = min(max(int(days_match.group(1)), 1), 7) if days_match else 0

    if num_days > 1:
        timeframe = "days"
    elif timeframe in ("now", "current", "currently", "jetzt", "aktuell", "ahora"):
        timeframe = "now"
    elif timeframe in ("week", "weekly", "7 days", "woche", "semana", "this week"):
        timeframe = "week"
    elif timeframe in ("tomorrow", "morgen", "mañana"):
        timeframe = "tomorrow"
    else:
        timeframe = "today"

    try:
        if timeframe == "now":
            data = get_current_weather(city)
            context = format_weather_context(data, "now")
        elif timeframe == "week":
            data = get_weekly_forecast(city)
            context = format_weather_context(data, "week")
        elif timeframe == "days":
            data = get_weekly_forecast(city)[:num_days]
            context = format_weather_context(data, "week")
        elif timeframe == "tomorrow":
            data = get_weekly_forecast(city)
            tomorrow_data = [data[1]] if len(data) > 1 else data
            context = format_weather_context(tomorrow_data, "week")
        else:
            data = get_today_forecast(city)
            context = format_weather_context(data, "today")

    except ValueError as e:
        return _t(
            f"I couldn't get the weather — {e}",
            f"Ich konnte das Wetter nicht abrufen — {e}",
            f"No pude obtener el clima — {e}",
            lang
        )
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return _t(
            "Couldn't reach the weather service right now — try again in a moment?",
            "Der Wetterdienst ist gerade nicht erreichbar — kurz warten und nochmal versuchen?",
            "No pude contactar el servicio del clima — ¿intentamos de nuevo en un momento?",
            lang
        )

    weather_prompt = f"""{_persona_prompt(lang)}

The user asked: "{question}"

Here is the weather data (this is the ONLY weather information you have):
{context}

Respond naturally in Chatette's voice — warm, concise, useful.
Rules for your response:
- Use only the data above: name the location it is for, and never make up or estimate temperatures, conditions or places that are not in it.
- If the data covers several days, say a few words about each day (one short sentence per day).
- Give temperatures with time references: say "X degrees in the morning / during the day / in the evening / at night", not "between X and Y degrees".
- Describe wind qualitatively, not as a speed: use words like "calm", "light breeze", "moderate wind", "strong wind", "very windy". Never say km/h.
- Mention rain only if it is likely (≥40% probability).
- Otherwise keep it to 2-3 sentences maximum.
Respond in {_LANG_NAMES.get(lang, 'English')}."""

    return llm_invoke(weather_prompt).strip()


def handle_about_chatette(question: str, lang: str = "en") -> str:
    """Answer questions about Chatette."""
    about_prompt = (
        f"{_persona_prompt(lang)}\n\n"
        f"{_build_conversation_context()}"
        f"Answer naturally in 1-2 sentences.\n"
        f"Always respond in {_LANG_NAMES.get(lang, 'English')}.\n\n"
        f"Question: {question}\n"
        f"Chatette:"
    )
    return llm_invoke(about_prompt).strip()


def handle_confirmation(question: str, lang: str = "en") -> str:
    """Handle yes/no confirmation for all pending actions."""
    action = _pending().action
    reminder_text = _pending().text
    line_to_delete = _pending().line_to_delete
    conflict = _pending().conflict
    event_data = _pending().event_data
    lang = _pending().lang

    is_yes = any(word in question.lower() for word in
                 ["yes", "correct", "right", "yep", "yeah", "sure", "ok", "okay",
                  "ja", "sí", "si", "claro", "correcto", "genau", "klar", "gut"])
    is_no = any(word in question.lower() for word in
                ["no", "wrong", "incorrect", "nope", "cancel",
                 "nein", "falsch", "abbrechen"])

    if is_yes:
        if action == "save":
            if line_to_delete and conflict in ["duplicate", "conflict"]:
                delete_reminder_by_line(line_to_delete)
            create_reminder(reminder_text, due=_pending().due)
            ingest_notes()
            _pending().clear()
            return _t(
                f"On it — '{reminder_text}' is on your list.",
                f"Erledigt — '{reminder_text}' ist auf deiner Liste.",
                f"Listo — '{reminder_text}' está en tu lista.",
                lang
            )

        elif action == "save_batch":
            items = _pending().items or []
            for item in items:
                if isinstance(item, dict):
                    create_reminder(item["text"], due=item.get("due"))
                else:
                    create_reminder(item)
            ingest_notes()
            _pending().clear()
            count = len(items)
            return _t(
                f"Done — {count} reminder{'s' if count != 1 else ''} saved.",
                f"Erledigt — {count} Erinnerung{'en' if count != 1 else ''} gespeichert.",
                f"Listo — {count} recordatorio{'s' if count != 1 else ''} guardado{'s' if count != 1 else ''}.",
                lang
            )

        elif action == "delete":
            result = delete_reminder_by_line(line_to_delete)
            ingest_notes()
            _pending().clear()
            return result

        elif action == "delete_all":
            count = delete_all_reminders()
            ingest_notes()
            _pending().clear()
            return _t(
                f"Done — cleared {count} reminder{'s' if count != 1 else ''}.",
                f"Erledigt — {count} Erinnerungen gelöscht.",
                f"Listo — {count} recordatorios eliminados.",
                lang
            )

        elif action == "personal_note":
            save_personal_note(reminder_text)
            try:
                ingest_notes()
            except Exception as e:
                print(f"⚠️ RAG ingestion failed after saving note: {e}")
            _pending().clear()
            return _t(
                "Done — added to your personal notes.",
                "Erledigt — zu deinen persönlichen Notizen hinzugefügt.",
                "Listo — añadido a tus notas personales.",
                lang
            )

        elif action == "personal_note_batch":
            items = _pending().items or []
            for item in items:
                save_personal_note(item)
            try:
                ingest_notes()
            except Exception as e:
                print(f"⚠️ RAG ingestion failed after saving notes: {e}")
            _pending().clear()
            count = len(items)
            return _t(
                f"Done — {count} note{'s' if count != 1 else ''} added to your personal notes.",
                f"Erledigt — {count} Eintrag/Einträge zu deinen persönlichen Notizen hinzugefügt.",
                f"Listo — {count} nota{'s' if count != 1 else ''} añadida{'s' if count != 1 else ''} a tus notas personales.",
                lang
            )

        elif action == "create_list":
            title = event_data["title"]
            items = event_data["items"]
            filename = create_list(title, items)
            ingest_lists()
            _pending().clear()
            return _t(
                f"List '{title}' is ready — find it in your documents.",
                f"Liste '{title}' ist bereit — du findest sie in deinen Dokumenten.",
                f"Lista '{title}' lista — encuéntrala en tus documentos.",
                lang
            )

        elif action == "add_to_list":
            item = event_data["item"]
            filename = event_data["filename"]
            add_item_to_list(filename, item)
            ingest_lists()
            _pending().clear()
            return _t(
                f"'{item}' is on the list.",
                f"'{item}' steht auf der Liste.",
                f"'{item}' está en la lista.",
                lang
            )

        elif action == "add_to_list_batch":
            items = event_data["items"]
            filename = event_data["filename"]
            for item in items:
                add_item_to_list(filename, item)
            ingest_lists()
            _pending().clear()
            count = len(items)
            return _t(
                f"Done — {count} item{'s' if count != 1 else ''} added to '{filename}'.",
                f"Erledigt — {count} Einträge zu '{filename}' hinzugefügt.",
                f"Listo — {count} elemento{'s' if count != 1 else ''} añadido{'s' if count != 1 else ''} a '{filename}'.",
                lang
            )

        elif action == "remove_from_list":
            filename = event_data["filename"]
            line_index = event_data["line_index"]
            delete_list_item(filename, line_index)
            ingest_lists()
            _pending().clear()
            return _t(
                f"'{reminder_text}' is off the list.",
                f"'{reminder_text}' wurde von der Liste entfernt.",
                f"'{reminder_text}' eliminado de la lista.",
                lang
            )

        elif action == "delete_list":
            success = delete_list(line_to_delete)
            ingest_lists()
            _pending().clear()
            if success:
                return _t(
                    f"List '{line_to_delete}' deleted.",
                    f"Liste '{line_to_delete}' gelöscht.",
                    f"Lista '{line_to_delete}' eliminada.",
                    lang
                )
            return _t(
                "Something went wrong — try again?",
                "Etwas ist schiefgelaufen — nochmal versuchen?",
                "Algo salió mal — ¿intentamos de nuevo?",
                lang
            )

        elif action == "calendar":
            if event_data is None:
                return _t(
                    "Lost the event details somewhere — try again?",
                    "Die Termindetails sind verloren gegangen — nochmal versuchen?",
                    "Perdí los detalles del evento — ¿intentamos de nuevo?",
                    lang
                )
            create_calendar_event(
                title=event_data["title"],
                start_datetime=event_data["start"],
                end_datetime=event_data.get("end"),
                description=event_data.get("description", ""),
                attendees=event_data.get("attendees", [])
            )
            ingest_calendar_events()
            _pending().clear()
            result = _t(
                f"Locked in — '{event_data['title']}' is on your calendar.",
                f"Eingetragen — '{event_data['title']}' ist in deinem Kalender.",
                f"Agendado — '{event_data['title']}' está en tu calendario.",
                lang
            )
            if event_data.get("attendees"):
                result += " " + _t(
                    f"Invitations sent to {', '.join(event_data['attendees'])}.",
                    f"Einladungen an {', '.join(event_data['attendees'])} gesendet.",
                    f"Invitaciones enviadas a {', '.join(event_data['attendees'])}.",
                    lang
                )
            return result

        elif action == "delete_event":
            if event_data is None:
                return _t(
                    "Lost the event details — try again?",
                    "Die Termindetails sind verloren gegangen — nochmal versuchen?",
                    "Perdí los detalles del evento — ¿intentamos de nuevo?",
                    lang
                )
            try:
                delete_calendar_event(event_data["id"])
                ingest_calendar_events()
                _pending().clear()
                return _t(
                    f"Done — '{event_data['title']}' has been removed from your calendar.",
                    f"Erledigt — '{event_data['title']}' wurde aus deinem Kalender entfernt.",
                    f"Listo — '{event_data['title']}' fue eliminado de tu calendario.",
                    lang
                )
            except Exception as e:
                _pending().clear()
                print(f"Failed to delete event: {e}")
                return _t(
                    "Something went wrong deleting the event — try again?",
                    "Beim Löschen des Termins ist etwas schiefgelaufen — nochmal versuchen?",
                    "Algo salió mal al eliminar el evento — ¿intentamos de nuevo?",
                    lang
                )

    elif is_no:
        _pending().clear()
        return _t(
            "No problem — consider it dropped.",
            "Kein Problem — betrachte es als erledigt.",
            "Sin problema — lo descarto.",
            lang
        )

    else:
        if action == "save":
            return _t(
                f"Save '{reminder_text}' to reminders? Yes or no.",
                f"'{reminder_text}' in die Erinnerungen speichern? Ja oder Nein.",
                f"¿Guardar '{reminder_text}' en recordatorios? Sí o no.",
                lang
            )
        elif action == "delete":
            return _t(
                f"Delete '{line_to_delete}'? Yes or no.",
                f"'{line_to_delete}' löschen? Ja oder Nein.",
                f"¿Eliminar '{line_to_delete}'? Sí o no.",
                lang
            )
        elif action == "delete_all":
            return _t(
                "Delete all reminders? Yes or no.",
                "Alle Erinnerungen löschen? Ja oder Nein.",
                "¿Eliminar todos los recordatorios? Sí o no.",
                lang
            )
        elif action == "personal_note":
            return _t(
                "Add this to your personal notes? Yes or no.",
                "Das zu deinen persönlichen Notizen hinzufügen? Ja oder Nein.",
                "¿Añadir esto a tus notas personales? Sí o no.",
                lang
            )
        elif action == "create_list":
            return _t(
                f"Create list '{event_data['title']}'? Yes or no.",
                f"Liste '{event_data['title']}' erstellen? Ja oder Nein.",
                f"¿Crear lista '{event_data['title']}'? Sí o no.",
                lang
            )
        elif action == "add_to_list":
            return _t(
                f"Add '{event_data['item']}' to the list? Yes or no.",
                f"'{event_data['item']}' zur Liste hinzufügen? Ja oder Nein.",
                f"¿Añadir '{event_data['item']}' a la lista? Sí o no.",
                lang
            )
        elif action == "save_batch":
            items = _pending().items or []
            preview = ", ".join(items)
            return _t(
                f"Save these reminders: {preview}? Yes or no.",
                f"Diese Erinnerungen speichern: {preview}? Ja oder Nein.",
                f"¿Guardar estos recordatorios: {preview}? Sí o no.",
                lang
            )
        elif action == "personal_note_batch":
            items = _pending().items or []
            preview = ", ".join(items)
            return _t(
                f"Add these to your personal notes: {preview}? Yes or no.",
                f"Diese Einträge zu den persönlichen Notizen hinzufügen: {preview}? Ja oder Nein.",
                f"¿Añadir estos elementos a las notas personales: {preview}? Sí o no.",
                lang
            )
        elif action == "add_to_list_batch":
            items = event_data["items"]
            filename = event_data["filename"]
            preview = ", ".join(items)
            return _t(
                f"Add {preview} to '{filename}'? Yes or no.",
                f"{preview} zu '{filename}' hinzufügen? Ja oder Nein.",
                f"¿Añadir {preview} a '{filename}'? Sí o no.",
                lang
            )
        elif action == "remove_from_list":
            return _t(
                f"Remove '{reminder_text}' from the list? Yes or no.",
                f"'{reminder_text}' von der Liste entfernen? Ja oder Nein.",
                f"¿Eliminar '{reminder_text}' de la lista? Sí o no.",
                lang
            )
        elif action == "delete_list":
            return _t(
                f"Delete '{line_to_delete}'? Yes or no.",
                f"'{line_to_delete}' löschen? Ja oder Nein.",
                f"¿Eliminar '{line_to_delete}'? Sí o no.",
                lang
            )
        elif action == "calendar":
            return _t(
                f"Lock in '{event_data['title']}' on your calendar? Yes or no.",
                f"'{event_data['title']}' in den Kalender eintragen? Ja oder Nein.",
                f"¿Agendar '{event_data['title']}'? Sí o no.",
                lang
            )
        elif action == "delete_event":
            return _t(
                f"Delete '{event_data['title']}' from your calendar? Yes or no.",
                f"'{event_data['title']}' aus dem Kalender löschen? Ja oder Nein.",
                f"¿Eliminar '{event_data['title']}' del calendario? Sí o no.",
                lang
            )


# ===================================
# BULB CONTROL
# ===================================

_BULB_PARSE_PROMPT = """\
The user wants to control a smart light bulb. Parse their command into a JSON object.

Possible actions:
- {{"action": "on"}}
- {{"action": "off"}}
- {{"action": "brightness", "value": <1-100>}}
- {{"action": "color_temp", "value": <2500-6500>}}   // warm=2700, neutral=4000, cool=6000
- {{"action": "color", "hue": <0-360>, "saturation": <0-100>}}
  // red=0, orange=30, yellow=60, green=120, teal=180, blue=240, purple=270, pink=300
- {{"action": "unknown"}}   // if the command is not clear

User command: {command}
Reply with ONLY a valid JSON object, nothing else."""


def handle_bulbs(question: str, lang: str = "en") -> str:
    """Parse a natural-language bulb command and execute it."""
    raw = llm_invoke(_BULB_PARSE_PROMPT.format(command=question))

    try:
        cmd = json.loads(raw)
    except Exception:
        return _t(
            "Sorry, I couldn't understand that light command.",
            "Entschuldigung, diesen Lichtbefehl habe ich nicht verstanden.",
            "Lo siento, no entendí ese comando de luz.",
            lang
        )

    action = cmd.get("action", "unknown")
    try:
        if action == "on":
            bulb_controller.turn_on()
            return _t("Light on.", "Licht an.", "Luz encendida.", lang)

        elif action == "off":
            bulb_controller.turn_off()
            return _t("Light off.", "Licht aus.", "Luz apagada.", lang)

        elif action == "brightness":
            level = int(cmd.get("value", 50))
            bulb_controller.set_brightness(level)
            return _t(f"Brightness set to {level}%.", f"Helligkeit auf {level}% gesetzt.", f"Brillo al {level}%.", lang)

        elif action == "color_temp":
            kelvin = int(cmd.get("value", 4000))
            bulb_controller.set_color_temperature(kelvin)
            label = "warm" if kelvin < 3500 else "cool" if kelvin > 5000 else "neutral"
            return _t(f"Light set to {label} white.", f"Licht auf {label}es Weiß gesetzt.", f"Luz en blanco {label}.", lang)

        elif action == "color":
            hue = int(cmd.get("hue", 0))
            sat = int(cmd.get("saturation", 100))
            bulb_controller.set_color(hue, sat)
            return _t("Color updated.", "Farbe geändert.", "Color actualizado.", lang)

        else:
            return _t(
                "I didn't catch that — try saying 'turn on', 'dim to 30%', or 'set it to blue'.",
                "Das habe ich nicht verstanden — sag zum Beispiel 'einschalten', 'auf 30% dimmen' oder 'blau'.",
                "No entendí — prueba 'encender', 'atenuar al 30%' o 'color azul'.",
                lang
            )

    except RuntimeError as e:
        print(f"❌ Bulb error: {e}")
        return _t(
            "Couldn't reach the bulb — check TAPO_EMAIL, TAPO_PASSWORD and TAPO_BULB_IP in settings.",
            "Die Lampe war nicht erreichbar — prüfe TAPO_EMAIL, TAPO_PASSWORD und TAPO_BULB_IP.",
            "No se pudo alcanzar la bombilla — verifica TAPO_EMAIL, TAPO_PASSWORD y TAPO_BULB_IP.",
            lang
        )
    except Exception as e:
        print(f"❌ Bulb error: {e}")
        return _t(
            "Something went wrong controlling the light.",
            "Beim Steuern des Lichts ist etwas schiefgelaufen.",
            "Algo salió mal al controlar la luz.",
            lang
        )


# ===================================
# MAIN ASK FUNCTION
# ===================================

RELEVANCE_THRESHOLD = 0.93


# ── Answer language = the language of the message ────────────────────────────────────
# Clients send their UI language as `lang`, but the user may write in another one ("Wie wird
# das Wetter morgen?" with the app set to English). Every prompt that says "Respond in …" and
# every fixed reply (_t) uses `lang`, so it is set once here, from the message itself.
# Only words that are unambiguous between the three languages are used.
_LANG_WORDS = {
    "de": {"ich", "du", "wie", "ist", "das", "der", "die", "und", "nicht", "wird", "morgen", "heute",
           "wetter", "mein", "meine", "meinen", "habe", "hast", "bin", "bist", "wo", "wann", "warum",
           "welche", "welcher", "gibt", "ein", "eine", "einen", "auf", "mit", "für", "zu", "mir", "mich",
           "dich", "kannst", "erinnere", "bitte", "heißt", "sind", "haben", "noch", "schon", "ja", "nein"},
    "es": {"el", "la", "los", "las", "que", "qué", "cómo", "como", "dónde", "donde", "cuándo", "cuando",
           "mañana", "hoy", "tiempo", "clima", "mi", "mis", "tengo", "soy", "estoy", "está", "una", "para",
           "por", "con", "y", "del", "puedes", "recuérdame", "cuál", "cual", "quién", "hace", "sí"},
    "en": {"the", "is", "what", "how", "where", "when", "my", "do", "does", "will", "it", "weather",
           "tomorrow", "today", "and", "of", "to", "for", "you", "are", "have", "who", "which", "can",
           "remind", "please", "yes"},
}


def _detect_language(text: str, fallback: str) -> str:
    """'en', 'de' or 'es' from the message itself; `fallback` (the UI language) when unclear."""
    words = re.findall(r"[a-zäöüßñáéíóú]+", text.lower())
    scores = {lang: sum(w in vocab for w in words) for lang, vocab in _LANG_WORDS.items()}
    if re.search(r"[äöüß]", text.lower()):
        scores["de"] += 2
    if re.search(r"[ñ¿¡]", text.lower()):
        scores["es"] += 2
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] == 0 or ranked[0] == ranked[1]:
        return fallback
    return best


def ask(question: str, mode: str = "auto", lang: str = "en", device_id: str = "app", voice: bool = False) -> str:
    detected = _detect_language(question, lang)
    if detected != lang:
        print(f"🌐 Message language: {detected} (client language was {lang}) — answering in {detected}")
        lang = detected
    print(f"\nYou: {question} [mode: {mode}] [lang: {lang}] [device: {device_id}] [voice: {voice}]")
    token = _current_device.set(device_id)
    token_v = _current_voice.set(voice or device_id == "s_phone")
    token_q = _current_question.set(question)
    try:
        answer = _ask_internal(question, mode, lang)
        return answer
    except RateLimitError:
        print("❌ Groq rate limit reached")
        return _t(
            "Heads up — Groq's usage limit has been reached. "
            "You can switch to a local model in Settings to keep going.",
            "Hinweis — Groqs Nutzungslimit wurde erreicht. "
            "Du kannst in den Einstellungen auf ein lokales Modell wechseln.",
            "Aviso — se ha alcanzado el límite de uso de Groq. "
            "Puedes cambiar a un modelo local en Configuración para continuar.",
            lang
        )
    except Exception as e:
        print(f"❌ Unexpected error in ask(): {e}")
        raise
    finally:
        _current_device.reset(token)
        _current_voice.reset(token_v)
        _current_question.reset(token_q)


_SAVE_INTENTS = {"create_reminder", "create_event", "save_personal_note", "create_list"}

# Read-only intents that personal mode answers from RAG instead of their own handler.
_PERSONAL_RAG_INTENTS = {"general", "view_events", "view_reminders", "view_agenda", "view_emails"}

# Words that show the user really wants something saved / scheduled (en, de, es).
_EXPLICIT_SAVE_RE = re.compile(
    r"\b(remind|reminder|remember|write (it |this |that )?down|note (it|this|that|down)|save|add|put|"
    r"schedule|calendar|book|don't let me forget|do not let me forget|journal|diary|list|"
    r"erinner|merk|notier|speicher|trag|kalender|termin eintragen|vergiss nicht|tagebuch|liste|"
    r"recu[eé]rd|recordatorio|apunta|anota|guarda|agenda|calendario|no me dejes olvidar|diario|lista)",
    re.IGNORECASE)
# Messages that open by talking about oneself ("I have…", "I'm…", "My flight…") are sharing,
# not giving an instruction — unlike bare to-dos such as "buy flowers on Sunday".
_FIRST_PERSON_START_RE = re.compile(
    r"^\s*(i|i'm|im|i've|i'll|my|we|we're|our|ich|mein|meine|wir|yo|tengo|estoy|mi|mis|voy)\b",
    re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^\s*(what|how|why|when|where|which|who|do|does|did|can|could|should|would|will|is|are|any|"
    r"was|wie|warum|wann|wo|welche|kannst|hast|soll|gibt|qué|que|cómo|como|por qué|cuándo|dónde|puedes|tienes|hay)\b",
    re.IGNORECASE)


def _looks_like_question(text: str) -> bool:
    """True if any sentence of the message is a question or asks for advice."""
    if "?" in text or re.search(r"\b(tips?|advice|ideas?|suggestions?|tipps?|rat|consejos?|ideas?)\b", text, re.IGNORECASE):
        return True
    return any(_QUESTION_START_RE.match(s) for s in re.split(r"[.!\n]+", text) if s.strip())


def _ask_internal(question: str, mode: str = "auto", lang: str = "en") -> str:

    # Auto RAG selector on: the chat modes (auto/personal/general) are ignored — Chatette
    # decides herself, per question, whether to look in personal data (see _handle_general).
    # Phone key intents (create_reminder, …) and control_bulbs are not chat modes and stay.
    if AUTO_RAG_SELECTOR and mode in ("auto", "personal", "general"):
        if mode != "auto":
            print(f"↪️ Auto RAG selector on: '{mode}' mode ignored")
        mode = "auto"

    # 0. Hardware shortcut — skip classification for direct-intent modes
    if mode == "control_bulbs":
        answer = handle_bulbs(question, lang)
        print(f"Chatette: {answer}")
        _remember(question, answer)
        return answer

    # The app's Web search button: the message is a web search, no classification needed.
    if mode == "web_search":
        answer = handle_web_search(question, {}, lang)
        print(f"Chatette: {answer}")
        _remember(question, answer)
        return answer

    # 1. Pending confirmation — check before classification
    if _pending().action is not None:
        if _pending().action in DRAFT_STAGES:
            answer = handle_draft_followup(question, lang)
        else:
            answer = handle_confirmation(question, lang)
        print(f"Chatette: {answer}")
        _remember(question, answer)
        return answer

    # 2. Classify intent
    classification = classify_intent(question, lang)
    intent = classification.intent
    confidence = classification.confidence
    extracted = classification.extracted

    # 2a. Guard against "saving" small talk: small models file anything with a future date
    # ("I'm having surgery in two days, any tips?") as a reminder/event/list. Without an
    # explicit request to save something, questions and first-person statements go to
    # normal conversation instead.
    if (intent in _SAVE_INTENTS and not _EXPLICIT_SAVE_RE.search(question)
            and (_looks_like_question(question) or _FIRST_PERSON_START_RE.match(question))):
        print(f"↪️ '{intent}' overridden to 'general' — no explicit request to save anything")
        intent, extracted = "general", {}

    # 2b. Phone shortcut hard-override — trust the key press, not the LLM classification.
    # Extraction still ran above so handler-specific data (text, title, etc.) is available.
    # Key 0 ('general'/'free') is intentionally excluded — it defers to LLM classification.
    # LLM capabilities are limited in the phone, but the goal is intended for quick actions.
    _PHONE_INTENTS = {
        "create_reminder", "create_event", "create_draft", "get_weather",
        "create_list", "view_agenda", "view_emails", "set_alarm",
    }
    if mode in _PHONE_INTENTS:
        intent = mode
        confidence = "high"

    # 2c. Personal mode: every question is answered from the user's own data — the best
    # RAG chunks across calendar, emails, notes and lists — instead of the specialised
    # "show me…" handlers (e.g. "when is my birthday?" was sent to view_events and came
    # back as a calendar listing). Actions (create/delete/lights/TV…) are unaffected.
    if mode == "personal" and intent in _PERSONAL_RAG_INTENTS:
        if intent != "general":
            print(f"↪️ Personal mode: '{intent}' answered from personal data (RAG)")
        intent, confidence = "general", "high"

    # 3. Low confidence on action intents → ask for clarification
    if confidence == "low" and intent != "general":
        answer = _t(
            "I'm not quite sure what you'd like me to do — could you rephrase?",
            "Ich bin nicht ganz sicher, was du möchtest — könntest du es umformulieren?",
            "No estoy segura de lo que quieres — ¿puedes reformularlo?",
            lang
        )
        print(f"Chatette: {answer}")
        _remember(question, answer)
        return answer

    # 4. Dispatch to handler
    if intent == "create_reminder":
        answer = handle_reminder(question, extracted, lang)
    elif intent == "delete_reminder":
        answer = handle_delete(question, extracted, lang)
    elif intent == "delete_event":
        answer = handle_delete_event(question, extracted, lang)
    elif intent == "view_reminders":
        answer = handle_view_reminders(question, lang)
    elif intent == "view_events":
        answer = handle_view_events(question, lang)
    elif intent == "view_emails":
        answer = handle_view_emails(question, extracted, lang)
    elif intent == "view_agenda":
        answer = handle_view_agenda(question, lang)
    elif intent == "get_weather":
        answer = handle_weather(question, extracted, lang)
    elif intent == "web_search" and _EXPLICIT_SEARCH_RE.search(question):
        answer = handle_web_search(question, extracted, lang)
    elif intent == "find_places":
        answer = handle_places(question, extracted, lang)
    elif intent == "about_chatette":
        answer = handle_about_chatette(question, lang)
    elif intent == "control_bulbs":
        answer = handle_bulbs(question, lang)
    elif intent == "create_event":
        answer = handle_calendar_event(question, extracted, lang)
    elif intent == "save_personal_note":
        answer = handle_personal_note(question, extracted, lang)
    elif intent == "create_draft":
        # Typed chat gets the guided flow (questions first); voice drafts are written right away.
        answer = handle_draft(question, extracted, lang, guided=not _current_voice.get())
    elif intent == "create_list":
        answer = handle_create_list(question, extracted, lang)
    elif intent == "add_to_list":
        answer = handle_add_to_list(question, extracted, lang)
    elif intent == "remove_from_list":
        answer = handle_remove_from_list(question, extracted, lang)
    elif intent == "delete_list":
        answer = handle_delete_list(question, extracted, lang)
    elif intent == "set_alarm":
        answer = handle_set_alarm(question, extracted, lang)
    elif intent == "cast_tv_power":
        answer = handle_cast_tv_power(question, extracted, lang)
    elif intent == "cast_volume":
        answer = handle_cast_volume(question, extracted, lang)
    elif intent == "cast_youtube":
        answer = handle_cast_youtube(question, extracted, lang)
    elif intent == "cast_channel":
        answer = handle_cast_channel(question, extracted, lang)
    elif intent == "cast_stop":
        answer = handle_cast_stop(question, lang)
    else:
        answer = _handle_general(question, mode, lang)

    print(f"Chatette: {answer}")
    _remember(question, answer)
    return answer


_ACK_WORDS = {
    "yes", "yeah", "yep", "no", "nope", "ok", "okay", "sure", "thanks", "thank", "you", "great",
    "cool", "nice", "perfect", "fine", "good", "alright",
    "ja", "nein", "danke", "schön", "gut", "super", "klar", "alles", "genau", "prima", "passt",
    "sí", "si", "gracias", "vale", "perfecto", "bueno", "claro", "genial", "muchas",
}


def _is_acknowledgement(text: str) -> bool:
    """True for short replies made only of yes/no/thanks words ("Ja", "ok danke", "¡gracias!")."""
    words = re.findall(r"\w+", text.lower())
    return 0 < len(words) <= 3 and all(w in _ACK_WORDS for w in words)


def _handle_general(question: str, mode: str, lang: str) -> str:
    """Handle general questions via RAG or pure LLM."""

    _general_prompt_base = (
        f"{_persona_prompt(lang)}\n\n"
        f"{_build_conversation_context()}"
        f"Answer the following question in 1-2 sentences.\n"
        f"- Always respond in {_LANG_NAMES.get(lang, 'English')}.\n"
        f"- If the question refers to the previous exchange, use it.\n"
        f"\n"
        f"Question: {{question}}\n"
        f"Chatette:"
    )

    if mode == "general":
        print("Mode: general knowledge (+ calendar context)")
        cal_scored = [(doc, score) for doc, score in _retrieve_scored(question, collections=("calendar",))
                      if score <= RELEVANCE_THRESHOLD]
        if cal_scored:
            return _rag_query(question, scored=cal_scored, lang=lang)
        return llm_invoke(_general_prompt_base.format(question=question)).strip()

    if mode == "personal":
        print("Mode: personal/RAG")
        return _rag_query(question, lang=lang)

    if _is_acknowledgement(question):
        # "Ja", "ok danke"… carry no meaning to search for — their embedding sits vaguely
        # close to everything, so the cut-off let random emails through. Reply from the
        # recent conversation only.
        print("Mode: auto → acknowledgement, no search")
        return llm_invoke(_general_prompt_base.format(question=question)).strip()

    if AUTO_RAG_SELECTOR:
        about_user = _ask_about_user(question)
        print(f"🔎 Auto RAG selector: about_user={about_user}")
        if about_user:
            # Which sources? Emails/calendar only when the question needs them — otherwise
            # notes only, so emails and calendar don't crowd the context (tested as variant F:
            # same accuracy as searching everything, ~73% notes in the context instead of ~51%).
            if _ask_needs_agenda(question):
                print("Mode: auto → about_user=true, agenda=true → notes + emails + calendar")
                return _rag_query(question, lang=lang)
            print("Mode: auto → about_user=true, agenda=false → notes only")
            answer = _rag_query(question, scored=_retrieve_scored(question, collections=("notes",)), lang=lang)
            if _says_not_found(answer):
                # Fallback: the agenda call can miss (e.g. "which company sent me a notice…")
                print("↪️ Not found in notes → searching all sources")
                answer = _rag_query(question, lang=lang)
            return answer
        # about_user=false: fall through to the relevance cut-off below (as in plain auto mode)

    scored = _retrieve_scored(question)
    best_score = min((score for _, score in scored), default=float("inf"))
    print(f"Best relevance score: {best_score:.3f}")

    if best_score > RELEVANCE_THRESHOLD:
        print("Using general knowledge...")
        return llm_invoke(_general_prompt_base.format(question=question)).strip()

    answer = _rag_query(question, scored=scored, lang=lang)
    if _says_not_found(answer):
        # The cut-off only says *something* in the personal data looked similar. If the answer
        # isn't there, it was probably a general question ("In which city were the 2024 Olympics
        # held?" → "I don't have that in my notes") — answer it from general knowledge instead.
        print("↪️ Not found in personal data → general knowledge")
        # Guard: the question may still be personal (the yes/no call can miss, e.g. "What does
        # my partner do for a living?") — general knowledge must never invent personal details.
        guarded = _general_prompt_base.replace(
            "Question:",
            f"- Answer general/world-knowledge questions normally. But if the question is about "
            f"{USER_NAME} or people in his life (family, partner, friends) and you don't know the "
            f"answer, say you don't have that information — never invent personal details.\n\nQuestion:", 1)
        return llm_invoke(guarded.format(question=question)).strip()
    return answer


_NOT_FOUND = re.compile(
    r"don'?t (have|know|see|find)|do not (have|know|see|find)|no (information|record|details|mention)|"
    r"not (in|mentioned in) (my|your) notes|couldn'?t find|can'?t find|isn'?t (in|mentioned)|"
    r"keine informationen|habe ich nicht|nicht in (meinen|deinen) notizen|"
    r"no tengo|no encuentro|no aparece|no est[aá] en (mis|tus) notas",
    re.IGNORECASE)


def _says_not_found(answer: str) -> bool:
    """True when an answer says the information isn't in the personal data."""
    return bool(_NOT_FOUND.search(answer or ""))


_AGENDA_PROMPT = """Answer about this question from {user} to his personal assistant.
Does answering it need information from his EMAILS or CALENDAR — appointments, meetings,
interviews, invitations, scheduled events, messages or letters he received?
Things from his own life that are not scheduled or sent to him (facts about himself, his family,
his history, his health, past trips, routines) do NOT count.

Question: "{q}"
Reply with only: yes or no"""


def _ask_needs_agenda(question: str) -> bool:
    """Second small call of the auto RAG selector: does the answer live in emails/calendar?"""
    try:
        reply = llm_invoke(_AGENDA_PROMPT.format(user=USER_NAME, q=question)).strip().lower()
        return reply.startswith("yes")
    except RateLimitError:
        raise
    except Exception as e:
        print(f"⚠️ agenda call failed ({e}) — searching all sources")
        return True


# ── Web search ──────────────────────────────────────────────────────────────────────
# Only on explicit request ("search the web for…") or the app's Web search button. Two small
# calls before searching: is the information time-sensitive (→ only the last 3 days), and a
# short search-engine query. The query leaves the house (DuckDuckGo), so it may contain only
# the public topic — never the user's name, people from their life or anything from their notes.

_CURRENT_PROMPT = """Should a web search for this question be limited to the LAST 3 DAYS?

Yes — only when the answer is about something that just happened or is changing day by day:
news, recent events, results (elections, matches), today's prices, rates or scores, anything
asked as "latest", "yesterday", "this week", "right now".
No — everything else: history, definitions, how-tos, recipes, biographies, reviews and
recommendations, facts about places, and future events whose dates are already known.

Examples: "US election results yesterday" → yes. "current EUR to USD rate" → yes.
"next solar eclipse in Europe" → no. "good Python courses" → no. "history of Cologne
Cathedral" → no. "tamales recipe" → no.

Question: "{q}"
Reply with only: yes or no"""

_SEARCH_QUERY_PROMPT = """{context}Turn this request into a short web search query (3-8 keywords) that a search engine
answers well.

Request: "{q}"

Rules:
- Leave out the instruction itself ("search the web for", "google", "look up", "online",
  "im Internet", "such nach", "busca en internet").
- Never add a date, month or year the user did not write. Leave out relative time words
  (yesterday, today, latest, this week, gestern, ayer) — the search is already limited to
  recent results when that matters.
- Only the public topic: never include the user's name, names of people in their life
  (partner, family, friends, colleagues), or anything personal from the conversation. Keep
  public names — companies, places, public figures.
- Use the language of the request, unless the topic is clearly international and English
  finds it better.

Examples:
"search the web for the election results in the US yesterday" → US election results
"busca en internet la receta de tamales" → receta tamales
"Such im Internet nach den Öffnungszeiten vom Kölner Dom" → Kölner Dom Öffnungszeiten
"google the latest iPhone price" → iPhone price
"search the web for what my friend Tom's company Brightline does" → Brightline company

Reply with only the query."""


# Words that show the user explicitly wants a web search (en, de, es). Without one, a message the
# classifier sent to web_search ("Who won the election yesterday?") stays an ordinary question.
_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(web|internet|online|google\w*|search\w*|look (it |that |this )?up|"
    r"netz|such\w*|recherch\w*|googel\w*|"
    r"busca\w*|búsca\w*|búsqueda|en (la )?red)\b",
    re.IGNORECASE)


# Names that follow a relationship word ("my partner Anna", "mein Bruder Tom", "mi hermana Lena")
# belong to the user's private life. They are removed from the search query in code — the model
# alone sometimes keeps them.
_RELATION_WORDS = (
    r"partner|girlfriend|boyfriend|wife|husband|fianc[eé]e?|brother|sister|mother|mom|mum|father|dad|"
    r"son|daughter|uncle|aunt|cousin|grandma|grandmother|grandpa|grandfather|friend|colleague|boss|"
    r"neighbou?r|roommate|"
    r"partnerin|freundin|freund|frau|mann|bruder|schwester|mutter|mama|vater|papa|sohn|tochter|onkel|"
    r"tante|cousine?|oma|opa|kollegin|kollege|chef|chefin|nachbarin|nachbar|"
    r"pareja|novia|novio|esposa|esposo|mujer|marido|hermana|hermano|madre|mam[aá]|padre|pap[aá]|hija|"
    r"hijo|t[ií]a|t[ií]o|prima|primo|abuela|abuelo|amiga|amigo|colega|jefa|jefe|vecina|vecino"
)
_PRIVATE_NAME_RE = re.compile(
    r"\b(?:my|our|mein\w*|unser\w*|mi|mis|nuestr\w*)\s+(?:" + _RELATION_WORDS + r")(?:'s)?,?\s+"
    r"([A-ZÄÖÜÁÉÍÓÚÑ][\w\-]+)",
    re.IGNORECASE)


def _private_names(text: str) -> set[str]:
    names = {m.group(1) for m in _PRIVATE_NAME_RE.finditer(text) if m.group(1)[0].isupper()}
    if USER_NAME:
        names.add(USER_NAME)
    return names


def _strip_private_names(query: str, question: str) -> str:
    """Remove the user's name and names of people in their life from a search query."""
    for name in _private_names(question):
        query = re.sub(rf"\b{re.escape(name)}(?:'s|s)?\b", "", query, flags=re.IGNORECASE)
    return " ".join(query.split())


def _needs_current_info(question: str) -> bool:
    try:
        return llm_invoke(_CURRENT_PROMPT.format(q=question)).strip().lower().startswith("yes")
    except RateLimitError:
        raise
    except Exception as e:
        print(f"⚠️ 'current?' call failed ({e}) — searching without a time limit")
        return False


def _search_query(question: str) -> str:
    prompt = _SEARCH_QUERY_PROMPT.format(context=_build_conversation_context(), q=question)
    query = llm_invoke(prompt).strip().splitlines()[0].strip().strip('"').strip("'")
    query = _strip_private_names(query, question)
    return query or _strip_private_names(question, question)


def handle_web_search(question: str, extracted: dict, lang: str = "en") -> str:
    """Search the web (DuckDuckGo) and answer from the results, with short source links."""
    from web_search import SearchUnavailable, search

    current = _needs_current_info(question)
    query = _search_query(question)
    print(f"🌐 Web search: query='{query}' current={current}")
    try:
        results = search(query, current=current, lang=lang)
    except SearchUnavailable as e:
        print(f"⚠️ Web search unavailable: {e}")
        return _t("I can't reach the web search right now — please try again in a few minutes.",
                  "Ich komme gerade nicht an die Websuche — versuch es bitte in ein paar Minuten nochmal.",
                  "Ahora mismo no puedo acceder a la búsqueda web — inténtalo de nuevo en unos minutos.", lang)
    print(f"🌐 {len(results)} result(s): {[r.source for r in results]}")
    if not results:
        return _t(
            "I couldn't find anything on the web for that. The search engine may also be limiting "
            "requests right now — try again later or rephrase it.",
            "Dazu habe ich im Web nichts gefunden. Die Suchmaschine bremst vielleicht auch gerade "
            "Anfragen aus — versuch es später nochmal oder formuliere es anders.",
            "No encontré nada en la web sobre eso. Puede que el buscador esté limitando las "
            "peticiones ahora — inténtalo más tarde o reformúlalo.",
            lang)

    blocks = []
    for i, r in enumerate(results, 1):
        when = f" ({r.date.astimezone().strftime('%A %d %B, %H:%M')})" if r.date else ""
        blocks.append(f"[{i}] {r.title} — {r.source}{when}\n{r.snippet}")
    voice = _current_voice.get()
    length = "1-2 short sentences" if voice else "2-4 sentences"
    prompt = f"""{_persona_prompt(lang)}
Today is {datetime.now().strftime("%A, %B %d, %Y, %H:%M")}.
{_build_conversation_context()}The user asked: "{question}"

Web search results:
{chr(10).join(blocks)}

Answer in {length}, using only these results. If they don't answer the question, say you
couldn't find it. For news, say when it happened or how recent it is. Don't mention result
numbers, sources or links — they are added separately.
Always respond in {_LANG_NAMES.get(lang, "English")}.
Chatette:"""
    answer = llm_invoke(prompt).strip()
    if voice:
        return answer

    links, seen = [], set()
    for r in results:
        if r.source not in seen and r.url:
            seen.add(r.source)
            links.append(f"[{r.source}]({r.url})")
        if len(links) == 3:
            break
    label = _t("Sources", "Quellen", "Fuentes", lang)
    return f"{answer}\n\n{label}: " + " · ".join(links)


RAG_COLLECTIONS = ("calendar", "emails", "notes", "lists", "documents")
MAX_CONTEXT_CHARS = 2400


# Cross-language search: the notes are mostly English, and a German or Spanish question
# matches German/Spanish emails and calendar entries far better than English notes
# ("Wo habe ich diesen Sommer meinen Urlaub verbracht?" ranked the English vacation note
# #12 and dropped it). Non-English questions are therefore also searched in English.
_NON_ENGLISH_HINT = re.compile(
    r"[äöüßñ¿¡]|\b(ich|du|mein|meine|meinen|habe|hast|bin|bist|wo|wann|wie|warum|welche|"
    r"und|nicht|eine?|yo|mis?|tengo|soy|estoy|d[oó]nde|cu[aá]ndo|qu[eé]|c[oó]mo|los|las|por)\b",
    re.IGNORECASE)
_translation_cache: dict[str, str] = {}


def _search_queries(question: str) -> list[str]:
    """The question itself, plus an English translation when it looks non-English."""
    if len(_NON_ENGLISH_HINT.findall(question)) < 2 and not re.search(r"[äöüßñ¿¡]", question):
        return [question]
    if question not in _translation_cache:
        try:
            english = llm_invoke(
                "Translate this question into English. Keep names as they are. "
                f"Reply with only the translation.\n\n{question}").strip().strip('"')
        except Exception as e:
            print(f"⚠️ Search translation failed ({e}) — searching with the original only")
            english = ""
        _translation_cache[question] = english
        if len(_translation_cache) > 200:
            _translation_cache.pop(next(iter(_translation_cache)))
    english = _translation_cache[question]
    if english and english.lower() != question.lower():
        print(f"🌐 Also searching in English: {english!r}")
        return [question, english]
    return [question]


def _retrieve_scored(question: str, collections=RAG_COLLECTIONS,
                     k_emails: int = 7, k_other: int = 5) -> list[tuple]:
    """Top-k (doc, distance) pairs from each collection, for the question and (if it is not
    English) its English translation. Returned best-first; each chunk keeps its best distance
    (lower = more relevant) for logging and the relevance cut-off.

    With two queries, the order comes from reciprocal rank fusion (rank-based merging):
    distances from different queries aren't comparable — a German question scores German
    emails higher than its English translation scores the English notes, so sorting by
    distance let German emails crowd out the notes ("Wie heißt mein Bruder?").
    """
    queries = _search_queries(question)
    best: dict[str, tuple] = {}
    fused: dict[str, float] = {}
    for query in queries:
        ranked = sorted(_search_once(query, collections, k_emails, k_other), key=lambda p: p[1])
        for rank, (doc, score) in enumerate(ranked):
            key = doc.page_content.strip()
            if key not in best or score < best[key][1]:
                best[key] = (doc, score)
            fused[key] = fused.get(key, 0.0) + 1.0 / (60 + rank)
    if len(queries) == 1:
        return sorted(best.values(), key=lambda p: p[1])
    return [best[k] for k in sorted(fused, key=fused.get, reverse=True)]


def _search_once(question: str, collections, k_emails: int, k_other: int) -> list[tuple]:
    """Top-k (doc, distance) pairs from each collection for one query string."""
    scored = []
    for collection in collections:
        k = k_emails if collection == "emails" else k_other
        while k >= 1:
            try:
                scored += vectorstore.similarity_search_with_score(
                    question, k=k, filter={"collection": collection})
                break
            except Exception as e:
                # The HNSW index can fail to return k hits ("ef or M is too small");
                # retry with fewer instead of silently giving up on the collection.
                if "contiguous" in str(e).lower() or "ef" in str(e).lower():
                    print(f"⚠️ '{collection}': search for k={k} failed ({e}) — retrying with fewer")
                    k //= 2
                else:
                    print(f"Could not retrieve from '{collection}': {e}")
                    break
    return scored


def _select_chunks(scored: list[tuple], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[list, list]:
    """Take chunks in the given best-first order (as returned by _retrieve_scored), drop
    exact duplicates, and fill the context budget best-first.

    Returns (selected_docs, rows) where rows describe every candidate for the log.
    """
    selected, rows, seen, used = [], [], set(), 0
    for doc, score in scored:
        text = doc.page_content.strip()
        if text in seen:
            rows.append((doc, score, "DUPLICATE (skipped)"))
            continue
        seen.add(text)
        cost = len(text) + (2 if selected else 0)          # "\n\n" separator
        if used + cost <= max_chars:
            selected.append(doc)
            used += cost
            rows.append((doc, score, "SENT"))
        else:
            rows.append((doc, score, "DROPPED (over limit)"))
    return selected, rows


def _log_rag_chunks(question: str, rows: list, max_chars: int) -> None:
    """Print every candidate chunk to the session log, best first, and whether it reached the LLM.
    Scores are Chroma distances: lower = more relevant."""
    sent = sum(1 for *_, status in rows if status == "SENT")
    print(f"\n📚 RAG chunks for: {question!r} — {len(rows)} candidates, {sent} sent "
          f"(best first, context limit {max_chars} chars)")
    for i, (doc, score, status) in enumerate(rows, 1):
        meta = doc.metadata or {}
        print(f"  [{i:02d}] {status:<22} score {score:.3f} | "
              f"{meta.get('collection', '?')}/{meta.get('source_file', '?')}")
        text = " ".join(doc.page_content.split())          # one line, whitespace collapsed
        print("       " + (text[:60] + " [...]" if len(text) > 60 else text))
    print("📚 end of chunks\n")


def _rag_query(question: str, scored: list[tuple] | None = None, lang: str = "en") -> str:
    """Answer from the most relevant chunks. `scored` = (doc, distance) pairs; retrieved
    from every collection when not given."""
    if scored is None:
        scored = _retrieve_scored(question)
    docs, rows = _select_chunks(scored)
    _log_rag_chunks(question, rows, MAX_CONTEXT_CHARS)
    context = "\n\n".join(doc.page_content.strip() for doc in docs)

    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y at %I:%M %p")
    tomorrow = (now + timedelta(days=1)).strftime("%A, %B %d, %Y")
    day_after = (now + timedelta(days=2)).strftime("%A, %B %d, %Y")

    date_facts = _date_facts(USER_PROFILE or "", context)

    context = f"""Today is: {current_date}
Tomorrow is: {tomorrow}
The day after tomorrow is: {day_after}
{date_facts}
{_build_conversation_context()}{context}"""

    formatted_prompt = prompt_template.format(
        context=context, question=question,
        user_name=USER_NAME, user_profile=USER_PROFILE,
        lang_name=_LANG_NAMES.get(lang, "English")
    )
    return llm_invoke(formatted_prompt).strip()


if __name__ == "__main__":
    ask("Do I have any appointments coming up?")
    ask("What's on my shopping list?")