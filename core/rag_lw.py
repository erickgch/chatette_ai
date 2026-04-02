import os
import json
import re
import threading
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
    global llm, USE_GROQ, GROQ_MODEL, OLLAMA_MODEL, CALENDAR_DAYS_AHEAD, CALENDAR_DAYS_BEHIND
    load_dotenv(override=True)
    USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    CALENDAR_DAYS_AHEAD = int(os.getenv("CALENDAR_DAYS_AHEAD", "14"))
    CALENDAR_DAYS_BEHIND = int(os.getenv("CALENDAR_DAYS_BEHIND", "1"))
    if USE_GROQ:
        from langchain_groq import ChatGroq
        llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=GROQ_MODEL)
        print(f"LLM reloaded: {GROQ_MODEL} via Groq ☁️")
    else:
        llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        print(f"LLM reloaded: {OLLAMA_MODEL} via Ollama 🖥️")


def llm_invoke(prompt: str) -> str:
    """Invoke LLM and always return a plain string."""
    try:
        _prompt = prompt
        if USE_GROQ and "qwen" in GROQ_MODEL.lower():
            _prompt = prompt + " /no_think"

        response = llm.invoke(_prompt)
        text = response.content if hasattr(response, 'content') else str(response)

        # Strip any <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Strip markdown bold/italic added by Qwen
        if USE_GROQ and "qwen" in GROQ_MODEL.lower():
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
- Warm, friendly and competent — like a smart friend who always has things under control.
- You have your own voice. Not a generic AI bot. Not a corporate assistant.
- Direct and accurate. You answer first, add warmth second.
- Occasionally use cool, casual phrases that feel natural — not forced.
- Very rarely, a subtle cat hint slips through — natural, never cheesy.
- Never over-explain or pad responses with unnecessary pleasantries.
- Never apologize unnecessarily.
- Always respond in {lang_name}."""

CHATETTE_RULES = """
STRICT RULES:
- Only answer what was specifically asked. Nothing more. Do not make additional comments.
- Keep answers to 1-2 sentences maximum.
- Never calculate days of the week from memory — derive them from the current date in context.
- Always respond in the same language of the QUESTION, not the context.
- If the question refers to a previous exchange, use it to answer correctly.
- Never invent personal information not in the profile.
- Never refer to {user_name} in third person — use 'you' and 'your'."""


# Custom RAG prompt
prompt_template = PromptTemplate(
    template=CHATETTE_PERSONA + CHATETTE_RULES + """

If you don't find the answer in the context, say so briefly.

Examples:
Context: Today is Monday, March 16, 2026. Calendar event: Dentist on Wednesday, March 18 at 10am.
{user_name}: Do I have anything this week?
Chatette: Heads up — dentist on Wednesday at 10am. You're all set for the rest of the week.

Context: User enjoys technology and building AI assistants.
{user_name}: Can you recommend a podcast?
Chatette: Given your interest in tech and AI — Lex Fridman or TWIML AI Podcast are worth your time.

Context: Previous exchange: User asked about capital of Mexico. Chatette said Mexico City.
{user_name}: since when?
Chatette: Mexico City has been the capital since 1521, founded on the ruins of Tenochtitlan after the Spanish conquest.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question", "user_name", "user_profile"]
)

_current_device: ContextVar[str] = ContextVar("device_id", default="app")

_pending_state: dict[str, "PendingState"] = {}
_conversation_contexts: dict[str, dict] = {}


def _pending() -> "PendingState":
    device = _current_device.get()
    if device not in _pending_state:
        _pending_state[device] = PendingState()
    return _pending_state[device]


def _ctx() -> dict:
    device = _current_device.get()
    if device not in _conversation_contexts:
        _conversation_contexts[device] = {"last_question": None, "last_answer": None}
    return _conversation_contexts[device]


def clear_device_context(device_id: str) -> None:
    if device_id in _conversation_contexts:
        _conversation_contexts[device_id]["last_question"] = None
        _conversation_contexts[device_id]["last_answer"] = None
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
- view_events: user wants to see their calendar events or appointments (e.g. "do I have anything this week?", "what's on my calendar?", "any appointments coming up?", "what are my plans for tomorrow?")
- view_agenda: user wants a combined overview of reminders AND events — a quick picture of what is coming up (e.g. "what do I have going on?", "give me my agenda", "what's coming up this week?", "anything important soon?")
- view_emails: user wants to see recent emails (e.g. "any new emails?", "check my inbox", "what emails did I get?", "did I get any emails?")
- get_weather: user wants weather info (e.g. "what's the weather?", "will it rain today?", "forecast for Berlin")
- about_chatette: user is asking about Chatette (e.g. "who are you?", "what can you do?")
- control_bulbs: user wants to control a smart light bulb (e.g. "turn on the lights", "dim to 50%", "set color to blue", "red lights on", "turn off the lamp")
- set_alarm: user wants to set a countdown timer ("set a timer for 10 minutes", "remind me in 30 seconds") OR an alarm at a specific time ("set an alarm for 7 AM", "wake me up at 6:30", "alarm at 20:00")
- cast_youtube: user wants to play a YouTube video on TV ("play Bonobo on YouTube", "put on some jazz on the TV", "cast this to the TV")
- cast_channel: user wants to watch a live TV channel ("I want to watch ARD", "put on ZDF", "turn on Euronews", "open Arte", "put on Milenio", "TV5 Monde", "NHK World")
- cast_tv_power: user wants to turn the TV on or off ("turn on the TV", "turn off the TV", "switch the TV off")
- cast_volume: user wants to change TV volume ("volume up", "volume down", "set volume to 40", "louder", "quieter", "mute")
- cast_stop: user wants to stop TV playback ("stop the TV", "pause the TV", "stop casting")
- general: anything else — general questions, conversation, advice

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
- get_weather: {{"city": "city name or empty for home", "timeframe": "now|today|tomorrow|week"}}
- view_emails: {{"max_results": 5}} — optional, default 5
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

Reply with ONLY a valid JSON object, nothing else:
{{
  "intent": "intent_name",
  "confidence": "high|medium|low",
  "extracted": {{...}}
}}"""

    try:
        response = llm_invoke(prompt)
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

def _build_conversation_context() -> str:
    """Build previous exchange string if available."""
    if _ctx()["last_answer"]:
        last_answer = _ctx()["last_answer"][:200]
        last_question = _ctx()["last_question"][:100]
        return (
            f"Previous exchange:\n"
            f"User: {last_question}\n"
            f"Chatette: {last_answer}\n\n"
        )
    return ""


def _persona_prompt(lang: str = "en") -> str:
    """Return the base persona prompt with user info filled in."""
    return CHATETTE_PERSONA.format(
        user_name=USER_NAME,
        user_profile=USER_PROFILE,
        lang_name=_LANG_NAMES.get(lang, "English")
    )


# ===================================
# HANDLERS
# ===================================

def _normalise_items(extracted: dict, key: str = "items", fallback_key: str = "text") -> list:
    """Normalise extracted data to always return a list of strings."""
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

"save a reminder due tomorrow to wash white clothes"
→ ["Wash white clothes tomorrow"]

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
    all_reminders = get_all_reminders()
    if all_reminders == "No reminders found.":
        return _t(
            "Nothing on your reminder list.",
            "Keine Erinnerungen vorhanden.",
            "No tienes recordatorios.",
            lang
        )

    previous_context = _build_conversation_context()
    hint = extracted.get("text", "")

    match_prompt = f"""Find the reminder to delete.
{previous_context}
The user wants to delete: "{question}"
Hint from classifier: "{hint}"

Reminders:
{all_reminders}

- Find the best match, even if wording differs
- If the user says "that" or "it", use the previous exchange
- Return ONLY the exact line including the timestamp
- If nothing matches, return NO_MATCH"""

    matched_line = llm_invoke(match_prompt).strip().strip('"').strip("'")
    print(f"LLM matched: '{matched_line}'")

    if "NO_MATCH" in matched_line or not matched_line:
        return _t(
            "Nothing matching that in your reminders. Can you be more specific?",
            "Keine passende Erinnerung gefunden. Kannst du genauer sein?",
            "No encontré un recordatorio que coincida. ¿Puedes ser más específico?",
            lang
        )

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


def handle_draft(question: str, extracted: dict, lang: str = "en") -> str:
    """Generate a draft document."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    draft_type = extracted.get("type", "")
    draft_purpose = extracted.get("purpose", "")

    if not draft_type or not draft_purpose:
        extraction_prompt = f"""Today is {current_date}.
{previous_context}
The user said: "{question}"

Extract the draft type and purpose.
Return ONLY a JSON object:
- type: document type (e.g. "email", "letter", "message")
- purpose: what it's about

Output:"""
        response = llm_invoke(extraction_prompt).strip()
        response = response[response.find("{"):response.rfind("}")+1]
        try:
            draft_info = json.loads(response)
            draft_type = draft_info.get("type", "document")
            draft_purpose = draft_info.get("purpose", question)
        except json.JSONDecodeError:
            draft_type = "document"
            draft_purpose = question

    lang_name = {"en": "English", "de": "German", "es": "Spanish"}.get(lang, "English")
    draft_prompt = f"""Today is {current_date}.
{previous_context}
Write a {draft_type} {draft_purpose} for {USER_NAME}.
- Write entirely in {lang_name}
- Professional and well-structured
- Use [placeholder] for unknown details
- Concise

Write the {draft_type}:"""

    draft_content = llm_invoke(draft_prompt).strip()
    title = f"{draft_type}_{draft_purpose.replace(' ', '_')[:30]}"

    _pending().text = draft_content
    _pending().action = "draft"
    _pending().conflict = None
    _pending().line_to_delete = title
    _pending().event_data = {"type": draft_type, "purpose": draft_purpose}
    _pending().lang = lang
    return f"Here's a draft {draft_type} {draft_purpose}:\n\n{draft_content}\n\n" + _t(
        "Want me to save this?",
        "Soll ich das speichern?",
        "¿Quieres que lo guarde?",
        lang
    )


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


def handle_calendar_event(question: str, extracted: dict, lang: str = "en") -> str:
    """Extract event details and confirm creation."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    extraction_prompt = f"""Today is {current_date}.
{previous_context}The user said: "{question}"
Hint from classifier — title: "{extracted.get('title', '')}", description: "{extracted.get('description', '')}"

Extract calendar event details. Return ONLY a JSON object:
- title: event name
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


def handle_view_events(question: str, lang: str = "en") -> str:
    """Return upcoming calendar events — RAG first, live fallback if empty."""
    current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # Try RAG first (fast, no API call)
    rag_docs = []
    try:
        results = vectorstore.similarity_search(
            question, k=5,
            filter={"collection": "calendar"}
        )
        rag_docs = results
    except Exception as e:
        print(f"Calendar RAG retrieval failed: {e}")

    if rag_docs:
        context = "\n\n".join([doc.page_content for doc in rag_docs])
        events_context = f"Today is: {current_date}\n\n{context}"
    else:
        # RAG empty or stale — hit Google Calendar directly
        print("Calendar RAG empty — fetching live events")
        try:
            # Uses CALENDAR_DAYS_AHEAD and CALENDAR_DAYS_BEHIND from .env
            events = get_upcoming_events(
                days_ahead=CALENDAR_DAYS_AHEAD,
                days_behind=CALENDAR_DAYS_BEHIND
            )
        except Exception as e:
            print(f"Live calendar fetch failed: {e}")
            return _t(
                "I couldn't reach your Google Calendar right now — is the server online?",
                "Ich konnte deinen Google Kalender nicht erreichen — ist der Server online?",
                "No pude acceder a tu Google Calendar — ¿está el servidor activo?",
                lang
            )

        if not events:
            return _t(
                "Nothing on your calendar for the next two weeks.",
                "Keine Termine in den nächsten zwei Wochen.",
                "Nada en tu calendario para las próximas dos semanas.",
                lang
            )

        lines = []
        for e in events:
            lines.append(f"- {e.title} on {e.start}")
        events_context = f"Today is: {current_date}\n\nUpcoming events:\n" + "\n".join(lines)

    format_prompt = f"""{_persona_prompt(lang)}

The user asked: "{question}"

{events_context}

List the relevant upcoming events in a friendly, natural way.
Respond in {_LANG_NAMES.get(lang, 'English')}.
Keep it concise — 1 to 4 events max, summarise the rest if there are more."""

    return llm_invoke(format_prompt).strip()


def handle_view_agenda(question: str, lang: str = "en") -> str:
    """Combined agenda: reminders + upcoming events in one LLM call."""
    current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    # ── Reminders ──
    reminders = get_all_reminders()
    reminders_section = reminders if reminders != "No reminders found." else None

    # ── Events: RAG first, live fallback ──
    events_section = None
    rag_docs = []
    try:
        results = vectorstore.similarity_search(
            question, k=5,
            filter={"collection": "calendar"}
        )
        rag_docs = results
    except Exception as e:
        print(f"Agenda calendar RAG failed: {e}")

    if rag_docs:
        events_section = "\n\n".join([doc.page_content for doc in rag_docs])
    else:
        try:
            events = get_upcoming_events(
                days_ahead=CALENDAR_DAYS_AHEAD,
                days_behind=CALENDAR_DAYS_BEHIND
            )
            if events:
                events_section = "\n".join(
                    f"- {e.title} on {e.start}" for e in events
                )
        except Exception as e:
            print(f"Agenda live calendar fetch failed: {e}")

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
    result = _yt_search.search_video(query)
    if result is None:
        if _yt_search._quota_exceeded:
            return _t("YouTube search is unavailable for today — the daily quota has been reached.", "Die YouTube-Suche ist heute nicht verfügbar — das Tageslimit wurde erreicht.", "La búsqueda de YouTube no está disponible hoy — se alcanzó el límite diario.", lang)
        return _t("I couldn't find that on YouTube.", "Ich konnte das auf YouTube nicht finden.", "No encontré eso en YouTube.", lang)
    video_id, title = result
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

    format_prompt = f"""{_persona_prompt(lang)}

The user asked: "{question}"

{emails_context}

Summarise the relevant emails in a friendly, natural way.
Respond in {_LANG_NAMES.get(lang, 'English')}.
Keep it concise — highlight sender and subject, mention body only if relevant."""

    return llm_invoke(format_prompt).strip()


def handle_weather(question: str, extracted: dict, lang: str = "en") -> str:
    """Fetch weather and return a natural Chatette-style response. Be concise. Do not give much information about the wind."""
    city = extracted.get("city", "").strip()
    timeframe = extracted.get("timeframe", "today").lower()

    if timeframe in ("now", "current", "currently", "jetzt", "aktuell", "ahora"):
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

Here is the current weather data:
{context}

Respond naturally in Chatette's voice — warm, concise, useful.
Highlight the most important things (temperature, rain, wind if notable). Don't be too technical.
Respond in {_LANG_NAMES.get(lang, 'English')}.
Keep it to 2-3 sentences maximum."""

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

        elif action == "personal_note":
            save_personal_note(reminder_text)
            ingest_notes()
            _pending().clear()
            return _t(
                "Done and dusted — added to your personal notes.",
                "Erledigt — zu deinen persönlichen Notizen hinzugefügt.",
                "Listo — añadido a tus notas personales.",
                lang
            )

        elif action == "personal_note_batch":
            items = _pending().items or []
            for item in items:
                save_personal_note(item)
            ingest_notes()
            _pending().clear()
            count = len(items)
            return _t(
                f"Done — {count} note{'s' if count != 1 else ''} added to your personal notes.",
                f"Erledigt — {count} Eintrag/Einträge zu deinen persönlichen Notizen hinzugefügt.",
                f"Listo — {count} nota{'s' if count != 1 else ''} añadida{'s' if count != 1 else ''} a tus notas personales.",
                lang
            )

        elif action == "draft":
            filename = save_draft(line_to_delete, reminder_text)
            ingest_notes()
            _pending().clear()
            return _t(
                f"Saved as '{filename}' in your drafts. You're all set.",
                f"Als '{filename}' in deinen Entwürfen gespeichert.",
                f"Guardado como '{filename}' en tus borradores.",
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
        elif action == "personal_note":
            return _t(
                "Add this to your personal notes? Yes or no.",
                "Das zu deinen persönlichen Notizen hinzufügen? Ja oder Nein.",
                "¿Añadir esto a tus notas personales? Sí o no.",
                lang
            )
        elif action == "draft":
            return _t(
                "Save this draft? Yes or no.",
                "Diesen Entwurf speichern? Ja oder Nein.",
                "¿Guardar este borrador? Sí o no.",
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


def ask(question: str, mode: str = "auto", lang: str = "en", device_id: str = "app") -> str:
    print(f"\nYou: {question} [mode: {mode}] [lang: {lang}] [device: {device_id}]")
    token = _current_device.set(device_id)
    try:
        return _ask_internal(question, mode, lang)
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


def _ask_internal(question: str, mode: str = "auto", lang: str = "en") -> str:

    # 0. Hardware shortcut — skip classification for direct-intent modes
    if mode == "control_bulbs":
        answer = handle_bulbs(question, lang)
        print(f"Chatette: {answer}")
        _ctx()["last_question"] = question
        _ctx()["last_answer"] = answer
        return answer

    # 1. Pending confirmation — check before classification
    if _pending().action is not None:
        answer = handle_confirmation(question, lang)
        print(f"Chatette: {answer}")
        _ctx()["last_question"] = question
        _ctx()["last_answer"] = answer
        return answer

    # 2. Classify intent
    classification = classify_intent(question, lang)
    intent = classification.intent
    confidence = classification.confidence
    extracted = classification.extracted

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

    # 3. Low confidence on action intents → ask for clarification
    if confidence == "low" and intent != "general":
        answer = _t(
            "I'm not quite sure what you'd like me to do — could you rephrase?",
            "Ich bin nicht ganz sicher, was du möchtest — könntest du es umformulieren?",
            "No estoy segura de lo que quieres — ¿puedes reformularlo?",
            lang
        )
        print(f"Chatette: {answer}")
        _ctx()["last_question"] = question
        _ctx()["last_answer"] = answer
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
    elif intent == "about_chatette":
        answer = handle_about_chatette(question, lang)
    elif intent == "control_bulbs":
        answer = handle_bulbs(question, lang)
    elif intent == "create_event":
        answer = handle_calendar_event(question, extracted, lang)
    elif intent == "save_personal_note":
        answer = handle_personal_note(question, extracted, lang)
    elif intent == "create_draft":
        answer = handle_draft(question, extracted, lang)
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
    _ctx()["last_question"] = question
    _ctx()["last_answer"] = answer
    return answer


def _handle_general(question: str, mode: str, lang: str) -> str:
    """Handle general questions via RAG or pure LLM."""

    _general_prompt_base = (
        f"{_persona_prompt(lang)}\n\n"
        f"{_build_conversation_context()}"
        f"Answer the following question in 1-2 sentences.\n"
        f"- Always respond in {_LANG_NAMES.get(lang, 'English')}.\n"
        f"- If the question refers to the previous exchange, use it.\n\n"
        f"Question: {{question}}\n"
        f"Chatette:"
    )

    if mode == "general":
        print("Mode: general knowledge")
        return llm_invoke(_general_prompt_base.format(question=question)).strip()

    if mode == "personal":
        print("Mode: personal/RAG")
        return _rag_query(question, k_emails=7, k_other=5, lang=lang)

    all_docs = []
    best_score = float("inf")

    for collection in ["calendar", "emails", "notes", "lists", "documents"]:
        k = 7 if collection == "emails" else 5
        try:
            results = vectorstore.similarity_search_with_score(
                question, k=k,
                filter={"collection": collection}
            )
            for doc, score in results:
                all_docs.append(doc)
                if score < best_score:
                    best_score = score
        except Exception as e:
            print(f"Could not retrieve from '{collection}': {e}")
            continue

    print(f"Best relevance score: {best_score:.3f}")

    if best_score > RELEVANCE_THRESHOLD or not all_docs:
        print("Using general knowledge...")
        return llm_invoke(_general_prompt_base.format(question=question)).strip()

    return _rag_query(question, k_emails=7, k_other=5, docs=all_docs, lang=lang)


def _rag_query(question: str, k_emails: int = 7, k_other: int = 5,
               docs: list = None, lang: str = "en") -> str:
    """Run RAG query with provided or freshly retrieved docs."""
    if docs is None:
        docs = []
        for collection in ["calendar", "emails", "notes", "lists", "documents"]:
            k = k_emails if collection == "emails" else k_other
            try:
                results = vectorstore.similarity_search(
                    question, k=k,
                    filter={"collection": collection}
                )
                docs.extend(results)
            except Exception as e:
                print(f"Could not retrieve from '{collection}': {e}")
                continue

    context = "\n\n".join([doc.page_content for doc in docs])
    MAX_CONTEXT_CHARS = 2400
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "..."

    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y at %I:%M %p")
    tomorrow = (now + timedelta(days=1)).strftime("%A, %B %d, %Y")
    day_after = (now + timedelta(days=2)).strftime("%A, %B %d, %Y")

    context = f"""Today is: {current_date}
Tomorrow is: {tomorrow}
The day after tomorrow is: {day_after}

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