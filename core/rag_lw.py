import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from ingestion import ingest_notes, ingest_calendar_events, ingest_lists
from note_manager import (
    save_reminder, delete_reminder_by_line, get_all_reminders,
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
DELETE_EVENT_DAYS_AHEAD = int(os.getenv("DELETE_EVENT_DAYS_AHEAD", "60"))

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

def _t(en: str, de: str, es: str, lang: str) -> str:
    """Return the right translation based on language code."""
    if lang == "de":
        return de
    if lang == "es":
        return es
    return en


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
- Always respond in the same language the user used."""

CHATETTE_RULES = """
STRICT RULES:
- Only answer what was specifically asked. Nothing more.
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

conversation_context = {"last_question": None, "last_answer": None}


# ===================================
# PYDANTIC MODELS
# ===================================

VALID_INTENTS = [
    "save_reminder",
    "delete_reminder",
    "create_event",
    "delete_event",
    "save_personal_note",
    "create_draft",
    "create_list",
    "add_to_list",
    "remove_list_item",
    "delete_list",
    "view_reminders",
    "view_events",
    "get_weather",
    "about_chatette",
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


# Module-level pending state instance — mutated in place by all handlers
pending_reminder = PendingState()



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
- save_reminder: user wants to save a reminder (e.g. "remind me to...", "write down...", "note that...")
- delete_reminder: user wants to delete a reminder (e.g. "delete the reminder...", "forget about...")
- create_event: user wants to add a calendar event (e.g. "add to my calendar", "schedule a meeting...")
- delete_event: user wants to remove a calendar event (e.g. "cancel my appointment", "remove the event...", "can you delete the...")
- save_personal_note: user wants to save to personal notes/diary (e.g. "add to my diary", "save to personal notes...")
- create_draft: user wants to compose a document/email/letter (e.g. "compose a...", "write a draft...", "help me write...")
- create_list: user wants to create a new list (e.g. "create a shopping list", "make a list...")
- add_to_list: user wants to add an item to an existing list (e.g. "add milk to my shopping list")
- remove_list_item: user wants to remove an item from a list (e.g. "remove milk from the list")
- delete_list: user wants to delete an entire list (e.g. "delete my shopping list")
- view_reminders: user wants to see their saved reminders or to-do items (e.g. "show my reminders", "what do I have to do?", "what did I write down?")
- view_events: user wants to see their calendar events or appointments (e.g. "do I have anything this week?", "what's on my calendar?", "any appointments coming up?", "what are my plans for tomorrow?")
- get_weather: user wants weather info (e.g. "what's the weather?", "will it rain today?", "forecast for Berlin")
- about_chatette: user is asking about Chatette (e.g. "who are you?", "what can you do?")
- general: anything else — general questions, conversation, advice

For the extracted field, include only what is relevant.
IMPORTANT: If the user mentions multiple items of the same type, extract ALL of them as an array.

- save_reminder: {{"items": ["reminder 1", "reminder 2", ...]}} — ALWAYS use items array, even for one item
- delete_reminder: {{"text": "the reminder description"}}
- create_event: {{"events": [{{"title": "...", "description": "..."}}, ...]}} — ALWAYS use events array, even for one event
- delete_event: {{"title": "event title"}}
- save_personal_note: {{"items": ["note 1", "note 2", ...]}} — ALWAYS use items array
- create_draft: {{"type": "email/letter/message", "purpose": "what it's about"}}
- create_list: {{"title": "list name", "items": ["item1", "item2"]}}
- add_to_list: {{"items": ["item1", "item2", ...], "list_name": "target list"}} — ALWAYS use items array
- remove_list_item: {{"item": "item name", "list_name": "target list"}}
- delete_list: {{"list_name": "list to delete"}}
- get_weather: {{"city": "city name or empty for home", "timeframe": "now|today|tomorrow|week"}}
- view_reminders / view_events / about_chatette / general: {{}}

Examples of multi-item extraction:
User: "remind me to buy flowers, call mom and visit uncle"
→ {{"items": ["Buy flowers", "Call mom", "Visit uncle"]}}

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
    if conversation_context["last_answer"]:
        last_answer = conversation_context["last_answer"][:200]
        last_question = conversation_context["last_question"][:100]
        return (
            f"Previous exchange:\n"
            f"User: {last_question}\n"
            f"Chatette: {last_answer}\n\n"
        )
    return ""


def _persona_prompt() -> str:
    """Return the base persona prompt with user info filled in."""
    return CHATETTE_PERSONA.format(
        user_name=USER_NAME,
        user_profile=USER_PROFILE
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
- Remove trigger phrases like "write down", "remind me", "make a note"
- Keep dates in the text for now
- One item per reminder, no explanations

Examples:
"remind me to buy flowers, call mom and visit uncle"
→ ["Buy flowers", "Call mom", "Visit uncle"]

"write down dentist next Friday"
→ ["Dentist next Friday"]

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
                pending_reminder.text = reminder_text
                pending_reminder.items = None
                pending_reminder.action = "save"
                pending_reminder.conflict = "duplicate"
                pending_reminder.line_to_delete = existing_line
                pending_reminder.due = due_date
                pending_reminder.lang = lang
                return _t(
                    f"Heads up — you already have something similar: '{existing_line}'. Want to replace it with '{reminder_text}'?",
                    f"Hinweis — du hast bereits etwas Ähnliches: '{existing_line}'. Ersetzen mit '{reminder_text}'?",
                    f"Aviso — ya tienes algo similar: '{existing_line}'. ¿Reemplazar con '{reminder_text}'?",
                    lang
                )
            elif conflict_check.startswith("CONFLICT:"):
                existing_line = conflict_check.replace("CONFLICT:", "").strip()
                pending_reminder.text = reminder_text
                pending_reminder.items = None
                pending_reminder.action = "save"
                pending_reminder.conflict = "conflict"
                pending_reminder.line_to_delete = existing_line
                pending_reminder.due = due_date
                pending_reminder.lang = lang
                return _t(
                    f"Quick one — you already have '{existing_line}' around that time. Still want to save '{reminder_text}'?",
                    f"Kurze Frage — du hast bereits '{existing_line}' zu dieser Zeit. Trotzdem '{reminder_text}' speichern?",
                    f"Un momento — ya tienes '{existing_line}' a esa hora. ¿Guardar '{reminder_text}' igualmente?",
                    lang
                )

        pending_reminder.text = reminder_text
        pending_reminder.items = None
        pending_reminder.action = "save"
        pending_reminder.conflict = None
        pending_reminder.line_to_delete = None
        pending_reminder.due = due_date
        pending_reminder.lang = lang
        due_hint = f" (due {due_date})" if due_date else ""
        return _t(
            f"Just to confirm — save this to reminders: '{reminder_text}'{due_hint}?",
            f"Zur Bestätigung — in die Erinnerungen speichern: '{reminder_text}'{due_hint}?",
            f"Para confirmar — ¿guardar esto en recordatorios: '{reminder_text}'{due_hint}?",
            lang
        )

    preview = "\n".join(
        f"  • {d['text']}" + (f" (due {d['due']})" if d.get("due") else "")
        for d in due_items
    )
    pending_reminder.text = None
    pending_reminder.items = due_items
    pending_reminder.action = "save_batch"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = None
    pending_reminder.due = None
    pending_reminder.lang = lang
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

    pending_reminder.text = None
    pending_reminder.action = "delete"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = matched_line
    pending_reminder.lang = lang
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

    pending_reminder.text = event_title
    pending_reminder.action = "delete_event"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = event_id
    pending_reminder.event_data = {"id": event_id, "title": event_title}
    pending_reminder.lang = lang
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
        pending_reminder.text = note_text
        pending_reminder.items = None
        pending_reminder.action = "personal_note"
        pending_reminder.conflict = None
        pending_reminder.line_to_delete = None
        pending_reminder.event_data = None
        pending_reminder.lang = lang
        return _t(
            f"Add this to your personal notes: '{note_text}'?",
            f"Das zu deinen persönlichen Notizen hinzufügen: '{note_text}'?",
            f"¿Añadir esto a tus notas personales: '{note_text}'?",
            lang
        )

    preview = "\n".join(f"  • {item}" for item in items)
    pending_reminder.text = None
    pending_reminder.items = items
    pending_reminder.action = "personal_note_batch"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = None
    pending_reminder.event_data = None
    pending_reminder.lang = lang
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

    draft_prompt = f"""Today is {current_date}.
{previous_context}
Write a {draft_type} {draft_purpose} for {USER_NAME}.
- Professional and well-structured
- Use [placeholder] for unknown details
- Concise

Write the {draft_type}:"""

    draft_content = llm_invoke(draft_prompt).strip()
    title = f"{draft_type}_{draft_purpose.replace(' ', '_')[:30]}"

    pending_reminder.text = draft_content
    pending_reminder.action = "draft"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = title
    pending_reminder.event_data = {"type": draft_type, "purpose": draft_purpose}
    pending_reminder.lang = lang
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

    pending_reminder.text = title
    pending_reminder.action = "create_list"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = None
    pending_reminder.event_data = {"title": title, "items": items}
    pending_reminder.lang = lang
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
        pending_reminder.text = item
        pending_reminder.items = None
        pending_reminder.action = "add_to_list"
        pending_reminder.conflict = None
        pending_reminder.line_to_delete = filename
        pending_reminder.event_data = {"item": item, "filename": filename}
        pending_reminder.lang = lang
        return _t(
            f"Add '{item}' to '{filename}'?",
            f"'{item}' zur Liste '{filename}' hinzufügen?",
            f"¿Añadir '{item}' a '{filename}'?",
            lang
        )

    preview = "\n".join(f"  • {i}" for i in items)
    pending_reminder.text = None
    pending_reminder.items = items
    pending_reminder.action = "add_to_list_batch"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = filename
    pending_reminder.event_data = {"items": items, "filename": filename}
    pending_reminder.lang = lang
    return _t(
        f"Add these {len(items)} items to '{filename}'?\n{preview}",
        f"Diese {len(items)} Einträge zu '{filename}' hinzufügen?\n{preview}",
        f"¿Añadir estos {len(items)} elementos a '{filename}'?\n{preview}",
        lang
    )


def handle_remove_list_item(question: str, extracted: dict, lang: str = "en") -> str:
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

    pending_reminder.text = matched['text']
    pending_reminder.action = "remove_list_item"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = filename
    pending_reminder.event_data = {"filename": filename, "line_index": matched['index']}
    pending_reminder.lang = lang
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

    pending_reminder.text = matched
    pending_reminder.action = "delete_list"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = matched
    pending_reminder.event_data = None
    pending_reminder.lang = lang
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
    pending_reminder.text = f"{validated.title} on {validated.start}"
    pending_reminder.action = "calendar"
    pending_reminder.conflict = None
    pending_reminder.line_to_delete = None
    pending_reminder.event_data = event_dict
    pending_reminder.lang = lang

    confirmation = _t(
        f"Lock in '{validated.title}' on {validated.start}?",
        f"'{validated.title}' am {validated.start} in den Kalender eintragen?",
        f"¿Agendar '{validated.title}' el {validated.start}?",
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

    format_prompt = f"""{_persona_prompt()}

The user asked: "{question}"

Their reminders:
{reminders}

Present these in a friendly, natural way — warm but concise.
Respond in the same language the user used."""

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

    format_prompt = f"""{_persona_prompt()}

The user asked: "{question}"

{events_context}

List the relevant upcoming events in a friendly, natural way.
Respond in the same language the user used.
Keep it concise — 1 to 4 events max, summarise the rest if there are more."""

    return llm_invoke(format_prompt).strip()


def handle_weather(question: str, extracted: dict, lang: str = "en") -> str:
    """Fetch weather and return a natural Chatette-style response."""
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

    weather_prompt = f"""{_persona_prompt()}

The user asked: "{question}"

Here is the current weather data:
{context}

Respond naturally in Chatette's voice — warm, concise, useful.
Highlight the most important things (temperature, rain, wind if notable). Don't be too technical.
Respond in the same language the user used.
Keep it to 2-3 sentences maximum."""

    return llm_invoke(weather_prompt).strip()


def handle_about_chatette(question: str, lang: str = "en") -> str:
    """Answer questions about Chatette."""
    about_prompt = (
        f"{_persona_prompt()}\n\n"
        f"{_build_conversation_context()}"
        f"Answer naturally in 1-2 sentences.\n"
        f"Always respond in the same language the user used.\n\n"
        f"Question: {question}\n"
        f"Chatette:"
    )
    return llm_invoke(about_prompt).strip()


def handle_confirmation(question: str, lang: str = "en") -> str:
    """Handle yes/no confirmation for all pending actions."""
    action = pending_reminder.action
    reminder_text = pending_reminder.text
    line_to_delete = pending_reminder.line_to_delete
    conflict = pending_reminder.conflict
    event_data = pending_reminder.event_data
    lang = pending_reminder.lang

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
            save_reminder(reminder_text, due=pending_reminder.due)
            ingest_notes()
            pending_reminder.clear()
            return _t(
                f"On it — '{reminder_text}' is on your list.",
                f"Erledigt — '{reminder_text}' ist auf deiner Liste.",
                f"Listo — '{reminder_text}' está en tu lista.",
                lang
            )

        elif action == "save_batch":
            items = pending_reminder.items or []
            for item in items:
                if isinstance(item, dict):
                    save_reminder(item["text"], due=item.get("due"))
                else:
                    save_reminder(item)
            ingest_notes()
            pending_reminder.clear()
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
            pending_reminder.clear()
            return result

        elif action == "personal_note":
            save_personal_note(reminder_text)
            ingest_notes()
            pending_reminder.clear()
            return _t(
                "Done and dusted — added to your personal notes.",
                "Erledigt — zu deinen persönlichen Notizen hinzugefügt.",
                "Listo — añadido a tus notas personales.",
                lang
            )

        elif action == "personal_note_batch":
            items = pending_reminder.items or []
            for item in items:
                save_personal_note(item)
            ingest_notes()
            pending_reminder.clear()
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
            pending_reminder.clear()
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
            pending_reminder.clear()
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
            pending_reminder.clear()
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
            pending_reminder.clear()
            count = len(items)
            return _t(
                f"Done — {count} item{'s' if count != 1 else ''} added to '{filename}'.",
                f"Erledigt — {count} Einträge zu '{filename}' hinzugefügt.",
                f"Listo — {count} elemento{'s' if count != 1 else ''} añadido{'s' if count != 1 else ''} a '{filename}'.",
                lang
            )

        elif action == "remove_list_item":
            filename = event_data["filename"]
            line_index = event_data["line_index"]
            delete_list_item(filename, line_index)
            ingest_lists()
            pending_reminder.clear()
            return _t(
                f"'{reminder_text}' is off the list.",
                f"'{reminder_text}' wurde von der Liste entfernt.",
                f"'{reminder_text}' eliminado de la lista.",
                lang
            )

        elif action == "delete_list":
            success = delete_list(line_to_delete)
            ingest_lists()
            pending_reminder.clear()
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
            pending_reminder.clear()
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
                pending_reminder.clear()
                return _t(
                    f"Done — '{event_data['title']}' has been removed from your calendar.",
                    f"Erledigt — '{event_data['title']}' wurde aus deinem Kalender entfernt.",
                    f"Listo — '{event_data['title']}' fue eliminado de tu calendario.",
                    lang
                )
            except Exception as e:
                pending_reminder.clear()
                print(f"Failed to delete event: {e}")
                return _t(
                    "Something went wrong deleting the event — try again?",
                    "Beim Löschen des Termins ist etwas schiefgelaufen — nochmal versuchen?",
                    "Algo salió mal al eliminar el evento — ¿intentamos de nuevo?",
                    lang
                )

    elif is_no:
        pending_reminder.clear()
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
            items = pending_reminder.items or []
            preview = ", ".join(items)
            return _t(
                f"Save these reminders: {preview}? Yes or no.",
                f"Diese Erinnerungen speichern: {preview}? Ja oder Nein.",
                f"¿Guardar estos recordatorios: {preview}? Sí o no.",
                lang
            )
        elif action == "personal_note_batch":
            items = pending_reminder.items or []
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
        elif action == "remove_list_item":
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
# MAIN ASK FUNCTION
# ===================================

RELEVANCE_THRESHOLD = 0.93


def ask(question: str, mode: str = "auto", lang: str = "en") -> str:
    print(f"\nYou: {question} [mode: {mode}] [lang: {lang}]")

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


def _ask_internal(question: str, mode: str = "auto", lang: str = "en") -> str:

    # 1. Pending confirmation — check before classification
    if pending_reminder.action is not None:
        answer = handle_confirmation(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 2. Classify intent
    classification = classify_intent(question, lang)
    intent = classification.intent
    confidence = classification.confidence
    extracted = classification.extracted

    # 3. Low confidence on action intents → ask for clarification
    if confidence == "low" and intent != "general":
        answer = _t(
            "I'm not quite sure what you'd like me to do — could you rephrase?",
            "Ich bin nicht ganz sicher, was du möchtest — könntest du es umformulieren?",
            "No estoy segura de lo que quieres — ¿puedes reformularlo?",
            lang
        )
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 4. Dispatch to handler
    if intent == "save_reminder":
        answer = handle_reminder(question, extracted, lang)
    elif intent == "delete_reminder":
        answer = handle_delete(question, extracted, lang)
    elif intent == "delete_event":
        answer = handle_delete_event(question, extracted, lang)
    elif intent == "view_reminders":
        answer = handle_view_reminders(question, lang)
    elif intent == "view_events":
        answer = handle_view_events(question, lang)
    elif intent == "get_weather":
        answer = handle_weather(question, extracted, lang)
    elif intent == "about_chatette":
        answer = handle_about_chatette(question, lang)
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
    elif intent == "remove_list_item":
        answer = handle_remove_list_item(question, extracted, lang)
    elif intent == "delete_list":
        answer = handle_delete_list(question, extracted, lang)
    else:
        answer = _handle_general(question, mode, lang)

    print(f"Chatette: {answer}")
    conversation_context["last_question"] = question
    conversation_context["last_answer"] = answer
    return answer


def _handle_general(question: str, mode: str, lang: str) -> str:
    """Handle general questions via RAG or pure LLM."""

    _general_prompt_base = (
        f"{_persona_prompt()}\n\n"
        f"{_build_conversation_context()}"
        f"Answer the following question in 1-2 sentences.\n"
        f"- Always respond in the same language the user used.\n"
        f"- If the question refers to the previous exchange, use it.\n\n"
        f"Question: {{question}}\n"
        f"Chatette:"
    )

    if mode == "general":
        print("Mode: general knowledge")
        return llm_invoke(_general_prompt_base.format(question=question)).strip()

    if mode == "personal":
        print("Mode: personal/RAG")
        return _rag_query(question, k_emails=7, k_other=5)

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

    return _rag_query(question, k_emails=7, k_other=5, docs=all_docs)


def _rag_query(question: str, k_emails: int = 7, k_other: int = 5,
               docs: list = None) -> str:
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
        user_name=USER_NAME, user_profile=USER_PROFILE
    )
    return llm_invoke(formatted_prompt).strip()


if __name__ == "__main__":
    ask("Do I have any appointments coming up?")
    ask("What's on my shopping list?")