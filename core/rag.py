import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from ingestion import ingest_notes, ingest_calendar_events, ingest_lists
from note_manager import (
    create_reminder, delete_reminder_by_line, get_all_reminders,
    save_personal_note, save_draft,
    create_list, add_item_to_list, find_list_by_name,
    get_all_lists, delete_list, get_list_items, delete_list_item
)
from datetime import datetime, timedelta
from google_integration import create_calendar_event, delete_calendar_event, get_upcoming_events

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
USER_NAME = os.getenv("USER_NAME")
USER_PROFILE = os.getenv("USER_PROFILE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
    import re
    try:
        # For Qwen3 models, disable chain-of-thought thinking
        # (/no_think appended to prompt suppresses <think> blocks)
        _prompt = prompt
        if USE_GROQ and "qwen" in GROQ_MODEL.lower():
            _prompt = prompt + " /no_think"

        response = llm.invoke(_prompt)
        text = response.content if hasattr(response, 'content') else str(response)

        # Strip any <think>...</think> blocks just in case
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
  Examples: "I've had my eye on that.", "That one slipped through my paws."
- Never over-explain or pad responses with unnecessary pleasantries.
- Never apologize unnecessarily.
- Always respond in the same language the user used."""

CHATETTE_RULES = """
STRICT RULES:
- Only answer what was specifically asked. Nothing more.
- Keep answers to 1-3 sentences maximum.
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

# ===== Pending state =====
pending_reminder = {
    "text": None,
    "action": None,
    "conflict": None,
    "line_to_delete": None,
    "event_data": None,
    "lang": "en"
}

# ===================================
# TRIGGERS
# ===================================

REMINDER_TRIGGERS = [
    "remind me", "remember that", "don't forget",
    "make a note", "add a note", "save this",
    "note that", "write down", "write that down",
    "save that", "note this", "note down"
]

DELETE_REMINDER_TRIGGERS = [
    "delete the reminder", "remove the reminder",
    "cancel the reminder", "delete the note",
    "remove the note", "forget about",
]

DELETE_EVENT_TRIGGERS = [
    "remove the event", "delete the event",
    "cancel the event", "remove from my calendar",
    "delete from my calendar", "cancel my appointment",
    "cancel my meeting",
    "please remove from my calendar", "please delete from my calendar",
]

CALENDAR_TRIGGERS = [
    "add to my calendar", "add to calendar",
    "add it to my calendar", "add this to my calendar",
    "add that to my calendar", "schedule a", "schedule an",
    "create an event", "put it in my calendar",
    "put that in my calendar", "put in my calendar",
    "save that in the calendar", "book a",
    "can you put", "can you schedule"
]

PERSONAL_NOTE_TRIGGERS = [
    "add to my personal notes", "add to my diary",
    "add to my journal", "save to my personal notes",
    "write in my diary", "write in my journal",
    "note in my diary", "note in my journal"
]

DRAFT_TRIGGERS = [
    "help me write", "write a draft", "draft a", "draft an",
    "write an email", "write a letter", "write a message",
    "compose an email", "compose a"
]

LIST_CREATE_TRIGGERS = [
    "create a list", "make a list", "new list",
    "start a list", "create a shopping list",
    "make a shopping list", "create a to-do list",
    "make a to-do list", "create a checklist"
]

LIST_ADD_TRIGGERS = [
    "add to my list", "add to the list",
    "add to my shopping list", "add to the shopping list",
    "add this to my list", "add that to my list",
    "put on my list", "add to my to-do list"
]

LIST_REMOVE_ITEM_TRIGGERS = [
    "remove from my list", "remove from the list",
    "delete from my list", "delete from the list",
    "take off my list", "remove from my shopping list",
    "delete from my shopping list", "cross off",
    "take off the list", "remove this from"
]

LIST_DELETE_TRIGGERS = [
    "delete the list", "remove the list",
    "delete my list", "remove my list"
]

CHATETTE_TRIGGERS = [
    "what are you", "who are you",
    "what's your name", "what is your name",
    "tell me about you"
]

REMINDERS_VIEW_TRIGGERS = [
    "check my reminders", "show my reminders",
    "what are my reminders", "list my reminders",
    "any reminders", "my reminders",
    "do i have any reminders", "what reminders"
]


# ===================================
# INTENT DETECTION
# ===================================

def _starts_with_any(question: str, triggers: list) -> bool:
    """Check if question starts with any trigger phrase."""
    q = question.lower().strip()
    return any(q.startswith(trigger) for trigger in triggers)

def _contains_any(question: str, triggers: list) -> bool:
    """Check if question contains any trigger phrase anywhere."""
    q = question.lower()
    return any(trigger in q for trigger in triggers)


# Action triggers — startswith only (avoids false positives)
def is_reminder_request(question: str) -> bool:
    return _starts_with_any(question, REMINDER_TRIGGERS)

def is_delete_reminder_request(question: str) -> bool:
    return _starts_with_any(question, DELETE_REMINDER_TRIGGERS)

def is_delete_event_request(question: str) -> bool:
    return _starts_with_any(question, DELETE_EVENT_TRIGGERS)

def is_calendar_request(question: str) -> bool:
    return _starts_with_any(question, CALENDAR_TRIGGERS)

def is_personal_note_request(question: str) -> bool:
    return _starts_with_any(question, PERSONAL_NOTE_TRIGGERS)

def is_draft_request(question: str) -> bool:
    return _starts_with_any(question, DRAFT_TRIGGERS)

def is_list_create_request(question: str) -> bool:
    return _starts_with_any(question, LIST_CREATE_TRIGGERS)

def is_list_add_request(question: str) -> bool:
    return _starts_with_any(question, LIST_ADD_TRIGGERS)

def is_list_remove_item_request(question: str) -> bool:
    return _starts_with_any(question, LIST_REMOVE_ITEM_TRIGGERS)

def is_list_delete_request(question: str) -> bool:
    return _starts_with_any(question, LIST_DELETE_TRIGGERS)


# Query triggers — contains (questions, not commands)
def is_about_chatette(question: str) -> bool:
    return _contains_any(question, CHATETTE_TRIGGERS)

def is_reminders_view_request(question: str) -> bool:
    return _contains_any(question, REMINDERS_VIEW_TRIGGERS)


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

def handle_reminder(question: str, lang: str = "en") -> str:
    """Extract reminder using conversation context, check conflicts, confirm."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    all_reminders = get_all_reminders()
    previous_context = _build_conversation_context()

    extraction_prompt = f"""Today is {current_date}.
{previous_context}
The user wants to save this reminder: "{question}"

Extract ONLY the reminder text. Nothing else.
- Remove trigger phrases like "write down", "remind me", "make a note", "note that"
- If the user refers to something from the previous exchange, resolve it
- If a date is mentioned, convert it to an actual calendar date
- If NO date is mentioned, return the reminder exactly as stated
- Return ONLY the reminder text — no explanations

Examples:
Input: "write down buy green pesto"
Output: Buy green pesto

Input: "remind me dentist next Friday"
Output: Dentist appointment on Friday, March 20, 2026

Previous exchange: User asked if they have toilet paper. Chatette said no info.
Input: "write that down please"
Output: Buy toilet paper

Now extract from: "{question}"
Output:"""

    reminder_text = llm_invoke(extraction_prompt).strip()
    reminder_text = reminder_text.split("\n")[0].strip().strip('"').strip("'")

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
            pending_reminder.update({
                "text": reminder_text, "action": "save",
                "conflict": "duplicate", "line_to_delete": existing_line,
                "lang": lang
            })
            lang = pending_reminder["lang"]
            return (
                _t(
                    f"Heads up — you already have something similar: '{existing_line}'. Want to replace it with '{reminder_text}'?",
                    f"Hinweis — du hast bereits etwas Ähnliches: '{existing_line}'. Ersetzen mit '{reminder_text}'?",
                    f"Aviso — ya tienes algo similar: '{existing_line}'. ¿Reemplazar con '{reminder_text}'?",
                    lang
                )
            )

        elif conflict_check.startswith("CONFLICT:"):
            existing_line = conflict_check.replace("CONFLICT:", "").strip()
            pending_reminder.update({
                "text": reminder_text, "action": "save",
                "conflict": "conflict", "line_to_delete": existing_line,
                "lang": lang
            })
            lang = pending_reminder["lang"]
            return (
                _t(
                    f"Quick one — you already have '{existing_line}' around that time. Still want to save '{reminder_text}'?",
                    f"Kurze Frage — du hast bereits '{existing_line}' zu dieser Zeit. Trotzdem '{reminder_text}' speichern?",
                    f"Un momento — ya tienes '{existing_line}' a esa hora. ¿Guardar '{reminder_text}' igualmente?",
                    lang
                )
            )

    pending_reminder.update({
        "text": reminder_text, "action": "save",
        "conflict": None, "line_to_delete": None, "lang": lang
    })
    return _t(
        f"Just to confirm — save this to reminders: '{reminder_text}'?",
        f"Zur Bestätigung — in die Erinnerungen speichern: '{reminder_text}'?",
        f"Para confirmar — ¿guardar esto en recordatorios: '{reminder_text}'?",
        lang
    )


def handle_delete(question: str, lang: str = "en") -> str:
    """Use LLM to find matching reminder and ask confirmation."""
    all_reminders = get_all_reminders()
    if all_reminders == "No reminders found.":
        return _t(
            "Nothing on your reminder list.",
            "Keine Erinnerungen vorhanden.",
            "No tienes recordatorios.",
            lang
        )

    previous_context = _build_conversation_context()

    match_prompt = f"""Find the reminder to delete.
{previous_context}
The user wants to delete: "{question}"

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

    pending_reminder.update({
        "text": None, "action": "delete",
        "conflict": None, "line_to_delete": matched_line, "lang": lang
    })
    return _t(
        f"Delete this one: '{matched_line}'?",
        f"Diese Erinnerung löschen: '{matched_line}'?",
        f"¿Eliminar este recordatorio: '{matched_line}'?",
        lang
    )


def handle_delete_event(question: str, lang: str = "en") -> str:
    """Use LLM to find matching calendar event and ask confirmation."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    # Fetch upcoming events from Google Calendar
    try:
        events = get_upcoming_events(days_ahead=60, days_behind=1)
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

    # Build a summary of events for the LLM to match against
    events_summary = "\n".join([
        f"- ID: {e['id']} | {e['title']} on {e['start']}"
        for e in events
        if 'id' in e
    ])

    match_prompt = f"""Today is {current_date}.
{previous_context}
The user wants to delete a calendar event: "{question}"

Upcoming events:
{events_summary}

Instructions:
- Find the event that best matches what the user wants to delete
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

    pending_reminder.update({
        "text": event_title,
        "action": "delete_event",
        "conflict": None,
        "line_to_delete": event_id,
        "event_data": {"id": event_id, "title": event_title},
        "lang": lang
    })
    return _t(
        f"Delete '{event_title}' from your Google Calendar?",
        f"'{event_title}' aus deinem Google Kalender löschen?",
        f"¿Eliminar '{event_title}' de tu Google Calendar?",
        lang
    )


def handle_personal_note(question: str, lang: str = "en") -> str:
    """Extract and save a personal note using conversation context."""
    previous_context = _build_conversation_context()

    extraction_prompt = f"""The user said: "{question}"
{previous_context}
Extract ONLY the note content to save.
- Remove trigger phrases like "add to my personal notes", "add to my diary"
- If the user refers to something from the previous exchange, resolve it
- Return ONLY the note text

Output:"""

    note_text = llm_invoke(extraction_prompt).strip()
    note_text = note_text.split("\n")[0].strip().strip('"').strip("'")

    pending_reminder.update({
        "text": note_text, "action": "personal_note",
        "conflict": None, "line_to_delete": None, "event_data": None,
        "lang": lang
    })
    lang = pending_reminder["lang"]
    return _t(
        f"Add this to your personal notes: '{note_text}'?",
        f"Das zu deinen persönlichen Notizen hinzufügen: '{note_text}'?",
        f"¿Añadir esto a tus notas personales: '{note_text}'?",
        lang
    )


def handle_draft(question: str, lang: str = "en") -> str:
    """Generate a draft document using conversation context."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    import json
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
    except json.JSONDecodeError:
        draft_info = {"type": "document", "purpose": question}

    draft_type = draft_info.get("type", "document")
    draft_purpose = draft_info.get("purpose", question)

    draft_prompt = f"""Today is {current_date}.
{previous_context}
Write a {draft_type} {draft_purpose} for {USER_NAME}.
- Professional and well-structured
- Use [placeholder] for unknown details
- Concise

Write the {draft_type}:"""

    draft_content = llm_invoke(draft_prompt).strip()
    title = f"{draft_type}_{draft_purpose.replace(' ', '_')[:30]}"

    pending_reminder.update({
        "text": draft_content, "action": "draft",
        "conflict": None, "line_to_delete": title,
        "event_data": {"type": draft_type, "purpose": draft_purpose},
        "lang": lang
    })
    lang = pending_reminder["lang"]
    return f"Here's a draft {draft_type} {draft_purpose}:\n\n{draft_content}\n\n" + _t(
        "Want me to save this?",
        "Soll ich das speichern?",
        "¿Quieres que lo guarde?",
        lang
    )


def handle_create_list(question: str, lang: str = "en") -> str:
    """Extract list title and items, confirm creation."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    import json
    extraction_prompt = f"""Today is {current_date}.
{previous_context}
The user said: "{question}"

Extract list title and items.
Return ONLY a JSON object:
- title: list name
- items: array of items (empty if none mentioned)

Examples:
Input: "create a shopping list with milk, eggs and bread"
Output: {{"title": "Shopping List", "items": ["Milk", "Eggs", "Bread"]}}

Input: "make a packing list for my weekend trip"
Output: {{"title": "Packing List Weekend Trip", "items": []}}

Output:"""

    response = llm_invoke(extraction_prompt).strip()
    response = response[response.find("{"):response.rfind("}")+1]

    try:
        list_data = json.loads(response)
    except json.JSONDecodeError:
        list_data = {"title": "New List", "items": []}

    title = list_data.get("title", "New List")
    items = list_data.get("items", [])
    items_preview = ", ".join(items) if items else "empty for now"

    pending_reminder.update({
        "text": title, "action": "create_list",
        "conflict": None, "line_to_delete": None,
        "event_data": {"title": title, "items": items},
        "lang": lang
    })
    return _t(
        f"Create a list called '{title}' — {items_preview}?",
        f"Eine Liste namens '{title}' erstellen — {items_preview}?",
        f"¿Crear una lista llamada '{title}' — {items_preview}?",
        lang
    )


def handle_add_to_list(question: str, lang: str = "en") -> str:
    """Extract item and target list, confirm adding."""
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

    import json
    extraction_prompt = f"""The user said: "{question}"
{previous_context}
Available lists:
{lists_summary}

Extract item and target list.
Return ONLY a JSON object:
- item: the item to add
- list_name: target list name or filename

Output:"""

    response = llm_invoke(extraction_prompt).strip()
    response = response[response.find("{"):response.rfind("}")+1]

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return _t(
            "Didn't quite catch that — can you be more specific?",
            "Das habe ich nicht ganz verstanden — kannst du genauer sein?",
            "No entendí bien — ¿puedes ser más específico?",
            lang
        )

    item = data.get("item", "")
    list_name = data.get("list_name", "")

    if not item:
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

    pending_reminder.update({
        "text": item, "action": "add_to_list",
        "conflict": None, "line_to_delete": filename,
        "event_data": {"item": item, "filename": filename},
        "lang": lang
    })
    return _t(
        f"Add '{item}' to '{filename}'?",
        f"'{item}' zur Liste '{filename}' hinzufügen?",
        f"¿Añadir '{item}' a '{filename}'?",
        lang
    )


def handle_remove_from_list(question: str, lang: str = "en") -> str:
    """Find and remove a specific item from a list."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return _t("No lists on file.", "Keine Listen vorhanden.", "Sin listas en archivo.", lang)

    import json

    lists_with_items = []
    for l in all_lists:
        items = get_list_items(l['filename'])
        if items:
            item_names = [i['text'] for i in items]
            lists_with_items.append(
                f"{l['filename']}: {', '.join(item_names)}"
            )

    if not lists_with_items:
        return _t("Your lists are all empty.", "Deine Listen sind alle leer.", "Tus listas están vacías.", lang)

    lists_summary = "\n".join(lists_with_items)

    extraction_prompt = f"""The user said: "{question}"
{previous_context}
Lists and their items:
{lists_summary}

Extract what item to remove and from which list.
Return ONLY a JSON object:
- item: the item text to remove (match as closely as possible)
- filename: the exact list filename

Output:"""

    response = llm_invoke(extraction_prompt).strip()
    response = response[response.find("{"):response.rfind("}")+1]

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return _t(
            "Didn't catch that — which item and which list?",
            "Das habe ich nicht verstanden — welches Element und welche Liste?",
            "No entendí — ¿qué elemento y de qué lista?",
            lang
        )

    item = data.get("item", "")
    filename = data.get("filename", "")

    if not item or not filename:
        return _t(
            "Can you be more specific about what to remove and from which list?",
            "Kannst du genauer angeben, was entfernt werden soll und aus welcher Liste?",
            "¿Puedes ser más específico sobre qué eliminar y de qué lista?",
            lang
        )

    items = get_list_items(filename)
    matched = next(
        (i for i in items if item.lower() in i['text'].lower() or
         i['text'].lower() in item.lower()),
        None
    )

    if not matched:
        return _t(
            f"Couldn't find '{item}' in that list.",
            f"'{item}' wurde in der Liste nicht gefunden.",
            f"No encontré '{item}' en esa lista.",
            lang
        )

    pending_reminder.update({
        "text": matched['text'],
        "action": "remove_from_list",
        "conflict": None,
        "line_to_delete": filename,
        "event_data": {"filename": filename, "line_index": matched['index']},
        "lang": lang
    })
    return _t(
        f"Remove '{matched['text']}' from '{filename}'?",
        f"'{matched['text']}' aus '{filename}' entfernen?",
        f"¿Eliminar '{matched['text']}' de '{filename}'?",
        lang
    )


def handle_delete_list(question: str, lang: str = "en") -> str:
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

    match_prompt = f"""The user wants to delete a list: "{question}"
{previous_context}
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

    pending_reminder.update({
        "text": matched, "action": "delete_list",
        "conflict": None, "line_to_delete": matched,
        "event_data": None, "lang": lang
    })
    return _t(
        f"About to delete '{matched}' — this can't be undone. You sure?",
        f"'{matched}' wird gelöscht — das kann nicht rückgängig gemacht werden. Sicher?",
        f"Se eliminará '{matched}' — esto no se puede deshacer. ¿Seguro?",
        lang
    )


def handle_calendar_event(question: str, lang: str = "en") -> str:
    """Extract event details using conversation context."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    import json
    extraction_prompt = f"""Today is {current_date}.
{previous_context}The user said: "{question}"

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

    pending_reminder.update({
        "text": f"{event_data['title']} on {event_data['start']}",
        "action": "calendar", "conflict": None,
        "line_to_delete": None, "event_data": event_data,
        "lang": lang
    })

    confirmation = _t(
        f"Lock in '{event_data['title']}' on {event_data['start']}?",
        f"'{event_data['title']}' am {event_data['start']} in den Kalender eintragen?",
        f"¿Agendar '{event_data['title']}' el {event_data['start']}?",
        lang
    )
    if event_data.get("attendees"):
        confirmation += " " + _t(
            f"I'll invite {', '.join(event_data['attendees'])} too.",
            f"Ich lade auch {', '.join(event_data['attendees'])} ein.",
            f"También invitaré a {', '.join(event_data['attendees'])}.",
            lang
        )
    return confirmation


def handle_view_reminders(question: str, lang: str = "en") -> str:
    """Return all current reminders directly from file."""
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


def handle_confirmation(question: str, lang: str = "en") -> str:
    """Handle yes/no confirmation for all pending actions."""
    action = pending_reminder["action"]
    reminder_text = pending_reminder["text"]
    line_to_delete = pending_reminder["line_to_delete"]
    conflict = pending_reminder["conflict"]
    event_data = pending_reminder["event_data"]
    lang = pending_reminder.get("lang", lang)

    is_yes = any(word in question.lower() for word in
                 ["yes", "correct", "right", "yep", "yeah", "sure", "ok", "okay",
                  "ja", "sí", "si", "claro", "correcto", "genau", "klar", "gut"])
    is_no = any(word in question.lower() for word in
                ["no", "wrong", "incorrect", "nope", "cancel",
                 "nein", "falsch", "abbrechen"])

    def _clear_pending():
        pending_reminder.update({
            "text": None, "action": None,
            "conflict": None, "line_to_delete": None,
            "event_data": None, "lang": "en"
        })

    if is_yes:
        if action == "save":
            if line_to_delete and conflict in ["duplicate", "conflict"]:
                delete_reminder_by_line(line_to_delete)
            create_reminder(reminder_text)
            ingest_notes()
            _clear_pending()
            return _t(
                f"On it — '{reminder_text}' is on your list.",
                f"Erledigt — '{reminder_text}' ist auf deiner Liste.",
                f"Listo — '{reminder_text}' está en tu lista.",
                lang
            )

        elif action == "delete":
            result = delete_reminder_by_line(line_to_delete)
            ingest_notes()
            _clear_pending()
            return result

        elif action == "personal_note":
            save_personal_note(reminder_text)
            ingest_notes()
            _clear_pending()
            return _t(
                "Done and dusted — added to your personal notes.",
                "Erledigt — zu deinen persönlichen Notizen hinzugefügt.",
                "Listo — añadido a tus notas personales.",
                lang
            )

        elif action == "draft":
            filename = save_draft(line_to_delete, reminder_text)
            ingest_notes()
            _clear_pending()
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
            _clear_pending()
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
            _clear_pending()
            return _t(
                f"'{item}' is on the list.",
                f"'{item}' steht auf der Liste.",
                f"'{item}' está en la lista.",
                lang
            )

        elif action == "remove_from_list":
            filename = event_data["filename"]
            line_index = event_data["line_index"]
            delete_list_item(filename, line_index)
            ingest_lists()
            _clear_pending()
            return _t(
                f"'{reminder_text}' is off the list.",
                f"'{reminder_text}' wurde von der Liste entfernt.",
                f"'{reminder_text}' eliminado de la lista.",
                lang
            )

        elif action == "delete_list":
            success = delete_list(line_to_delete)
            ingest_lists()
            _clear_pending()
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
            _clear_pending()
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
                _clear_pending()
                return _t(
                    f"Done — '{event_data['title']}' has been removed from your calendar.",
                    f"Erledigt — '{event_data['title']}' wurde aus deinem Kalender entfernt.",
                    f"Listo — '{event_data['title']}' fue eliminado de tu calendario.",
                    lang
                )
            except Exception as e:
                _clear_pending()
                print(f"Failed to delete event: {e}")
                return _t(
                    "Something went wrong deleting the event — try again?",
                    "Beim Löschen des Termins ist etwas schiefgelaufen — nochmal versuchen?",
                    "Algo salió mal al eliminar el evento — ¿intentamos de nuevo?",
                    lang
                )

    elif is_no:
        _clear_pending()
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

    # 1. Pending confirmation
    if pending_reminder["action"] is not None:
        answer = handle_confirmation(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 2. Delete reminder
    if is_delete_reminder_request(question):
        answer = handle_delete(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 3. Delete calendar event
    if is_delete_event_request(question):
        answer = handle_delete_event(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 5. View reminders
    if is_reminders_view_request(question):
        answer = handle_view_reminders(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 6. About Chatette
    if is_about_chatette(question):
        about_prompt = (
            f"{_persona_prompt()}\n\n"
            f"{_build_conversation_context()}"
            f"Answer naturally in 1-2 sentences.\n"
            f"Always respond in the same language the user used.\n\n"
            f"Question: {question}\n"
            f"Chatette:"
        )
        answer = llm_invoke(about_prompt).strip()
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 7. Save reminder
    if is_reminder_request(question):
        answer = handle_reminder(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 8. Calendar event
    if is_calendar_request(question):
        answer = handle_calendar_event(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 9. Personal note
    if is_personal_note_request(question):
        answer = handle_personal_note(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 10. Draft
    if is_draft_request(question):
        answer = handle_draft(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 11. Create list
    if is_list_create_request(question):
        answer = handle_create_list(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 12. Add to list
    if is_list_add_request(question):
        answer = handle_add_to_list(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 13. Remove list item
    if is_list_remove_item_request(question):
        answer = handle_remove_from_list(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 14. Delete list
    if is_list_delete_request(question):
        answer = handle_delete_list(question, lang)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # ===== MODE HANDLING =====

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
        general_prompt = _general_prompt_base.format(question=question)
        answer = llm_invoke(general_prompt).strip()
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    if mode == "personal":
        print("Mode: personal/RAG")
        all_docs = []
        for collection in ["calendar", "emails", "notes", "lists", "documents"]:
            k = 7 if collection == "emails" else 5
            try:
                results = vectorstore.similarity_search(
                    question, k=k,
                    filter={"collection": collection}
                )
                all_docs.extend(results)
            except Exception as e:
                print(f"Could not retrieve from '{collection}': {e}")
                continue

        context = "\n\n".join([doc.page_content for doc in all_docs])
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
        answer = llm_invoke(formatted_prompt).strip()
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # AUTO mode
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
        general_prompt = _general_prompt_base.format(question=question)
        answer = llm_invoke(general_prompt).strip()
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    context = "\n\n".join([doc.page_content for doc in all_docs])
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

    answer = llm_invoke(formatted_prompt).strip()
    print(f"Chatette: {answer}")
    conversation_context["last_question"] = question
    conversation_context["last_answer"] = answer
    return answer


if __name__ == "__main__":
    ask("Do I have any appointments coming up?")
    ask("What's on my shopping list?")