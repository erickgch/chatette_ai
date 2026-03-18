import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from ingestion import ingest_notes, ingest_calendar_events, ingest_lists
from note_manager import (
    save_reminder, delete_reminder_by_line, get_all_reminders,
    save_personal_note, save_draft,
    create_list, add_item_to_list, find_list_by_name,
    get_all_lists, delete_list
)
from datetime import datetime, timedelta
from google_integration import create_calendar_event

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
    response = llm.invoke(prompt)
    if hasattr(response, 'content'):
        return response.content
    return str(response)


# ===================================
# PERSONALITY
# ===================================

CHATETTE_PERSONA = """You are Chatette — a small robotic cat and personal assistant to {user_name}.
About {user_name}: {user_profile}

Your personality:
- Warm, friendly and competent — like a smart friend who always has things under control.
- You have your own voice. Not a generic AI bot. Not a corporate assistant.
- Direct and accurate. You answer first, add warmth second.
- Rarely — maybe once every 5-6 responses — use a cool casual phrase naturally.
  When you do, pick from: "On it.", "Consider it done.", "Locked in.", "Done and dusted.",
  "Nothing on the radar.", "All clear.", "Heads up —", "Quick one:",
  "You're all set.", "Just keeping you in the loop."
  Most responses should NOT start with these phrases.
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
    "event_data": None
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

DELETE_TRIGGERS = [
    "delete the reminder", "remove the reminder",
    "cancel the reminder", "delete the note",
    "remove the note", "forget about",
    "remove the event", "delete the event"
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

def is_reminder_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in REMINDER_TRIGGERS)

def is_delete_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in DELETE_TRIGGERS)

def is_calendar_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in CALENDAR_TRIGGERS)

def is_personal_note_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in PERSONAL_NOTE_TRIGGERS)

def is_draft_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in DRAFT_TRIGGERS)

def is_list_create_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in LIST_CREATE_TRIGGERS)

def is_list_add_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in LIST_ADD_TRIGGERS)

def is_list_delete_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in LIST_DELETE_TRIGGERS)

def is_about_chatette(question: str) -> bool:
    return any(trigger in question.lower() for trigger in CHATETTE_TRIGGERS)

def is_reminders_view_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in REMINDERS_VIEW_TRIGGERS)


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

def handle_reminder(question: str) -> str:
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
                "conflict": "duplicate", "line_to_delete": existing_line
            })
            return (f"Heads up — you already have something similar: '{existing_line}'. "
                    f"Want to replace it with '{reminder_text}'?")

        elif conflict_check.startswith("CONFLICT:"):
            existing_line = conflict_check.replace("CONFLICT:", "").strip()
            pending_reminder.update({
                "text": reminder_text, "action": "save",
                "conflict": "conflict", "line_to_delete": existing_line
            })
            return (f"Quick one — you already have '{existing_line}' around that time. "
                    f"Still want to save '{reminder_text}'?")

    pending_reminder.update({
        "text": reminder_text, "action": "save",
        "conflict": None, "line_to_delete": None
    })
    return f"Just to confirm — save this to reminders: '{reminder_text}'?"


def handle_delete(question: str) -> str:
    """Use LLM to find matching reminder and ask confirmation."""
    all_reminders = get_all_reminders()
    if all_reminders == "No reminders found.":
        return "Nothing on your reminder list."

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
        return "Nothing matching that in your reminders. Can you be more specific?"

    pending_reminder.update({
        "text": None, "action": "delete",
        "conflict": None, "line_to_delete": matched_line
    })
    return f"Delete this one: '{matched_line}'?"


def handle_personal_note(question: str) -> str:
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
        "conflict": None, "line_to_delete": None, "event_data": None
    })
    return f"Add this to your personal notes: '{note_text}'?"


def handle_draft(question: str) -> str:
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
        "event_data": {"type": draft_type, "purpose": draft_purpose}
    })
    return f"Here's a draft {draft_type} {draft_purpose}:\n\n{draft_content}\n\nWant me to save this?"


def handle_create_list(question: str) -> str:
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
        "event_data": {"title": title, "items": items}
    })
    return f"Create a list called '{title}' — {items_preview}?"


def handle_add_to_list(question: str) -> str:
    """Extract item and target list, confirm adding."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return "No lists yet — want to create one first?"

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
        return "Didn't quite catch that — can you be more specific?"

    item = data.get("item", "")
    list_name = data.get("list_name", "")

    if not item:
        return "What would you like to add to the list?"

    filename = find_list_by_name(list_name)
    if not filename:
        return f"No list matching '{list_name}' on file."

    pending_reminder.update({
        "text": item, "action": "add_to_list",
        "conflict": None, "line_to_delete": filename,
        "event_data": {"item": item, "filename": filename}
    })
    return f"Add '{item}' to '{filename}'?"


def handle_delete_list(question: str) -> str:
    """Find and confirm deletion of a list."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return "No lists to delete — all clear."

    lists_summary = "\n".join([f"- {l['filename']}" for l in all_lists])

    match_prompt = f"""The user wants to delete a list: "{question}"
{previous_context}
Available lists:
{lists_summary}

Return ONLY the exact filename to delete.
If nothing matches, return NO_MATCH."""

    matched = llm_invoke(match_prompt).strip().strip('"').strip("'")

    if "NO_MATCH" in matched or not matched:
        return "Nothing matching that in your lists — can you be more specific?"

    pending_reminder.update({
        "text": matched, "action": "delete_list",
        "conflict": None, "line_to_delete": matched,
        "event_data": None
    })
    return f"About to delete '{matched}' — this can't be undone. You sure?"


def handle_calendar_event(question: str) -> str:
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
        return "Couldn't parse the event details — try again?"

    print(f"📅 Extracted event: {event_data}")

    pending_reminder.update({
        "text": f"{event_data['title']} on {event_data['start']}",
        "action": "calendar", "conflict": None,
        "line_to_delete": None, "event_data": event_data
    })

    confirmation = f"Lock in '{event_data['title']}' on {event_data['start']}?"
    if event_data.get("attendees"):
        confirmation += f" I'll invite {', '.join(event_data['attendees'])} too."
    return confirmation


def handle_view_reminders(question: str) -> str:
    """Return all current reminders directly from file."""
    reminders = get_all_reminders()
    if reminders == "No reminders found.":
        return "All clear — nothing on your reminder list."

    format_prompt = f"""{_persona_prompt()}

The user asked: "{question}"

Their reminders:
{reminders}

Present these in a friendly, natural way — warm but concise.
Respond in the same language the user used."""

    return llm_invoke(format_prompt).strip()


def handle_confirmation(question: str) -> str:
    """Handle yes/no confirmation for all pending actions."""
    action = pending_reminder["action"]
    reminder_text = pending_reminder["text"]
    line_to_delete = pending_reminder["line_to_delete"]
    conflict = pending_reminder["conflict"]
    event_data = pending_reminder["event_data"]

    is_yes = any(word in question.lower() for word in
                 ["yes", "correct", "right", "yep", "yeah", "sure", "ok", "okay"])
    is_no = any(word in question.lower() for word in
                ["no", "wrong", "incorrect", "nope", "cancel"])

    def _clear_pending():
        pending_reminder.update({
            "text": None, "action": None,
            "conflict": None, "line_to_delete": None, "event_data": None
        })

    if is_yes:
        if action == "save":
            if line_to_delete and conflict in ["duplicate", "conflict"]:
                delete_reminder_by_line(line_to_delete)
            save_reminder(reminder_text)
            ingest_notes()
            _clear_pending()
            return f"On it — '{reminder_text}' is on your list."

        elif action == "delete":
            result = delete_reminder_by_line(line_to_delete)
            ingest_notes()
            _clear_pending()
            return result

        elif action == "personal_note":
            save_personal_note(reminder_text)
            ingest_notes()
            _clear_pending()
            return "Done and dusted — added to your personal notes."

        elif action == "draft":
            filename = save_draft(line_to_delete, reminder_text)
            ingest_notes()
            _clear_pending()
            return f"Saved as '{filename}' in your drafts. You're all set."

        elif action == "create_list":
            title = event_data["title"]
            items = event_data["items"]
            filename = create_list(title, items)
            ingest_lists()
            _clear_pending()
            return f"List '{title}' is ready — find it in your documents."

        elif action == "add_to_list":
            item = event_data["item"]
            filename = event_data["filename"]
            add_item_to_list(filename, item)
            ingest_lists()
            _clear_pending()
            return f"'{item}' is on the list."

        elif action == "delete_list":
            success = delete_list(line_to_delete)
            ingest_lists()
            _clear_pending()
            return f"List '{line_to_delete}' deleted." if success else "Something went wrong — try again?"

        elif action == "calendar":
            if event_data is None:
                return "Lost the event details somewhere — try again?"
            create_calendar_event(
                title=event_data["title"],
                start_datetime=event_data["start"],
                end_datetime=event_data.get("end"),
                description=event_data.get("description", ""),
                attendees=event_data.get("attendees", [])
            )
            ingest_calendar_events()
            _clear_pending()
            result = f"Locked in — '{event_data['title']}' is on your calendar."
            if event_data.get("attendees"):
                result += f" Invitations sent to {', '.join(event_data['attendees'])}."
            return result

    elif is_no:
        _clear_pending()
        return "No problem — consider it dropped."

    else:
        if action == "save":
            return f"Save '{reminder_text}' to reminders? Yes or no."
        elif action == "delete":
            return f"Delete '{line_to_delete}'? Yes or no."
        elif action == "personal_note":
            return "Add this to your personal notes? Yes or no."
        elif action == "draft":
            return "Save this draft? Yes or no."
        elif action == "create_list":
            return f"Create list '{event_data['title']}'? Yes or no."
        elif action == "add_to_list":
            return f"Add '{event_data['item']}' to the list? Yes or no."
        elif action == "delete_list":
            return f"Delete '{line_to_delete}'? Yes or no."
        elif action == "calendar":
            return f"Lock in '{event_data['title']}' on your calendar? Yes or no."


# ===================================
# MAIN ASK FUNCTION
# ===================================

RELEVANCE_THRESHOLD = 0.93


def ask(question: str, mode: str = "auto") -> str:
    print(f"\nYou: {question} [mode: {mode}]")

    # 1. Pending confirmation
    if pending_reminder["action"] is not None:
        answer = handle_confirmation(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 2. Delete reminder
    if is_delete_request(question):
        answer = handle_delete(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 3. View reminders
    if is_reminders_view_request(question):
        answer = handle_view_reminders(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 4. About Chatette
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

    # 5. Save reminder
    if is_reminder_request(question):
        answer = handle_reminder(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 6. Calendar event
    if is_calendar_request(question):
        answer = handle_calendar_event(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 7. Personal note
    if is_personal_note_request(question):
        answer = handle_personal_note(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 8. Draft
    if is_draft_request(question):
        answer = handle_draft(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 9. Create list
    if is_list_create_request(question):
        answer = handle_create_list(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 10. Add to list
    if is_list_add_request(question):
        answer = handle_add_to_list(question)
        print(f"Chatette: {answer}")
        conversation_context["last_question"] = question
        conversation_context["last_answer"] = answer
        return answer

    # 11. Delete list
    if is_list_delete_request(question):
        answer = handle_delete_list(question)
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