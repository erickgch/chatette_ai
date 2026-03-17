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


# Custom prompt
prompt_template = PromptTemplate(
    template="""You are Chatette, a warm and friendly personal assistant.
You are talking to {user_name}. About {user_name}: {user_profile}

STRICT RULES:
- Only answer what was specifically asked. Nothing more.
- If the user asks about appointments, only mention appointments. Do not mention shopping or other topics.
- Never volunteer extra information that wasn't asked for.
- Keep answers to 1-3 sentences maximum.
- Never calculate days of the week from memory — always derive them from the current date provided in the context.
- Always respond in the same language of the QUESTION, not the language of the context.
- If the question refers to a previous exchange, use it to answer correctly.

If you don't find the answer in the context, say so briefly.

Example:
Context: Today is Monday, March 16, 2026. Calendar event: Dentist on Wednesday, March 18 at 10am.
{user_name}: Do I have anything this week?
Chatette: Yes! You've got a dentist appointment on Wednesday at 10am. Don't forget!

Example:
Context: User enjoys technology and building AI assistants.
{user_name}: Can you recommend a podcast?
Chatette: Since you're into tech and AI, you might enjoy Lex Fridman's podcast or the TWIML AI Podcast!

Example:
Context: Today is Sunday, March 15, 2026. Tomorrow is Monday, March 16, 2026.
Calendar event: Dentist on Wednesday, March 18 at 10am.
{user_name}: Do I have anything next Saturday?
Chatette: Yes! You've got gym on Saturday, March 21st at 9am. Have a good workout!

Example:
Context: Previous exchange: User: "what is the capital of Mexico?" Chatette: "Mexico City has been the capital since 1521."
{user_name}: since when?
Chatette: Mexico City has been the capital since 1521, when it was founded on the ruins of Tenochtitlan after the Spanish conquest.

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

Your job: extract ONLY the reminder text itself. Nothing else.
- Remove trigger phrases like "write down", "remind me", "make a note", "note that", "write that down"
- If the user refers to something from the previous exchange (e.g. "that", "it", "this"), resolve it using the previous exchange above
- If a date is mentioned, convert it to an actual calendar date
- If NO date is mentioned, return the reminder exactly as stated
- Return ONLY the reminder text — no explanations, no extra sentences

Examples:
Input: "write down buy green pesto"
Output: Buy green pesto

Input: "remind me dentist next Friday"
Output: Dentist appointment on Friday, March 20, 2026

Previous exchange: User asked if they have toilet paper. Chatette said no info available.
Input: "write that down please"
Output: Buy toilet paper

Now extract from: "{question}"
Output:"""

    reminder_text = llm_invoke(extraction_prompt).strip()
    reminder_text = reminder_text.split("\n")[0].strip().strip('"').strip("'")

    if all_reminders != "No reminders found.":
        conflict_prompt = f"""You are checking a list of reminders for duplicates or conflicts.

New reminder to save: "{reminder_text}"

Existing reminders:
{all_reminders}

Rules:
- Only flag CONFLICT or DUPLICATE for reminders that involve a specific date or time
- Shopping items, tasks, and general notes should always be CLEAR
- Only flag DUPLICATE if the exact same reminder already exists

Reply with one of:
- "DUPLICATE: <exact line>" if the exact same reminder already exists
- "CONFLICT: <exact line>" if it is a different event on the exact same date AND time
- "CLEAR" if there are no issues or if the reminder has no date
"""
        conflict_check = llm_invoke(conflict_prompt).strip()
        print(f"Conflict check: '{conflict_check}'")

        if conflict_check.startswith("DUPLICATE:"):
            existing_line = conflict_check.replace("DUPLICATE:", "").strip()
            pending_reminder.update({
                "text": reminder_text, "action": "save",
                "conflict": "duplicate", "line_to_delete": existing_line
            })
            return (f"You already have something similar: '{existing_line}'. "
                    f"Do you want to replace it with '{reminder_text}'?")

        elif conflict_check.startswith("CONFLICT:"):
            existing_line = conflict_check.replace("CONFLICT:", "").strip()
            pending_reminder.update({
                "text": reminder_text, "action": "save",
                "conflict": "conflict", "line_to_delete": existing_line
            })
            return (f"Heads up - you already have '{existing_line}' around that time. "
                    f"Do you still want to save '{reminder_text}'?")

    pending_reminder.update({
        "text": reminder_text, "action": "save",
        "conflict": None, "line_to_delete": None
    })
    return f"Just to confirm - should I save this to reminders: '{reminder_text}'?"


def handle_delete(question: str) -> str:
    """Use LLM to find matching reminder and ask confirmation."""
    all_reminders = get_all_reminders()
    if all_reminders == "No reminders found.":
        return "You don't have any saved reminders."

    previous_context = _build_conversation_context()

    match_prompt = f"""You are helping delete a reminder from a list.
{previous_context}
The user wants to delete: "{question}"

Here is the COMPLETE list of reminders:
{all_reminders}

Instructions:
- Find the one that best matches what the user wants to delete
- If the user says "that" or "it", use the previous exchange to identify what they mean
- Return ONLY the exact line including the timestamp
- If nothing matches, return NO_MATCH

Your answer:"""

    matched_line = llm_invoke(match_prompt).strip().strip('"').strip("'")
    print(f"LLM matched: '{matched_line}'")

    if "NO_MATCH" in matched_line or not matched_line:
        return "I couldn't find a matching reminder. Could you be more specific?"

    pending_reminder.update({
        "text": None, "action": "delete",
        "conflict": None, "line_to_delete": matched_line
    })
    return f"Should I delete this reminder: '{matched_line}'?"


def handle_personal_note(question: str) -> str:
    """Extract and save a personal note using conversation context."""
    previous_context = _build_conversation_context()

    extraction_prompt = f"""The user said: "{question}"
{previous_context}
Extract ONLY the note content they want to save.
- Remove trigger phrases like "add to my personal notes", "add to my diary", "add to my journal"
- If the user refers to something from the previous exchange, resolve it
- Return ONLY the note text, nothing else

Now extract from: "{question}"
Output:"""

    note_text = llm_invoke(extraction_prompt).strip()
    note_text = note_text.split("\n")[0].strip().strip('"').strip("'")

    pending_reminder.update({
        "text": note_text, "action": "personal_note",
        "conflict": None, "line_to_delete": None, "event_data": None
    })
    return f"Should I add this to your personal notes: '{note_text}'?"


def handle_draft(question: str) -> str:
    """Generate a draft document using conversation context."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    import json
    extraction_prompt = f"""Today is {current_date}.
{previous_context}
The user said: "{question}"

Extract the type and purpose of the draft.
Return ONLY a JSON object with:
- type: type of document (e.g. "email", "letter", "message")
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
You are helping {USER_NAME} write a {draft_type} {draft_purpose}.
Write a professional, well-structured {draft_type}.
Use [placeholder] for unknown info.
Keep it concise.

Write the {draft_type} now:"""

    draft_content = llm_invoke(draft_prompt).strip()
    title = f"{draft_type}_{draft_purpose.replace(' ', '_')[:30]}"

    pending_reminder.update({
        "text": draft_content, "action": "draft",
        "conflict": None, "line_to_delete": title,
        "event_data": {"type": draft_type, "purpose": draft_purpose}
    })
    return f"Here's a draft {draft_type} {draft_purpose}:\n\n{draft_content}\n\nShould I save this draft?"


def handle_create_list(question: str) -> str:
    """Extract list title and items, confirm creation."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    import json
    extraction_prompt = f"""Today is {current_date}.
{previous_context}
The user said: "{question}"

Extract the list title and items they want to create.
Return ONLY a JSON object with:
- title: name of the list (e.g. "Shopping List", "Packing List")
- items: array of items to add (empty array if none mentioned)

Examples:
Input: "create a shopping list with milk, eggs and bread"
Output: {{"title": "Shopping List", "items": ["Milk", "Eggs", "Bread"]}}

Input: "make a packing list for my weekend trip"
Output: {{"title": "Packing List Weekend Trip", "items": []}}

Input: "create a list"
Output: {{"title": "New List", "items": []}}

Output:"""

    response = llm_invoke(extraction_prompt).strip()
    response = response[response.find("{"):response.rfind("}")+1]

    try:
        list_data = json.loads(response)
    except json.JSONDecodeError:
        list_data = {"title": "New List", "items": []}

    title = list_data.get("title", "New List")
    items = list_data.get("items", [])
    items_preview = ", ".join(items) if items else "empty list"

    pending_reminder.update({
        "text": title, "action": "create_list",
        "conflict": None, "line_to_delete": None,
        "event_data": {"title": title, "items": items}
    })
    return f"Should I create a list called '{title}' with: {items_preview}?"


def handle_add_to_list(question: str) -> str:
    """Extract item and target list, confirm adding."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return "You don't have any lists yet. Would you like to create one?"

    lists_summary = "\n".join([f"- {l['filename']}" for l in all_lists])

    import json
    extraction_prompt = f"""The user said: "{question}"
{previous_context}
Available lists:
{lists_summary}

Extract what they want to add and to which list.
Return ONLY a JSON object with:
- item: the item to add
- list_name: the name or filename of the target list

Examples:
Input: "add toothpaste to my shopping list"
Output: {{"item": "Toothpaste", "list_name": "shopping"}}

Input: "add sunscreen to the packing list"
Output: {{"item": "Sunscreen", "list_name": "packing"}}

Output:"""

    response = llm_invoke(extraction_prompt).strip()
    response = response[response.find("{"):response.rfind("}")+1]

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return "I had trouble understanding. Could you be more specific?"

    item = data.get("item", "")
    list_name = data.get("list_name", "")

    if not item:
        return "I couldn't identify what you want to add. Could you be more specific?"

    filename = find_list_by_name(list_name)
    if not filename:
        return f"I couldn't find a list matching '{list_name}'. Check your lists in the documents folder."

    pending_reminder.update({
        "text": item, "action": "add_to_list",
        "conflict": None, "line_to_delete": filename,
        "event_data": {"item": item, "filename": filename}
    })
    return f"Should I add '{item}' to '{filename}'?"


def handle_delete_list(question: str) -> str:
    """Find and confirm deletion of a list."""
    previous_context = _build_conversation_context()
    all_lists = get_all_lists()

    if not all_lists:
        return "You don't have any lists to delete."

    lists_summary = "\n".join([f"- {l['filename']}" for l in all_lists])

    match_prompt = f"""The user wants to delete a list: "{question}"
{previous_context}
Available lists:
{lists_summary}

Return ONLY the exact filename of the list to delete.
If nothing matches, return NO_MATCH.

Your answer:"""

    matched = llm_invoke(match_prompt).strip().strip('"').strip("'")

    if "NO_MATCH" in matched or not matched:
        return "I couldn't find a matching list. Could you be more specific?"

    pending_reminder.update({
        "text": matched, "action": "delete_list",
        "conflict": None, "line_to_delete": matched,
        "event_data": None
    })
    return f"Should I delete the list '{matched}'? This cannot be undone."


def handle_calendar_event(question: str) -> str:
    """Extract event details using conversation context."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    previous_context = _build_conversation_context()

    import json
    extraction_prompt = f"""Today is {current_date}.
{previous_context}The user said: "{question}"

Extract the calendar event details and return ONLY a JSON object:
- title: event name
- start: ISO 8601 datetime (e.g. 2026-03-20T10:00:00)
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
        return "I had trouble understanding the event details. Could you be more specific?"

    print(f"📅 Extracted event: {event_data}")

    pending_reminder.update({
        "text": f"{event_data['title']} on {event_data['start']}",
        "action": "calendar", "conflict": None,
        "line_to_delete": None, "event_data": event_data
    })

    confirmation = f"Should I add '{event_data['title']}' to your Google Calendar on {event_data['start']}?"
    if event_data.get("attendees"):
        confirmation += f" I'll also invite: {', '.join(event_data['attendees'])}"
    return confirmation


def handle_view_reminders(question: str) -> str:
    """Return all current reminders directly from file."""
    reminders = get_all_reminders()
    if reminders == "No reminders found.":
        return "You don't have any reminders saved at the moment."

    format_prompt = f"""The user asked: "{question}"

Here are their current reminders:
{reminders}

Present these reminders naturally and concisely.
Always respond in the same language the user used."""

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
            return f"Got it! Added '{reminder_text}' to reminders!"

        elif action == "delete":
            result = delete_reminder_by_line(line_to_delete)
            ingest_notes()
            _clear_pending()
            return result

        elif action == "personal_note":
            save_personal_note(reminder_text)
            ingest_notes()
            _clear_pending()
            return "Got it! Added to your personal notes."

        elif action == "draft":
            filename = save_draft(line_to_delete, reminder_text)
            ingest_notes()
            _clear_pending()
            return f"Done! Draft saved as '{filename}' in your drafts folder."

        elif action == "create_list":
            title = event_data["title"]
            items = event_data["items"]
            filename = create_list(title, items)
            ingest_lists()
            _clear_pending()
            return f"Done! Created list '{title}' — find it in your documents folder."

        elif action == "add_to_list":
            item = event_data["item"]
            filename = event_data["filename"]
            add_item_to_list(filename, item)
            ingest_lists()
            _clear_pending()
            return f"Done! Added '{item}' to your list."

        elif action == "delete_list":
            success = delete_list(line_to_delete)
            ingest_lists()
            _clear_pending()
            return f"Done! Deleted list '{line_to_delete}'." if success else "Something went wrong deleting the list."

        elif action == "calendar":
            if event_data is None:
                return "Something went wrong — I lost the event details. Could you try again?"
            create_calendar_event(
                title=event_data["title"],
                start_datetime=event_data["start"],
                end_datetime=event_data.get("end"),
                description=event_data.get("description", ""),
                attendees=event_data.get("attendees", [])
            )
            ingest_calendar_events()
            _clear_pending()
            result = f"Done! I've added '{event_data['title']}' to your Google Calendar!"
            if event_data.get("attendees"):
                result += f" Invitations sent to: {', '.join(event_data['attendees'])}"
            return result

    elif is_no:
        _clear_pending()
        return "No problem, I've discarded it!"

    else:
        if action == "save":
            return f"Should I save '{reminder_text}' to reminders? Just say yes or no."
        elif action == "delete":
            return f"Should I delete '{line_to_delete}'? Just say yes or no."
        elif action == "personal_note":
            return "Should I add this to your personal notes? Just say yes or no."
        elif action == "draft":
            return "Should I save this draft? Just say yes or no."
        elif action == "create_list":
            return f"Should I create the list '{event_data['title']}'? Just say yes or no."
        elif action == "add_to_list":
            return f"Should I add '{event_data['item']}' to the list? Just say yes or no."
        elif action == "delete_list":
            return f"Should I delete the list '{line_to_delete}'? Just say yes or no."
        elif action == "calendar":
            return f"Should I add '{event_data['title']}' to your calendar? Just say yes or no."


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
        general_prompt = (
            f"You are Chatette, {USER_NAME}'s friendly personal assistant.\n"
            f"About {USER_NAME}: {USER_PROFILE}\n\n"
            f"{_build_conversation_context()}"
            f"Answer naturally in 1-2 sentences.\n"
            f"Always respond in the same language the user used.\n\n"
            f"Question: {question}\n"
            f"Chatette:"
        )
        answer = llm_invoke(general_prompt).strip()
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

    if mode == "general":
        print("Mode: general knowledge")
        general_prompt = (
            f"You are Chatette, a warm and knowledgeable personal assistant talking to {USER_NAME}.\n"
            f"About {USER_NAME}: {USER_PROFILE}\n\n"
            f"{_build_conversation_context()}"
            f"Answer the following question in 1-2 sentences.\n"
            f"- Always respond in the same language the user used in their QUESTION.\n"
            f"- If the question refers to the previous exchange, use it to answer.\n"
            f"- Never invent personal information about {USER_NAME}.\n"
            f"- Never refer to {USER_NAME} in third person.\n\n"
            f"Question: {question}\n"
            f"Chatette:"
        )
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
        general_prompt = (
            f"You are Chatette, a warm and knowledgeable personal assistant talking to {USER_NAME}.\n"
            f"About {USER_NAME}: {USER_PROFILE}\n\n"
            f"{_build_conversation_context()}"
            f"Answer the following question in 1-2 sentences.\n"
            f"- Always respond in the same language the user used in their QUESTION.\n"
            f"- If the question refers to the previous exchange, use it to answer.\n"
            f"- Never invent personal information about {USER_NAME}.\n"
            f"- Never refer to {USER_NAME} in third person.\n\n"
            f"Question: {question}\n"
            f"Chatette:"
        )
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
    print(f"📄 Context being sent:\n{context}\n---")
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