import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from ingestion import ingest_notes
from note_manager import save_reminder, delete_reminder_by_line, get_all_reminders
from datetime import datetime, timedelta

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
USER_NAME = os.getenv("USER_NAME")
USER_PROFILE = os.getenv("USER_PROFILE")

# Initialize embeddings
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model="nomic-embed-text"
)

# Connect to existing ChromaDB
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# Initialize LLM
llm = OllamaLLM(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL
)
print(f"LLM loaded: {OLLAMA_MODEL}")

# Custom prompt
prompt_template = PromptTemplate(
    template="""You are Chatette, a warm and friendly personal assistant.
You are talking to {user_name}. About {user_name}: {user_profile}

STRICT RULES:
- Only answer what was specifically asked. Nothing more.
- If the user asks about tomorrow, only mention tomorrow. Do not mention other days.
- If the user asks about appointments, only mention appointments. Do not mention shopping or other topics.
- Never volunteer extra information that wasn't asked for.
- Keep answers to 1-3 sentences maximum.
- Never calculate days of the week from memory — always derive them from the current date provided in the context.

If you find relevant information in the context, use it naturally in your response.
If you don't find the answer in the context, say so briefly.

Example:
Context: Today is Monday, March 16, 2026. Calendar event: Dentist on Wednesday, March 18 at 10am.
{user_name}: Do I have anything this week?
Chatette: Yes! You've got a dentist appointment on Wednesday at 10am. Don't forget!

Example:
Context: User enjoys technology and building AI assistants.
{user_name}: Can you recommend a podcast?
Chatette: Since you're into tech and AI, you might enjoy Lex Fridman's podcast or the TWIML AI Podcast!

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question", "user_name", "user_profile"]
)

# ===== Pending state =====
pending_reminder = {
    "text": None,
    "action": None,
    "conflict": None,
    "line_to_delete": None
}

# ===== Triggers =====
REMINDER_TRIGGERS = [
    "remind me",
    "remember that",
    "don't forget",
    "make a note",
    "add a note",
    "save this",
    "note that",
    "write down"
]

DELETE_TRIGGERS = [
    "delete the reminder",
    "remove the reminder",
    "cancel the reminder",
    "delete the note",
    "remove the note",
    "forget about",
    "remove the event",
    "delete the event"
]

CHATETTE_TRIGGERS = [
    "are you", "what are you", "who are you",
    "what's your name", "what is your name",
    "can you", "do you"
]

def is_about_chatette(question: str) -> bool:
    return any(trigger in question.lower() for trigger in CHATETTE_TRIGGERS)

def is_about_chatette(question: str) -> bool:
    return any(trigger in question.lower() for trigger in CHATETTE_TRIGGERS)

def is_reminder_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in REMINDER_TRIGGERS)


def is_delete_request(question: str) -> bool:
    return any(trigger in question.lower() for trigger in DELETE_TRIGGERS)


# ===== Reminder Creation =====
def handle_reminder(question: str) -> str:
    """Extract reminder, check for conflicts, ask for confirmation."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    all_reminders = get_all_reminders()

    extraction_prompt = f"""Today is {current_date}.
The user said: "{question}"

Extract the reminder they want to save.
Convert any relative dates (tomorrow, next Monday, in 3 days, etc.) into actual calendar dates.
Return ONLY the reminder text with the resolved date, nothing else, no punctuation at the end.

Examples:
Today is Monday, March 16, 2026. User: "remind me dinner with friends tomorrow at 7pm"
Output: "Dinner with friends on Tuesday, March 17, 2026 at 7pm"

Today is Monday, March 16, 2026. User: "remind me dentist next Friday"
Output: "Dentist appointment on Friday, March 20, 2026"
"""
    reminder_text = llm.invoke(extraction_prompt).strip()
    reminder_text = reminder_text.split("\n")[0].strip().strip('"').strip("'")

    # Check for duplicates or conflicts
    if all_reminders != "No reminders found.":
        conflict_prompt = f"""You are checking a list of reminders for duplicates or conflicts.

New reminder to save: "{reminder_text}"

Existing reminders:
{all_reminders}

Does any existing reminder duplicate or conflict with the new one?
Reply with one of:
- "DUPLICATE: <exact line>" if it is the same event
- "CONFLICT: <exact line>" if it is a different event on the same date/time
- "CLEAR" if there are no issues
"""
        conflict_check = llm.invoke(conflict_prompt).strip()
        print(f"Conflict check: '{conflict_check}'")

        if conflict_check.startswith("DUPLICATE:"):
            existing_line = conflict_check.replace("DUPLICATE:", "").strip()
            pending_reminder.update({
                "text": reminder_text,
                "action": "save",
                "conflict": "duplicate",
                "line_to_delete": existing_line
            })
            return (f"You already have something similar: '{existing_line}'. "
                    f"Do you want to replace it with '{reminder_text}'?")

        elif conflict_check.startswith("CONFLICT:"):
            existing_line = conflict_check.replace("CONFLICT:", "").strip()
            pending_reminder.update({
                "text": reminder_text,
                "action": "save",
                "conflict": "conflict",
                "line_to_delete": existing_line
            })
            return (f"Heads up - you already have '{existing_line}' around that time. "
                    f"Do you still want to save '{reminder_text}'?")

    # No conflict - ask simple confirmation
    pending_reminder.update({
        "text": reminder_text,
        "action": "save",
        "conflict": None,
        "line_to_delete": None
    })
    return f"Just to confirm - should I save this: '{reminder_text}'?"


# ===== Reminder Deletion =====
def handle_delete(question: str) -> str:
    """Use LLM to find matching reminder and ask confirmation."""
    all_reminders = get_all_reminders()

    if all_reminders == "No reminders found.":
        return "You don't have any saved reminders."

    match_prompt = f"""You are helping delete a reminder from a list.

The user wants to delete: "{question}"

Here is the COMPLETE list of reminders:
{all_reminders}

Instructions:
- Read each reminder carefully
- Find the one that best matches what the user wants to delete, even if the wording is different
- Use reasoning: "dinner with Annika and Ines" matches "[2026-03-11 20:41] Dinner with Ines and Annika on March 12th, at 18h."
- Return ONLY the exact line including the timestamp, copy it character by character
- If truly nothing matches, return NO_MATCH

Your answer:"""

    matched_line = llm.invoke(match_prompt).strip()
    matched_line = matched_line.strip('"').strip("'")
    print(f"LLM matched: '{matched_line}'")

    if "NO_MATCH" in matched_line or not matched_line:
        return "I couldn't find a matching reminder. Could you be more specific?"

    pending_reminder.update({
        "text": None,
        "action": "delete",
        "conflict": None,
        "line_to_delete": matched_line
    })
    return f"Should I delete this reminder: '{matched_line}'?"


# ===== Confirmation Handler =====
def handle_confirmation(question: str) -> str:
    """Handle yes/no confirmation for pending save or delete."""
    action = pending_reminder["action"]
    reminder_text = pending_reminder["text"]
    line_to_delete = pending_reminder["line_to_delete"]
    conflict = pending_reminder["conflict"]

    is_yes = any(word in question.lower() for word in
                 ["yes", "correct", "right", "yep", "yeah", "sure", "ok", "okay"])
    is_no = any(word in question.lower() for word in
                ["no", "wrong", "incorrect", "nope", "cancel"])

    if is_yes:
        if action == "save":
            if line_to_delete and conflict in ["duplicate", "conflict"]:
                delete_reminder_by_line(line_to_delete)
            save_reminder(reminder_text)
            ingest_notes()
            pending_reminder.update({"text": None, "action": None,
                                     "conflict": None, "line_to_delete": None})
            return f"Got it! I've saved: '{reminder_text}'"

        elif action == "delete":
            result = delete_reminder_by_line(line_to_delete)
            ingest_notes()
            pending_reminder.update({"text": None, "action": None,
                                     "conflict": None, "line_to_delete": None})
            return result

    elif is_no:
        pending_reminder.update({"text": None, "action": None,
                                 "conflict": None, "line_to_delete": None})
        return "No problem, I've discarded it!"

    else:
        if action == "save":
            return f"Should I save '{reminder_text}'? Just say yes or no."
        elif action == "delete":
            return f"Should I delete '{line_to_delete}'? Just say yes or no."


# ===== Main ask function =====
RELEVANCE_THRESHOLD = 0.9

def ask(question: str) -> str:
    print(f"\nYou: {question}")

    # Check for pending confirmation
    if pending_reminder["action"] is not None:
        answer = handle_confirmation(question)
        print(f"Chatette: {answer}")
        return answer

    # Check for delete request
    if is_delete_request(question):
        answer = handle_delete(question)
        print(f"Chatette: {answer}")
        return answer

    # Check for save reminder
    if is_reminder_request(question):
        answer = handle_reminder(question)
        print(f"Chatette: {answer}")
        return answer

    # Check if question is about Chatette herself
    if is_about_chatette(question):
        general_prompt = (
            f"You are Chatette, {USER_NAME}'s friendly personal assistant.\n"
            f"About {USER_NAME}: {USER_PROFILE}\n\n"
            f"Answer the following question naturally in 1-2 sentences.\n\n"
            f"Question: {question}\n"
            f"Chatette:"
        )
        answer = llm.invoke(general_prompt).strip()
        print(f"Chatette: {answer}")
        return answer

    # Step 1: Search each collection with relevance scores
    all_docs = []
    best_score = float("inf")

    for collection in ["calendar", "emails", "notes", "documents"]:
        k = 7 if collection == "emails" else 3
        try:
            results = vectorstore.similarity_search_with_score(
                question,
                k=k,
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

    # Step 2: If nothing relevant found, use general knowledge only
    if best_score > RELEVANCE_THRESHOLD or not all_docs:
        print("Using general knowledge...")
        general_prompt = (
            f"You are Chatette, {USER_NAME}'s friendly personal assistant.\n"
            f"About {USER_NAME}: {USER_PROFILE}\n\n"
            f"Answer the following question naturally in 1-2 sentences.\n"
            f"Do not invent information. If you don't know, say so honestly.\n\n"
            f"Question: {question}\n"
            f"Chatette:"
        )
        answer = llm.invoke(general_prompt).strip()
        print(f"Chatette: {answer}")
        return answer

    # Step 3: Build context from relevant chunks
    context = "\n\n".join([doc.page_content for doc in all_docs])

    # Step 4: Add current date
    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y at %I:%M %p")
    tomorrow = (now + timedelta(days=1)).strftime("%A, %B %d, %Y")
    day_after = (now + timedelta(days=2)).strftime("%A, %B %d, %Y")

    context = f"""Today is: {current_date}
Tomorrow is: {tomorrow}
The day after tomorrow is: {day_after}

{context}"""

    # Step 5: Format and send to LLM
    formatted_prompt = prompt_template.format(
        context=context,
        question=question,
        user_name=USER_NAME,
        user_profile=USER_PROFILE
    )

    answer = llm.invoke(formatted_prompt).strip()
    print(f"Chatette: {answer}")
    return answer


if __name__ == "__main__":
    ask("Do I have any appointments coming up?")
    ask("What do I need to buy at the supermarket?")
