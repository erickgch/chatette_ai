import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

NOTES_PATH = os.getenv("NOTES_PATH")
REMINDERS_FILE = os.path.join(NOTES_PATH, "reminders.txt")


def save_reminder(text: str):
    """Append a reminder to the reminders file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(REMINDERS_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")
    print(f"✅ Reminder saved: {text}")


def delete_reminder_by_line(line_to_delete: str) -> str:
    """Delete a reminder by closest line match."""
    if not os.path.exists(REMINDERS_FILE):
        return "No reminders found."

    with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    matching = [l for l in lines if line_to_delete.strip() in l.strip()
                or l.strip() in line_to_delete.strip()]
    remaining = [l for l in lines if l not in matching]

    if not matching:
        return "I couldn't find that reminder."

    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        f.writelines(remaining)

    deleted = [l.strip() for l in matching]
    print(f"🗑️ Deleted: {deleted}")
    return f"Done! I've removed: {', '.join(deleted)}"


def get_all_reminders() -> str:
    """Read all reminders from file."""
    if not os.path.exists(REMINDERS_FILE):
        return "No reminders found."
    with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else "No reminders found."