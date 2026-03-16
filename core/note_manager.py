import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

NOTES_PATH = os.getenv("NOTES_PATH")
REMINDERS_FILE = os.path.join(NOTES_PATH, "reminders.txt")
PERSONAL_NOTES_FILE = os.path.join(NOTES_PATH, "personal_notes.txt")
DRAFTS_PATH = os.path.join(NOTES_PATH, "drafts")


def _ensure_files_exist():
    """Create default files and folders if they don't exist."""
    Path(NOTES_PATH).mkdir(parents=True, exist_ok=True)
    Path(DRAFTS_PATH).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(REMINDERS_FILE):
        open(REMINDERS_FILE, "w", encoding="utf-8").close()
    if not os.path.exists(PERSONAL_NOTES_FILE):
        open(PERSONAL_NOTES_FILE, "w", encoding="utf-8").close()

_ensure_files_exist()


# ==========================
# Reminders
# ==========================

def save_reminder(text: str):
    """Append a reminder to reminders.txt."""
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
        f.writelines(l for l in remaining if l.strip())
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


# ==========================
# Personal Notes
# ==========================

def save_personal_note(text: str):
    """Append a note to personal_notes.txt."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PERSONAL_NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")
    print(f"✅ Personal note saved: {text}")


def get_all_personal_notes() -> str:
    """Read all personal notes."""
    if not os.path.exists(PERSONAL_NOTES_FILE):
        return "No personal notes found."
    with open(PERSONAL_NOTES_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else "No personal notes found."


# ==========================
# Drafts
# ==========================

def save_draft(title: str, content: str) -> str:
    """Save a draft to the drafts folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title)
    clean_title = clean_title.strip().replace(" ", "_").lower()[:40]
    if not clean_title:
        clean_title = "draft"
    filename = f"{clean_title}_{timestamp}.txt"
    filepath = os.path.join(DRAFTS_PATH, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Draft: {title}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("-" * 40 + "\n\n")
        f.write(content)
    print(f"✅ Draft saved: {filename}")
    return filename


def get_all_drafts() -> list:
    """List all drafts in the drafts folder."""
    if not os.path.exists(DRAFTS_PATH):
        return []
    drafts = []
    for file in sorted(Path(DRAFTS_PATH).iterdir(), reverse=True):
        if file.suffix == ".txt":
            drafts.append({
                "filename": file.name,
                "path": str(file),
                "modified": datetime.fromtimestamp(
                    file.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
            })
    return drafts


def get_draft_content(filename: str) -> str:
    """Read a specific draft."""
    filepath = os.path.join(DRAFTS_PATH, filename)
    if not os.path.exists(filepath):
        return "Draft not found."
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()