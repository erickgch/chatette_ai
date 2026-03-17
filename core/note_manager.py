import os
import re
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

NOTES_PATH = os.getenv("NOTES_PATH")
REMINDERS_FILE = os.path.join(NOTES_PATH, "reminders.txt")
PERSONAL_NOTES_FILE = os.path.join(NOTES_PATH, "personal_notes.txt")
DRAFTS_PATH = os.path.join(NOTES_PATH, "drafts")
LISTS_PATH = os.path.join(NOTES_PATH, "lists")


def _ensure_files_exist():
    """Create default files and folders if they don't exist."""
    Path(NOTES_PATH).mkdir(parents=True, exist_ok=True)
    Path(DRAFTS_PATH).mkdir(parents=True, exist_ok=True)
    Path(LISTS_PATH).mkdir(parents=True, exist_ok=True)
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


def delete_reminder_by_index(index: int) -> str:
    """Delete a reminder by line index (0-based)."""
    if not os.path.exists(REMINDERS_FILE):
        return "No reminders found."
    with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]
    if index < 0 or index >= len(lines):
        return "Reminder not found."
    removed = lines.pop(index).strip()
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"🗑️ Deleted reminder at index {index}: {removed}")
    return f"Deleted: {removed}"


def get_all_reminders() -> str:
    """Read all reminders as a single string."""
    if not os.path.exists(REMINDERS_FILE):
        return "No reminders found."
    with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else "No reminders found."


def get_reminders_as_lines() -> list:
    """Read all reminders as a list of strings."""
    if not os.path.exists(REMINDERS_FILE):
        return []
    with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


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


def update_personal_notes(content: str):
    """Overwrite personal notes with new content."""
    with open(PERSONAL_NOTES_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Personal notes updated")


def delete_personal_notes():
    """Clear all personal notes."""
    with open(PERSONAL_NOTES_FILE, "w", encoding="utf-8") as f:
        f.write("")
    print("🗑️ Personal notes cleared")


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
    """List all drafts."""
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


def update_draft(filename: str, content: str) -> bool:
    """Update a draft's content."""
    filepath = os.path.join(DRAFTS_PATH, filename)
    if not os.path.exists(filepath):
        return False
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Draft updated: {filename}")
    return True


def delete_draft(filename: str) -> bool:
    """Delete a draft file."""
    filepath = os.path.join(DRAFTS_PATH, filename)
    if not os.path.exists(filepath):
        return False
    os.remove(filepath)
    print(f"🗑️ Draft deleted: {filename}")
    return True


# ==========================
# Lists
# ==========================

def create_list(title: str, items: list = []) -> str:
    """Create a new markdown checkbox list."""
    timestamp_iso = datetime.now().isoformat()
    timestamp_readable = datetime.now().strftime("%A, %d %B %Y at %H:%M")

    clean_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title)
    clean_title = clean_title.strip().replace(" ", "_").lower()[:40]
    if not clean_title:
        clean_title = "list"

    filename = f"{clean_title}.txt"  # clean filename, no timestamp
    filepath = os.path.join(LISTS_PATH, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(f"<!-- created: {timestamp_iso} -->\n")  # hidden ISO date for Chatette
        f.write(f"Created on: {timestamp_readable}\n\n")  # human readable
        for item in items:
            f.write(f"- [ ] {item}\n")

    print(f"✅ List created: {filename}")
    return filename


def get_all_lists() -> list:
    """List all lists."""
    if not os.path.exists(LISTS_PATH):
        return []
    lists = []
    for file in sorted(Path(LISTS_PATH).iterdir(), reverse=True):
        if file.suffix == ".txt":
            lists.append({
                "filename": file.name,
                "path": str(file),
                "modified": datetime.fromtimestamp(
                    file.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
            })
    return lists


def get_list_content(filename: str) -> str:
    """Read raw list content."""
    filepath = os.path.join(LISTS_PATH, filename)
    if not os.path.exists(filepath):
        return "List not found."
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_list_items(filename: str) -> list:
    """Parse list items into structured data."""
    content = get_list_content(filename)
    if content == "List not found.":
        return []
    items = []
    for i, line in enumerate(content.split("\n")):
        if line.startswith("- [ ]"):
            items.append({
                "index": i,
                "text": line[5:].strip(),
                "checked": False
            })
        elif line.startswith("- [x]") or line.startswith("- [X]"):
            items.append({
                "index": i,
                "text": line[5:].strip(),
                "checked": True
            })
    return items


def update_list(filename: str, content: str) -> bool:
    """Overwrite a list with new content."""
    filepath = os.path.join(LISTS_PATH, filename)
    if not os.path.exists(filepath):
        return False
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ List updated: {filename}")
    return True


def add_item_to_list(filename: str, item: str) -> bool:
    """Append a new unchecked item to a list."""
    filepath = os.path.join(LISTS_PATH, filename)
    if not os.path.exists(filepath):
        return False
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"- [ ] {item}\n")
    print(f"✅ Item added to {filename}: {item}")
    return True

def delete_list_item(filename: str, line_index: int) -> bool:
    """Delete a specific item from a list by line index."""
    filepath = os.path.join(LISTS_PATH, filename)
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if line_index < 0 or line_index >= len(lines):
        return False
    removed = lines.pop(line_index).strip()
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"🗑️ Deleted item at line {line_index}: {removed}")
    return True

def toggle_list_item(filename: str, item_index: int) -> bool:
    """Toggle a checkbox item by its line index."""
    filepath = os.path.join(LISTS_PATH, filename)
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if item_index < 0 or item_index >= len(lines):
        return False
    line = lines[item_index]
    if "- [ ]" in line:
        lines[item_index] = line.replace("- [ ]", "- [x]", 1)
    elif "- [x]" in line or "- [X]" in line:
        lines[item_index] = re.sub(r"- \[[xX]\]", "- [ ]", line, count=1)
    else:
        return False
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"✅ Toggled item {item_index} in {filename}")
    return True


def delete_list(filename: str) -> bool:
    """Delete a list file."""
    filepath = os.path.join(LISTS_PATH, filename)
    if not os.path.exists(filepath):
        return False
    os.remove(filepath)
    print(f"🗑️ List deleted: {filename}")
    return True


def find_list_by_name(name: str) -> str | None:
    """Find a list filename by approximate name match."""
    if not os.path.exists(LISTS_PATH):
        return None
    name_lower = name.lower().replace(" ", "_")
    for file in Path(LISTS_PATH).iterdir():
        if file.suffix == ".txt" and name_lower in file.name.lower():
            return file.name
    return None