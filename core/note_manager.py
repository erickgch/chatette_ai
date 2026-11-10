import os
import re
from datetime import datetime, timedelta
from typing import TypedDict
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, field_validator, ValidationError

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH")
REMINDERS_FILE = os.path.join(DOCS_PATH, "reminders.txt")
PERSONAL_NOTES_FILE = os.path.join(DOCS_PATH, "personal_notes.txt")
DRAFTS_PATH = os.path.join(DOCS_PATH, "drafts")
LISTS_PATH = os.path.join(DOCS_PATH, "lists")


def _ensure_files_exist():
    """Create default files and folders if they don't exist."""
    Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)
    Path(DRAFTS_PATH).mkdir(parents=True, exist_ok=True)
    Path(LISTS_PATH).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(REMINDERS_FILE):
        open(REMINDERS_FILE, "w", encoding="utf-8").close()
    if not os.path.exists(PERSONAL_NOTES_FILE):
        open(PERSONAL_NOTES_FILE, "w", encoding="utf-8").close()

_ensure_files_exist()


# ==========================
# Pydantic models
# ==========================

class ReminderItem(BaseModel):
    text: str
    created: str = ""
    due: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Reminder text must not be empty")
        return v.strip()

    @field_validator("due")
    @classmethod
    def due_must_be_valid_date(cls, v: str | None) -> str | None:
        if v is not None:
            datetime.strptime(v, "%Y-%m-%d")
        return v


class PersonalNoteItem(BaseModel):
    text: str
    created_at: str = ""

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Personal note text must not be empty")
        return v.strip()


class DraftItem(BaseModel):
    title: str
    content: str
    created_at: str = ""

    @field_validator("title", "content")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


class ListItem(BaseModel):
    index: int
    text: str
    checked: bool


# TypedDicts — lightweight read-only return shapes, no validation needed
class DraftMeta(TypedDict):
    filename: str
    path: str
    modified: str


class ListMeta(TypedDict):
    filename: str
    path: str
    modified: str


# ==========================
# Reminders
# ==========================

def _read_reminders() -> list[ReminderItem]:
    """Parse all reminders from JSON Lines file."""
    if not os.path.exists(REMINDERS_FILE):
        return []
    items = []
    with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(ReminderItem.model_validate_json(line))
            except Exception:
                pass  # skip malformed lines
    return items


def _write_reminders(items: list[ReminderItem]) -> None:
    """Write reminders list back to JSON Lines file."""
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def create_reminder(text: str, due: str | None = None) -> None:
    """Validate and append a reminder to reminders.txt as JSON Lines."""
    try:
        item = ReminderItem(
            text=text,
            created=datetime.now().isoformat(timespec="seconds"),
            due=due
        )
    except ValidationError as e:
        print(f"⚠️ Invalid reminder — skipping: {e}")
        return

    with open(REMINDERS_FILE, "a", encoding="utf-8") as f:
        f.write(item.model_dump_json() + "\n")
    print(f"✅ Reminder saved: {item.text}")


def delete_reminder_by_line(line_to_delete: str) -> str:
    """Delete a reminder by closest text match."""
    items = _read_reminders()
    if not items:
        return "No reminders found."
    needle = line_to_delete.strip().lower()
    matching = [i for i in items if needle in i.text.lower() or i.text.lower() in needle]
    if not matching:
        return "I couldn't find that reminder."
    remaining = [i for i in items if i not in matching]
    _write_reminders(remaining)
    deleted = [i.text for i in matching]
    print(f"🗑️ Deleted: {deleted}")
    return f"Done! I've removed: {', '.join(deleted)}"


def delete_reminder_by_index(index: int) -> str:
    """Delete a reminder by index (0-based)."""
    items = _read_reminders()
    if not items:
        return "No reminders found."
    if index < 0 or index >= len(items):
        return "Reminder not found."
    removed = items.pop(index)
    _write_reminders(items)
    print(f"🗑️ Deleted reminder at index {index}: {removed.text}")
    return f"Deleted: {removed.text}"


def delete_all_reminders() -> int:
    """Delete every reminder. Returns how many were removed."""
    items = _read_reminders()
    _write_reminders([])
    print(f"🗑️ Deleted all reminders ({len(items)})")
    return len(items)


def get_all_reminders() -> str:
    """Read all reminders as a formatted string (for LLM context)."""
    items = _read_reminders()
    if not items:
        return "No reminders found."
    lines = []
    for item in items:
        due_part = f" (due: {item.due})" if item.due else ""
        lines.append(f"- {item.text}{due_part}")
    return "\n".join(lines)


def get_reminders_as_lines() -> list[str]:
    """Read all reminders as a list of display strings."""
    items = _read_reminders()
    lines = []
    for item in items:
        due_part = f" (due: {item.due})" if item.due else ""
        lines.append(f"{item.text}{due_part}")
    return lines


def get_reminders_list() -> list[ReminderItem]:
    """Return all reminders as structured ReminderItem objects."""
    return _read_reminders()


# ==========================
# Personal Notes
# ==========================

# ==========================
# Personal notes: one note per header
# ==========================
# Every note starts with a header that is set once, at creation, and never changes — it is
# the note's identity:
#   [2026-06-03 14:37] I want to go back to teaching Spanish…        (timestamped note)
#   [timeless · created 2026-10-30 14:05] I am left-handed.          (lasting fact)
# A note's text runs until the next header and may span several lines. Each note is one
# chunk in the search index, header included.

NOTE_HEADER_RE = re.compile(
    r"^\[(?:(timeless)\s*[·\-]\s*created\s+)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)\][ \t]*",
    re.MULTILINE | re.IGNORECASE)


def note_header(created: str, timeless: bool) -> str:
    return f"[timeless · created {created}]" if timeless else f"[{created}]"


def parse_personal_notes(content: str) -> tuple[str, list[dict]]:
    """Split the notes file into (preamble, notes). The preamble is any text before the first
    header (old-format content); notes are dicts: id, created, timeless, text."""
    matches = list(NOTE_HEADER_RE.finditer(content))
    preamble = content[:matches[0].start()].strip() if matches else content.strip()
    notes = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        notes.append({
            "id": m.group(2),                    # creation time = identity
            "created": m.group(2),
            "timeless": bool(m.group(1)),
            "text": content[m.end():end].strip(),
        })
    return preamble, notes


def serialize_personal_notes(preamble: str, notes: list[dict]) -> str:
    parts = [preamble.strip()] if preamble.strip() else []
    parts += [f"{note_header(n['created'], n['timeless'])} {n['text'].strip()}" for n in notes]
    return "\n\n".join(parts) + "\n"


def _read_personal_notes() -> tuple[str, list[dict]]:
    if not os.path.exists(PERSONAL_NOTES_FILE):
        return "", []
    with open(PERSONAL_NOTES_FILE, "r", encoding="utf-8") as f:
        return parse_personal_notes(f.read())


def _write_personal_notes(preamble: str, notes: list[dict]) -> None:
    with open(PERSONAL_NOTES_FILE, "w", encoding="utf-8") as f:
        f.write(serialize_personal_notes(preamble, notes))


def list_personal_notes() -> list[dict]:
    """All notes, in file order (oldest first)."""
    return _read_personal_notes()[1]


def add_personal_note(text: str, timestamped: bool = True) -> dict | None:
    """Append a note. Its header (creation time, and 'timeless' if not timestamped) is fixed."""
    text = text.strip()
    if not text:
        return None
    preamble, notes = _read_personal_notes()
    taken = {n["id"] for n in notes}
    now = datetime.now().replace(microsecond=0)
    created = now.strftime("%Y-%m-%d %H:%M")
    if created in taken:                         # same minute → add seconds, step until unique
        while now.strftime("%Y-%m-%d %H:%M:%S") in taken:
            now += timedelta(seconds=1)
        created = now.strftime("%Y-%m-%d %H:%M:%S")
    note = {"id": created, "created": created, "timeless": not timestamped, "text": text}
    notes.append(note)
    _write_personal_notes(preamble, notes)
    print(f"✅ Personal note added: {note_header(created, note['timeless'])} {text[:60]}")
    return note


def edit_personal_note(note_id: str, text: str) -> bool:
    """Change a note's text. Its header (identity, timestamp, timeless status) stays as is."""
    preamble, notes = _read_personal_notes()
    for n in notes:
        if n["id"] == note_id:
            n["text"] = text.strip()
            _write_personal_notes(preamble, notes)
            print(f"✅ Personal note edited: {note_id}")
            return True
    return False


def delete_personal_note(note_id: str) -> bool:
    preamble, notes = _read_personal_notes()
    kept = [n for n in notes if n["id"] != note_id]
    if len(kept) == len(notes):
        return False
    _write_personal_notes(preamble, kept)
    print(f"🗑️ Personal note deleted: {note_id}")
    return True


def save_personal_note(text: str) -> None:
    """Validate and add a note to personal_notes.txt (notes added via chat are timestamped)."""
    try:
        item = PersonalNoteItem(text=text)
    except ValidationError as e:
        print(f"⚠️ Invalid personal note — skipping: {e}")
        return
    add_personal_note(item.text, timestamped=True)


def get_all_personal_notes() -> str:
    """Read all personal notes."""
    if not os.path.exists(PERSONAL_NOTES_FILE):
        return "No personal notes found."
    with open(PERSONAL_NOTES_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else "No personal notes found."


def update_personal_notes(content: str) -> None:
    """Overwrite personal notes with new content."""
    with open(PERSONAL_NOTES_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Personal notes updated")


def delete_personal_notes() -> None:
    """Clear all personal notes."""
    with open(PERSONAL_NOTES_FILE, "w", encoding="utf-8") as f:
        f.write("")
    print("🗑️ Personal notes cleared")


# ==========================
# Drafts
# ==========================

def save_draft(title: str, content: str, filename: str | None = None) -> str:
    """Validate and save a draft to the drafts folder.

    If `filename` is given (the user named the document), that name is used as-is
    with a .txt extension; an existing file is never overwritten — a _2, _3, …
    suffix is added instead. Otherwise the name is derived from the title.
    """
    try:
        item = DraftItem(
            title=title,
            content=content,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    except ValidationError as e:
        print(f"⚠️ Invalid draft — aborting: {e}")
        return ""

    requested = ""
    if filename:
        stem = os.path.splitext(os.path.basename(filename.strip()))[0]
        requested = "".join(c if c.isalnum() or c in " _-." else "" for c in stem).strip().replace(" ", "_")[:80]

    if requested:
        filename = f"{requested}.txt"
        n = 2
        while os.path.exists(os.path.join(DRAFTS_PATH, filename)):
            filename = f"{requested}_{n}.txt"
            n += 1
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_title = "".join(c if c.isalnum() or c in " _-" else "" for c in item.title)
        clean_title = clean_title.strip().replace(" ", "_").lower()[:40]
        if not clean_title:
            clean_title = "draft"
        filename = f"{clean_title}_{timestamp}.txt"
    filepath = os.path.join(DRAFTS_PATH, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Draft: {item.title}\n")
        f.write(f"Created: {item.created_at}\n")
        f.write("-" * 40 + "\n\n")
        f.write(item.content)
    print(f"✅ Draft saved: {filename}")
    return filename


def get_all_drafts() -> list[DraftMeta]:
    """List all drafts."""
    if not os.path.exists(DRAFTS_PATH):
        return []
    drafts: list[DraftMeta] = []
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

def create_list(title: str, items: list[str] = []) -> str:
    """Create a new markdown checkbox list."""
    timestamp_iso = datetime.now().isoformat()
    timestamp_readable = datetime.now().strftime("%A, %d %B %Y at %H:%M")

    clean_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title)
    clean_title = clean_title.strip().replace(" ", "_").lower()[:40]
    if not clean_title:
        clean_title = "list"

    filename = f"{clean_title}.txt"
    filepath = os.path.join(LISTS_PATH, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(f"<!-- created: {timestamp_iso} -->\n")
        f.write(f"Created on: {timestamp_readable}\n\n")
        for item in items:
            f.write(f"- [ ] {item}\n")

    print(f"✅ List created: {filename}")
    return filename


def get_all_lists() -> list[ListMeta]:
    """List all lists."""
    if not os.path.exists(LISTS_PATH):
        return []
    lists: list[ListMeta] = []
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


def get_list_items(filename: str) -> list[ListItem]:
    """Parse list items into validated ListItem models."""
    content = get_list_content(filename)
    if content == "List not found.":
        return []
    items: list[ListItem] = []
    for i, line in enumerate(content.split("\n")):
        if line.startswith("- [ ]"):
            items.append(ListItem(index=i, text=line[5:].strip(), checked=False))
        elif line.startswith("- [x]") or line.startswith("- [X]"):
            items.append(ListItem(index=i, text=line[5:].strip(), checked=True))
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