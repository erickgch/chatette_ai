import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import itertools

load_dotenv()

NOTES_PATH = os.getenv("NOTES_PATH")
REMINDERS_FILE = os.path.join(NOTES_PATH, "reminders.txt")
PERSONAL_NOTES_FILE = os.path.join(NOTES_PATH, "personal_notes.txt")
LISTS_PATH = os.path.join(NOTES_PATH, "lists")
SYNC_META_FILE = os.path.join(NOTES_PATH, "sync_meta.json")


def _get_file_modified(path: str) -> str:
    """Return ISO timestamp of file last modification."""
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return datetime.min.isoformat()


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _load_sync_meta() -> dict:
    """Load sync metadata (last known phone timestamps)."""
    try:
        if os.path.exists(SYNC_META_FILE):
            with open(SYNC_META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_sync_meta(meta: dict):
    with open(SYNC_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ===========================
# Pull — phone requests current PC state
# ===========================

def build_pull_response() -> dict:
    """Return current state of all data for phone to cache."""
    # Reminders
    reminders_content = _read_file(REMINDERS_FILE)
    reminders_modified = _get_file_modified(REMINDERS_FILE)

    # Personal notes
    notes_content = _read_file(PERSONAL_NOTES_FILE)
    notes_modified = _get_file_modified(PERSONAL_NOTES_FILE)

    # Lists
    lists = []
    lists_path = Path(LISTS_PATH)
    if lists_path.exists():
        for f in sorted(itertools.chain(
                lists_path.glob("*.md"),
                lists_path.glob("*.txt")
        )):
            content = _read_file(str(f))
            lists.append({
                "filename": f.name,
                "content": content,
                "modified": _get_file_modified(str(f))
            })

    return {
        "pulled_at": datetime.now().isoformat(),
        "reminders": {
            "content": reminders_content,
            "modified": reminders_modified
        },
        "personal_notes": {
            "content": notes_content,
            "modified": notes_modified
        },
        "lists": lists
    }


# ===========================
# Push — phone sends changes to PC
# ===========================

def apply_push(payload: dict) -> dict:
    """
    Apply changes from phone to PC files.
    Uses last-write-wins based on timestamps.
    """
    meta = _load_sync_meta()
    updated = []

    # Reminders
    phone_reminders = payload.get("reminders")
    if phone_reminders:
        phone_modified = phone_reminders.get("modified", "")
        pc_modified = _get_file_modified(REMINDERS_FILE)
        last_sync = meta.get("reminders_last_sync", datetime.min.isoformat())

        # Phone is newer than last sync — phone wins
        if phone_modified > last_sync and phone_modified > pc_modified:
            _write_file(REMINDERS_FILE, phone_reminders["content"])
            meta["reminders_last_sync"] = datetime.now().isoformat()
            updated.append("reminders")
            print(f"✅ Reminders updated from phone")

    # Personal notes
    phone_notes = payload.get("personal_notes")
    if phone_notes:
        phone_modified = phone_notes.get("modified", "")
        pc_modified = _get_file_modified(PERSONAL_NOTES_FILE)
        last_sync = meta.get("notes_last_sync", datetime.min.isoformat())

        if phone_modified > last_sync and phone_modified > pc_modified:
            _write_file(PERSONAL_NOTES_FILE, phone_notes["content"])
            meta["notes_last_sync"] = datetime.now().isoformat()
            updated.append("personal_notes")
            print(f"✅ Personal notes updated from phone")

    # Lists
    phone_lists = payload.get("lists", [])
    for phone_list in phone_lists:
        filename = phone_list.get("filename", "")
        if not filename:
            continue

        list_path = os.path.join(LISTS_PATH, filename)
        phone_modified = phone_list.get("modified", "")
        pc_modified = _get_file_modified(list_path)
        last_sync = meta.get(f"list_{filename}_last_sync",
                              datetime.min.isoformat())

        if phone_modified > last_sync and phone_modified > pc_modified:
            _write_file(list_path, phone_list["content"])
            meta[f"list_{filename}_last_sync"] = datetime.now().isoformat()
            updated.append(f"list:{filename}")
            print(f"✅ List '{filename}' updated from phone")

    _save_sync_meta(meta)
    return {"updated": updated, "synced_at": datetime.now().isoformat()}