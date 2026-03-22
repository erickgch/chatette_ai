import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import itertools
from pydantic import BaseModel, field_validator, ValidationError, ConfigDict

load_dotenv()

NOTES_PATH = os.getenv("NOTES_PATH")
REMINDERS_FILE = os.path.join(NOTES_PATH, "reminders.txt")
PERSONAL_NOTES_FILE = os.path.join(NOTES_PATH, "personal_notes.txt")
LISTS_PATH = os.path.join(NOTES_PATH, "lists")
SYNC_META_FILE = os.path.join(NOTES_PATH, "sync_meta.json")


# ===========================
# Pydantic models
# ===========================

class SyncState(BaseModel):
    model_config = ConfigDict(extra="allow")

    last_sync: str = datetime.min.isoformat()
    device_id: str = ""
    pending_changes: list[str] = []

    @field_validator("last_sync", mode="before")
    @classmethod
    def validate_last_sync(cls, v: str) -> str:
        datetime.fromisoformat(v)
        return v


# ===========================
# Type-aware merge helpers
# ===========================

def _merge_reminders(pc_content: str, phone_content: str) -> str:
    """Union merge — deduplicate by stripped line text."""
    seen = set()
    merged = []
    for line in (pc_content + "\n" + phone_content).splitlines():
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            merged.append(line)
    return "\n".join(merged)


def _parse_list_items(content: str) -> list:
    """Parse markdown checklist lines → list of {text, checked}."""
    items = []
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("- [x] ") or s.startswith("- [X] "):
            items.append({"text": s[6:], "checked": True})
        elif s.startswith("- [ ] "):
            items.append({"text": s[6:], "checked": False})
        elif s:
            items.append({"text": s, "checked": None})
    return items


def _merge_list_content(pc_content: str, phone_content: str, pc_newer: bool) -> str:
    """Union merge list items; checked state taken from newer side."""
    pc_items = {item["text"]: item for item in _parse_list_items(pc_content)}
    phone_items = {item["text"]: item for item in _parse_list_items(phone_content)}

    merged = dict(pc_items)
    for text, item in phone_items.items():
        if text in merged:
            if not pc_newer and item["checked"] is not None:
                merged[text] = dict(merged[text])
                merged[text]["checked"] = item["checked"]
        else:
            merged[text] = item

    lines = []
    for item in merged.values():
        if item["checked"] is None:
            lines.append(item["text"])
        elif item["checked"]:
            lines.append(f"- [x] {item['text']}")
        else:
            lines.append(f"- [ ] {item['text']}")
    return "\n".join(lines)


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
    """Load sync metadata, validated via SyncState. Falls back to fresh state on error."""
    try:
        if os.path.exists(SYNC_META_FILE):
            with open(SYNC_META_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            state = SyncState.model_validate(raw)
            return state.model_dump()
    except (ValidationError, Exception) as e:
        print(f"⚠️ Could not load sync meta, starting fresh: {e}")
    return SyncState().model_dump()


def _save_sync_meta(meta: dict):
    try:
        validated = SyncState.model_validate(meta)
        with open(SYNC_META_FILE, "w", encoding="utf-8") as f:
            json.dump(validated.model_dump(), f, ensure_ascii=False, indent=2)
    except ValidationError as e:
        print(f"⚠️ Invalid sync meta, writing as-is: {e}")
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

    # Consume pending conflict files (generated during push)
    meta = _load_sync_meta()
    conflict_files = meta.pop("pending_conflict_files", [])
    if conflict_files:
        _save_sync_meta(meta)

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
        "lists": lists,
        "conflict_files": conflict_files
    }


# ===========================
# Push — phone sends changes to PC
# ===========================

def apply_push(payload: dict) -> dict:
    """
    Apply changes from phone to PC files using type-aware merge.
    - Reminders: union merge (deduplicate by text)
    - Lists: union merge per file (deduplicate by item text, checked from newer side)
    - Personal notes: keep PC on conflict, save phone version as journal_offline_edit_<ts>.txt
    """
    meta = _load_sync_meta()
    updated = []
    conflict_files = []

    # Reminders — union merge
    phone_reminders = payload.get("reminders")
    if phone_reminders and phone_reminders.get("content") is not None:
        phone_modified = phone_reminders.get("modified", "")
        last_sync = meta.get("reminders_last_sync", datetime.min.isoformat())

        if phone_modified > last_sync:
            pc_content = _read_file(REMINDERS_FILE)
            merged = _merge_reminders(pc_content, phone_reminders["content"])
            _write_file(REMINDERS_FILE, merged)
            meta["reminders_last_sync"] = datetime.now().isoformat()
            updated.append("reminders")
            print("✅ Reminders merged from phone")

    # Personal notes — conflict handling
    phone_notes = payload.get("personal_notes")
    if phone_notes and phone_notes.get("content") is not None:
        phone_modified = phone_notes.get("modified", "")
        pc_modified = _get_file_modified(PERSONAL_NOTES_FILE)
        last_sync = meta.get("notes_last_sync", datetime.min.isoformat())

        if phone_modified > last_sync:
            if pc_modified > last_sync:
                # Both sides changed — conflict: keep PC, save phone copy
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                conflict_filename = f"journal_offline_edit_{ts}.txt"
                conflict_path = os.path.join(NOTES_PATH, conflict_filename)
                _write_file(conflict_path, phone_notes["content"])
                pending = meta.get("pending_conflict_files", [])
                pending.append(conflict_filename)
                meta["pending_conflict_files"] = pending
                conflict_files.append(conflict_filename)
                print(f"⚠️ Notes conflict — saved phone copy as {conflict_filename}")
            else:
                # Only phone changed — phone wins
                _write_file(PERSONAL_NOTES_FILE, phone_notes["content"])
                updated.append("personal_notes")
                print("✅ Personal notes updated from phone")
            meta["notes_last_sync"] = phone_modified

    # Lists — union merge per file
    phone_lists = payload.get("lists", [])
    for phone_list in phone_lists:
        filename = phone_list.get("filename", "")
        if not filename:
            continue

        list_path = os.path.join(LISTS_PATH, filename)
        phone_modified = phone_list.get("modified", "")
        last_sync = meta.get(f"list_{filename}_last_sync", datetime.min.isoformat())

        if phone_modified > last_sync:
            pc_content = _read_file(list_path)
            pc_modified = _get_file_modified(list_path)
            pc_newer = pc_modified >= phone_modified
            merged = _merge_list_content(pc_content, phone_list["content"], pc_newer)
            _write_file(list_path, merged)
            meta[f"list_{filename}_last_sync"] = datetime.now().isoformat()
            updated.append(f"list:{filename}")
            print(f"✅ List '{filename}' merged from phone")

    _save_sync_meta(meta)
    return {"updated": updated, "conflict_files": conflict_files, "synced_at": datetime.now().isoformat()}