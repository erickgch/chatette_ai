"""One-off migration of data/docs/personal_notes.txt to the one-note-per-header format.

  python tools/migrate_personal_notes.py            → writes data/personal_notes.preview.txt
  python tools/migrate_personal_notes.py --apply    → backs up and replaces personal_notes.txt

Conversion (as agreed on 2026-10-30):
  - "About me" bullets → one timeless note each
  - training log lines → stay in the 2026-09-14 marathon note (they follow it in the file)
  - "Marathon finished successfully." → its own note dated 2026-10-04 (race day)
It also checks whether the duplicate journal_offline_edit_*.txt holds anything that is not
in personal_notes.txt (it is only deleted with --apply when it doesn't).
"""
import re, shutil, sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
import os
os.chdir(ROOT)
from note_manager import parse_personal_notes, serialize_personal_notes  # noqa: E402

NOTES = ROOT / "data" / "docs" / "personal_notes.txt"
PREVIEW = ROOT / "data" / "personal_notes.preview.txt"
DUPLICATES = sorted((ROOT / "data" / "docs").glob("journal_offline_edit_*.txt"))
MARATHON_DONE = "Marathon finished successfully."
RACE_DAY = "2026-10-04 00:00"


def _about_me_bullets(text: str) -> list[str] | None:
    """The bullets of an 'About me' block, or None if the text isn't one."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines or lines[0].strip().lower() != "about me:":
        return None
    rest = lines[1:]
    if any(not l.strip().startswith("-") for l in rest):
        raise SystemExit(f"Unexpected non-bullet line in 'About me': {[l for l in rest if not l.strip().startswith('-')]}")
    return [re.sub(r"^\s*-\s*", "", l).strip() for l in rest]


def convert(content: str) -> list[dict]:
    preamble, notes = parse_personal_notes(content)

    # "About me" bullets → timeless notes. The block is either text before the first header
    # or a note of its own (e.g. "[2026-01-01 06:00] About me: …").
    bullets = []
    if preamble:
        found = _about_me_bullets(preamble)
        if found is None:
            raise SystemExit(f"Unexpected text before the first note, not converted: {preamble[:200]!r}")
        bullets += found
    kept = []
    for n in notes:
        found = _about_me_bullets(n["text"])
        if found is None:
            kept.append(n)
        else:
            bullets += found
    notes = kept
    # Each timeless note needs a unique creation time (its ID): the migration time, one minute apart.
    start = datetime.now().replace(second=0, microsecond=0)
    timeless = [{"id": c, "created": c, "timeless": True, "text": b}
                for b, c in ((b, (start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M"))
                             for i, b in enumerate(bullets))]

    # "Marathon finished successfully." → its own note on race day
    extra = []
    for n in notes:
        if MARATHON_DONE in n["text"]:
            n["text"] = "\n".join(l for l in n["text"].splitlines() if l.strip() != MARATHON_DONE).strip()
            extra.append({"id": RACE_DAY, "created": RACE_DAY, "timeless": False, "text": MARATHON_DONE})
    dated = sorted(notes + extra, key=lambda n: n["created"])
    return timeless + dated


def duplicate_report() -> list[tuple[Path, list[str]]]:
    """Lines of each duplicate file that don't appear in personal_notes.txt (normalized)."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    have = norm(NOTES.read_text(encoding="utf-8"))
    out = []
    for f in DUPLICATES:
        missing = [l.strip() for l in f.read_text(encoding="utf-8").splitlines()
                   if l.strip() and norm(l) not in have]
        out.append((f, missing))
    return out


def main():
    apply = "--apply" in sys.argv
    notes = convert(NOTES.read_text(encoding="utf-8"))
    text = serialize_personal_notes("", notes)

    print(f"{len(notes)} notes: {sum(n['timeless'] for n in notes)} timeless, "
          f"{sum(not n['timeless'] for n in notes)} timestamped")
    dups = duplicate_report()
    for f, missing in dups:
        print(f"\nDuplicate {f.name}: {len(missing)} line(s) not found in personal_notes.txt")
        for m in missing:
            print(f"   {m}")

    if not apply:
        PREVIEW.write_text(text, encoding="utf-8")
        print(f"\nPreview written to {PREVIEW} — nothing else was changed.")
        return

    backup = NOTES.with_name(f"personal_notes.backup_{datetime.now():%Y%m%d_%H%M%S}.txt.bak")
    shutil.copy2(NOTES, backup)
    NOTES.write_text(text, encoding="utf-8")
    print(f"\nApplied. Backup of the old file: {backup.name}")
    for f, missing in dups:
        if missing:
            print(f"Kept {f.name}: it contains lines not in personal_notes.txt (see above).")
        else:
            moved = f.with_suffix(".txt.deleted")
            f.rename(moved)
            print(f"Removed duplicate {f.name} (renamed to {moved.name}, delete it whenever you like).")


if __name__ == "__main__":
    main()
