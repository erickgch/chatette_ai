import os
import time
import schedule
from datetime import datetime
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion import ingest_calendar_events, ingest_emails, ingest_notes, ingest_lists

load_dotenv()

NOTES_PATH = os.getenv("NOTES_PATH")


# ===== Notes File Watcher =====
class NotesChangeHandler(FileSystemEventHandler):
    """Re-ingests notes whenever a file in the notes folder changes."""

    def on_modified(self, event):
        if not event.is_directory:
            print(f"\n📝 Notes changed: {event.src_path} — re-ingesting...")
            ingest_notes()
            ingest_lists()
            print("✅ Notes re-ingested!")

    def on_created(self, event):
        if not event.is_directory:
            print(f"\n📝 New note detected: {event.src_path} — ingesting...")
            ingest_notes()
            ingest_lists()
            print("✅ Notes ingested!")


# ===== Scheduled Jobs =====
def sync_calendar():
    print(f"\n📅 [{datetime.now().strftime('%H:%M')}] Syncing Google Calendar...")
    ingest_calendar_events()

def sync_emails():
    print(f"\n📧 [{datetime.now().strftime('%H:%M')}] Syncing Gmail...")
    ingest_emails()

def sync_all():
    sync_calendar()
    sync_emails()

def generate_notifications_cache():
    """Generate LLM notification texts and store for phone to fetch."""
    print(f"\n🔔 [{datetime.now().strftime('%H:%M')}] Generating notifications cache...")
    try:
        from rag import llm_invoke
        from note_manager import get_all_reminders
        from google_integration import get_upcoming_events
        import json
        from pathlib import Path

        user_name = os.getenv("USER_NAME", "there")
        now = datetime.now()
        reminders = get_all_reminders()
        has_reminders = reminders != "No reminders found."

        def _generate(time_label: str, greeting: str) -> str:
            if has_reminders:
                prompt = f"""You are Chatette — a small robotic cat and personal assistant to {user_name}.
Your personality: warm and competent, like a smart friend who always has things under control.
You have your own voice — not a generic AI bot. Use cool, casual phrases naturally.
Examples: "Heads up —", "Just keeping you in the loop.", "Quick one:", "On your radar:"
Very rarely, a subtle cat hint slips through naturally.
Never say "Certainly!", "Of course!", "Great question!"
No emojis in text (they will be added separately).

It is {time_label}. Write a short notification for {user_name}.

Their current reminders:
{reminders}

Rules:
- 1-2 sentences maximum
- Mention 1-2 specific reminders naturally — don't list them all
- Start with a {time_label} opener like "{greeting}"
- Warm but not gushing — one friendly touch is enough

Write the message now:"""
            else:
                prompt = f"""You are Chatette — a small robotic cat and personal assistant to {user_name}.
Your personality: warm and competent, like a smart friend who always has things under control.
You have your own voice — not a generic AI bot. Use cool, casual phrases naturally.
Examples: "All clear.", "Nothing on the radar.", "You're all set."
Very rarely, a subtle cat hint slips through naturally.
Never say "Certainly!", "Of course!", "Great question!"
No emojis in text (they will be added separately).

It is {time_label}. {user_name} has no reminders right now.
Write a short, uplifting message for {user_name}.

Rules:
- 1-2 sentences maximum
- Start with a {time_label} opener like "{greeting}"
- Something about health, positivity, or enjoying the moment
- Casual and genuine — not motivational poster energy

Write the message now:"""
            return llm_invoke(prompt).strip()

        notifications = {
            "9h": _generate("morning", f"Good morning {user_name}!"),
            "12h": _generate("lunchtime", f"Lunchtime check-in, {user_name}!"),
            "18h": _generate("evening", f"Evening, {user_name}!"),
        }

        cache = {
            "generated_at": now.isoformat(),
            "valid_until": now.replace(hour=23, minute=59, second=59).isoformat(),
            "has_reminders": has_reminders,
            "notifications": notifications,
        }

        # Save to a JSON file so the API endpoint can serve it quickly
        cache_path = Path(os.getenv("NOTES_PATH", ".")) / "notifications_cache.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"✅ Notifications cache generated and saved to {cache_path}")
        print(f"   9h:  {notifications['9h'][:60]}...")
        print(f"   12h: {notifications['12h'][:60]}...")
        print(f"   18h: {notifications['18h'][:60]}...")

    except Exception as e:
        print(f"❌ Failed to generate notifications cache: {e}")


# ===== Main Scheduler =====
def start_scheduler():
    print("🚀 Scheduler started!")
    print(f"📁 Watching notes folder: {NOTES_PATH}")

    # Schedule Google sync every 30 minutes
    schedule.every(30).minutes.do(sync_all)

    # Generate notifications cache daily at 8am
    schedule.every().day.at("08:00").do(generate_notifications_cache)

    # Run initial sync and cache generation at startup
    print("\n🔄 Running initial sync...")
    sync_all()

    print("\n🔔 Generating initial notifications cache...")
    generate_notifications_cache()

    # Start notes file watcher
    event_handler = NotesChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=NOTES_PATH, recursive=True)
    observer.start()
    print("👀 Watching notes folder for changes...\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped.")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    start_scheduler()