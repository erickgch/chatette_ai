import os
import time
import schedule
from datetime import datetime
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion import ingest_calendar_events, ingest_emails, ingest_notes

load_dotenv()

NOTES_PATH = os.getenv("NOTES_PATH")

# ===== Notes File Watcher =====
class NotesChangeHandler(FileSystemEventHandler):
    """Re-ingests notes whenever a file in the notes folder changes."""

    def on_modified(self, event):
        if not event.is_directory:
            print(f"\n📝 Notes changed: {event.src_path} — re-ingesting...")
            ingest_notes()
            print("✅ Notes re-ingested!")

    def on_created(self, event):
        if not event.is_directory:
            print(f"\n📝 New note detected: {event.src_path} — ingesting...")
            ingest_notes()
            print("✅ Notes ingested!")


# ===== Scheduled Jobs =====
def sync_calendar():
    print(f"\n📅 [{datetime.now().strftime('%H:%M')}] Syncing Google Calendar...")
    ingest_calendar_events()  # uses env variables directly

def sync_emails():
    print(f"\n📧 [{datetime.now().strftime('%H:%M')}] Syncing Gmail...")
    ingest_emails()  # uses env variables directly

def sync_all():
    sync_calendar()
    sync_emails()


# ===== Main Scheduler =====
def start_scheduler():
    print("🚀 Scheduler started!")
    print(f"📁 Watching notes folder: {NOTES_PATH}")

    # Schedule Google sync every 30 minutes
    schedule.every(30).minutes.do(sync_all)

    # Also sync once at startup
    print("\n🔄 Running initial sync...")
    sync_all()

    # Start notes file watcher
    event_handler = NotesChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=NOTES_PATH, recursive=False)
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