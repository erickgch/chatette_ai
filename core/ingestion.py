# ingest absorbs the information in the database, i.e., the source text files
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, ValidationError

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================
# Load environment variables
# ==========================
load_dotenv(Path(__file__).parent.parent / ".env")
os.chdir(Path(__file__).parent.parent)

from google_integration import get_upcoming_events, get_recent_emails

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")
NOTES_PATH = os.getenv("NOTES_PATH")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

EMAIL_DAYS_WINDOW = int(os.getenv("EMAIL_DAYS_WINDOW", 14))
CALENDAR_DAYS_AHEAD = int(os.getenv("CALENDAR_DAYS_AHEAD", 14))
CALENDAR_DAYS_BEHIND = int(os.getenv("CALENDAR_DAYS_BEHIND", 1))

# ==========================
# Pydantic model
# ==========================

VALID_COLLECTIONS = {"documents", "notes", "lists", "calendar", "emails"}

class ChunkMetadata(BaseModel):
    collection: str
    source_file: str

    @field_validator("collection")
    @classmethod
    def must_be_valid_collection(cls, v: str) -> str:
        if v not in VALID_COLLECTIONS:
            raise ValueError(f"Unknown collection '{v}'. Must be one of: {VALID_COLLECTIONS}")
        return v

    @field_validator("source_file")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_file must not be empty")
        return v.strip()


def _apply_chunk_metadata(chunk, collection: str, source_file: str) -> bool:
    """
    Validate metadata with ChunkMetadata, then assign fields manually to the chunk.
    Returns False and skips the chunk if validation fails.
    """
    try:
        meta = ChunkMetadata(collection=collection, source_file=source_file)
    except ValidationError as e:
        print(f"⚠️ Invalid chunk metadata — skipping chunk: {e}")
        return False
    chunk.metadata["collection"] = meta.collection
    chunk.metadata["source_file"] = meta.source_file
    return True


# ==========================
# Utility: Human-readable dates
# ==========================
def human_datetime(iso_string):
    """Convert ISO datetime to readable format."""
    if not iso_string:
        return "Unknown time"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%A, %d %B %Y at %H:%M")
    except Exception:
        return iso_string


# ==========================
# Embeddings
# ==========================
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=EMBEDDING_MODEL
)

# ==========================
# Vector store
# ==========================
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# ==========================
# Text splitters
# ==========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

email_splitter = RecursiveCharacterTextSplitter(
    chunk_size=330,
    chunk_overlap=30
)


# ==========================
# File ingestion
# ==========================
def ingest_file(filepath: str, collection: str = "documents"):
    """Ingest a single file into the vector store."""
    path = Path(filepath)

    if not path.exists():
        print(f"File not found: {filepath}")
        return

    if path.stat().st_size == 0:
        print(f"⚠️ Skipping empty file: {path.name}")
        return

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        print(f"⚠️ Skipping blank file: {path.name}")
        return

    print(f"Ingesting {path.name}...")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        print(f"Unsupported file type: {path.suffix}")
        return

    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    valid_chunks = []
    for chunk in chunks:
        if _apply_chunk_metadata(chunk, collection, path.name):
            valid_chunks.append(chunk)

    if valid_chunks:
        vectorstore.add_documents(valid_chunks)


# ==========================
# Documents ingestion
# ==========================
def ingest_all_documents():
    """Ingest all files in the documents folder."""
    docs_path = Path(DOCUMENTS_PATH)

    if not docs_path.exists():
        print("Documents folder not found.")
        return

    for file in docs_path.iterdir():
        if file.is_file():
            ingest_file(str(file), collection="documents")


# ==========================
# Notes ingestion
# ==========================
def ingest_notes():
    """Ingest all notes (reminders, personal notes, drafts), replacing old ones."""

    existing = vectorstore.get(where={"collection": "notes"})
    if existing and existing["ids"]:
        vectorstore.delete(ids=existing["ids"])

    notes_path = Path(NOTES_PATH)
    if not notes_path.exists():
        print("Notes folder not found.")
        return

    for file in notes_path.iterdir():
        if file.is_file() and file.suffix in ('.txt', '.md'):
            ingest_file(str(file), collection="notes")

    drafts_path = notes_path / "drafts"
    if drafts_path.exists():
        for file in drafts_path.iterdir():
            if file.is_file():
                ingest_file(str(file), collection="notes")


# ==========================
# Lists ingestion
# ==========================
def ingest_lists():
    """Ingest all list files, replacing old ones."""

    existing = vectorstore.get(where={"collection": "lists"})
    if existing and existing["ids"]:
        vectorstore.delete(ids=existing["ids"])

    lists_path = Path(NOTES_PATH) / "lists"
    if not lists_path.exists():
        print("Lists folder not found — skipping.")
        return

    for file in lists_path.iterdir():
        if file.is_file() and file.suffix == ".txt":
            if file.stat().st_size == 0:
                print(f"⚠️ Skipping empty list: {file.name}")
                continue
            content = file.read_text(encoding="utf-8").strip()
            if not content:
                continue

            print(f"Ingesting list: {file.name}...")
            loader = TextLoader(str(file), encoding="utf-8")
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)

            valid_chunks = []
            for chunk in chunks:
                if _apply_chunk_metadata(chunk, "lists", file.name):
                    valid_chunks.append(chunk)

            if valid_chunks:
                vectorstore.add_documents(valid_chunks)


# ==========================
# Calendar ingestion
# ==========================
def ingest_calendar_events():
    """Fetch and ingest upcoming calendar events."""

    existing = vectorstore.get(where={"collection": "calendar"})
    if existing and existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
        print(f"🗑️ Cleared {len(existing['ids'])} old calendar chunks")

    events = get_upcoming_events(
        days_ahead=CALENDAR_DAYS_AHEAD,
        days_behind=CALENDAR_DAYS_BEHIND
    )

    if not events:
        print("No calendar events found.")
        return

    documents = []

    for event in events:
        event_dict = event.model_dump() if hasattr(event, "model_dump") else event
        start = human_datetime(event_dict.get("start"))
        end = human_datetime(event_dict.get("end"))

        text = f"""
[Calendar Event]

Title: {event_dict.get('title')}
Start: {start}
End: {end}
Description: {event_dict.get('description', '')}
"""

        doc = Document(
            page_content=text.strip(),
            metadata={}
        )
        if _apply_chunk_metadata(doc, "calendar", "google_calendar"):
            documents.append(doc)

    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(chunks)


# ==========================
# Email ingestion
# ==========================
def ingest_emails():
    """Fetch and ingest recent emails."""

    existing = vectorstore.get(where={"collection": "emails"})
    if existing and existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
        print(f"🗑️ Cleared {len(existing['ids'])} old email chunks")

    emails = get_recent_emails(days_window=EMAIL_DAYS_WINDOW)

    if not emails:
        print("No emails found.")
        return

    documents = []

    for email in emails:
        date = human_datetime(email.date)

        body = email.body
        if body:
            body = body.strip()[:400]

        text = f"""
[Email]

From: {email.from_}
Date: {date}
Subject: {email.subject}

Body:
{body}
"""

        doc = Document(
            page_content=text.strip(),
            metadata={}
        )
        if _apply_chunk_metadata(doc, "emails", "gmail"):
            documents.append(doc)

    chunks = email_splitter.split_documents(documents)
    vectorstore.add_documents(chunks)


# ==========================
# Full ingestion
# ==========================
def ingest_all():
    print("\n📄 Ingesting documents...")
    ingest_all_documents()

    print("\n📝 Ingesting notes...")
    ingest_notes()

    print("\n📋 Ingesting lists...")
    ingest_lists()

    print("\n📅 Ingesting calendar events...")
    ingest_calendar_events()

    print("\n📧 Ingesting emails...")
    ingest_emails()

    print("\n✅ All done!")


if __name__ == "__main__":
    ingest_all()