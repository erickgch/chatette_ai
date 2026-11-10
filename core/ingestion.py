# ingest absorbs the information in the database, i.e., the source text files
import os
import re
import html
import time
import hashlib
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
DOCS_PATH = os.getenv("DOCS_PATH")

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
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


def _strip_html(text: str) -> str:
    """Turn an HTML email body into plain text (drop head/style/script, tags, entities)."""
    if "<" not in text or not re.search(r"<\s*(html|head|body|div|p|br|table|meta|span|!doctype)\b", text, re.I):
        return text
    text = re.sub(r"(?is)<(head|style|script)\b.*?</\1>", " ", text)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"<[^>]*$", " ", text)          # tag cut off by truncation
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ==========================
# Chunk cleaner
# ==========================
MIN_CONTENT_CHARS = 10   # letters or digits — numbers count, e.g. "06.09.26: 6,7 km."
_TAG_LINE = re.compile(r"^\s*\[[^\]\n]{1,40}\]\s*$", re.M)          # [Email], [Calendar Event]
_EMPTY_LABEL_LINE = re.compile(r"^\s*[\w ]{1,30}:\s*$", re.M | re.U)  # "Body:", "Description:"


def _has_content(text: str) -> bool:
    """True if the chunk says something beyond labels, tags and markup."""
    text = _TAG_LINE.sub(" ", text)
    text = _EMPTY_LABEL_LINE.sub(" ", text)
    text = re.sub(r"<[^>]*>?", " ", text)
    return sum(ch.isalnum() for ch in text) >= MIN_CONTENT_CHARS


def _clean_chunks(chunks: list[Document]) -> tuple[list[Document], int]:
    """Drop chunks without real content (e.g. a lone "Body:" or an HTML fragment)."""
    kept = [c for c in chunks if _has_content(c.page_content)]
    return kept, len(chunks) - len(kept)


# ==========================
# Incremental sync
# ==========================
# Every chunk gets a stable ID derived from its content. A sync only adds chunks that are
# new and deletes chunks that disappeared — unchanged chunks are never touched.
#
# Why: the old "delete everything, re-add everything" approach ran every 15 minutes.
# Chroma's HNSW index only *marks* deleted vectors, so after months it held 23,633
# entries for 38 live chunks, and filtered searches failed ("ef or M is too small")
# — notes became unreachable and calendar searches returned a single chunk.

def _chunk_id(chunk: Document) -> str:
    meta = chunk.metadata
    key = f"{meta.get('collection')}\x1f{meta.get('source_file')}\x1f{chunk.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _sync_chunks(collection: str, chunks: list[Document], source_file: str | None = None) -> None:
    """Make the stored chunks of `collection` (or of one source file in it) equal `chunks`."""
    where = {"collection": collection}
    if source_file:
        where = {"$and": [{"collection": collection}, {"source_file": source_file}]}
    existing_ids = set(vectorstore.get(where=where)["ids"])

    chunks, dropped = _clean_chunks(chunks)
    wanted: dict[str, Document] = {}
    for chunk in chunks:
        wanted.setdefault(_chunk_id(chunk), chunk)   # identical chunks collapse into one

    to_delete = list(existing_ids - wanted.keys())
    to_add = [cid for cid in wanted if cid not in existing_ids]

    if to_delete:
        vectorstore.delete(ids=to_delete)
    if to_add:
        # Retry up to 3 times — Ollama may be busy with the LLM and need a moment to swap models
        for attempt in range(3):
            try:
                vectorstore.add_documents([wanted[cid] for cid in to_add], ids=to_add)
                break
            except ConnectionError:
                if attempt < 2:
                    print(f"Embedding connection failed, retrying in 3s (attempt {attempt + 1}/3)...")
                    time.sleep(3)
                else:
                    raise

    scope = f"{collection}/{source_file}" if source_file else collection
    print(f"🔄 {scope}: +{len(to_add)} added, -{len(to_delete)} removed, "
          f"{len(wanted) - len(to_add)} unchanged" + (f", {dropped} empty chunk(s) skipped" if dropped else ""))


# ==========================
# File ingestion
# ==========================
def _file_chunks(path: Path, collection: str) -> list[Document]:
    """Load and split one file into chunks tagged with collection/source metadata."""
    if not path.exists():
        print(f"File not found: {path}")
        return []
    if path.stat().st_size == 0:
        print(f"⚠️ Skipping empty file: {path.name}")
        return []

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in [".txt", ".md"]:
        if not path.read_text(encoding="utf-8").strip():
            print(f"⚠️ Skipping blank file: {path.name}")
            return []
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        print(f"Unsupported file type: {path.suffix}")
        return []

    chunks = text_splitter.split_documents(loader.load())
    return [c for c in chunks if _apply_chunk_metadata(c, collection, path.name)]


def _personal_note_chunks(path: Path) -> list[Document]:
    """personal_notes.txt: one chunk per note, header included (it tells Chatette when the note
    was written, or that it is a lasting fact). A note longer than CHUNK_SIZE is split, and every
    piece repeats the header."""
    from note_manager import parse_personal_notes, note_header

    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    preamble, notes = parse_personal_notes(path.read_text(encoding="utf-8"))
    chunks = []
    if preamble:                     # old-format text without headers, if any is left
        chunks += text_splitter.split_documents([Document(page_content=preamble, metadata={})])
    for note in notes:
        header = note_header(note["created"], note["timeless"])
        room = max(CHUNK_SIZE - len(header) - 1, 100)
        parts = RecursiveCharacterTextSplitter(
            chunk_size=room, chunk_overlap=min(CHUNK_OVERLAP, room // 4)
        ).split_text(note["text"]) or [""]
        for part in parts:
            chunks.append(Document(page_content=f"{header} {part}".strip(), metadata={}))
    return [c for c in chunks if _apply_chunk_metadata(c, "notes", path.name)]


def ingest_file(filepath: str, collection: str = "documents"):
    """Ingest (or re-sync) a single file into the vector store."""
    path = Path(filepath)
    _sync_chunks(collection, _file_chunks(path, collection), source_file=path.name)


# ==========================
# Documents ingestion
# ==========================
def ingest_all_documents():
    """Sync all files in the documents folder."""
    if not DOCUMENTS_PATH:
        print("DOCUMENTS_PATH not set in .env — skipping documents.")
        return
    docs_path = Path(DOCUMENTS_PATH)

    if not docs_path.exists():
        print("Documents folder not found.")
        return

    chunks = []
    for file in docs_path.iterdir():
        if file.is_file():
            chunks += _file_chunks(file, "documents")
    _sync_chunks("documents", chunks)


# ==========================
# Notes ingestion
# ==========================
def ingest_notes():
    """Sync all notes (reminders, personal notes, drafts)."""
    notes_path = Path(DOCS_PATH)
    if not notes_path.exists():
        print("Docs folder not found.")
        return

    chunks = []
    for file in notes_path.iterdir():
        if file.is_file() and file.suffix in ('.txt', '.md'):
            if file.name.startswith("journal_offline_edit_"):
                continue   # sync conflict backups — kept for the user to review, not knowledge
            if file.name == "personal_notes.txt":
                chunks += _personal_note_chunks(file)
            else:
                chunks += _file_chunks(file, "notes")

    drafts_path = notes_path / "drafts"
    if drafts_path.exists():
        for file in drafts_path.iterdir():
            if file.is_file():
                chunks += _file_chunks(file, "notes")

    _sync_chunks("notes", chunks)


# ==========================
# Lists ingestion
# ==========================
def ingest_lists():
    """Sync all list files."""
    lists_path = Path(DOCS_PATH) / "lists"
    if not lists_path.exists():
        print("Lists folder not found — skipping.")
        return

    chunks = []
    for file in lists_path.iterdir():
        if file.is_file() and file.suffix == ".txt":
            chunks += _file_chunks(file, "lists")
    _sync_chunks("lists", chunks)


# ==========================
# Calendar ingestion
# ==========================
def ingest_calendar_events():
    """Fetch and ingest upcoming calendar events."""

    # Fetch first: if Google is unreachable this raises and the old chunks stay in place.
    events = get_upcoming_events(
        days_ahead=CALENDAR_DAYS_AHEAD,
        days_behind=CALENDAR_DAYS_BEHIND
    )

    if not events:
        print("No calendar events found.")

    documents = []

    for event in events:
        event_dict = event.model_dump() if hasattr(event, "model_dump") else event
        start = human_datetime(event_dict.get("start"))
        end = human_datetime(event_dict.get("end"))

        description = (event_dict.get('description') or '').strip()
        text = f"""
[Calendar Event]

Title: {event_dict.get('title')}
Start: {start}
End: {end}
""" + (f"Description: {description}\n" if description else "")

        doc = Document(
            page_content=text.strip(),
            metadata={}
        )
        if _apply_chunk_metadata(doc, "calendar", "google_calendar"):
            documents.append(doc)

    _sync_chunks("calendar", text_splitter.split_documents(documents))


# ==========================
# Email ingestion
# ==========================
def ingest_emails():
    """Fetch and ingest recent emails."""

    # Fetch first: if Gmail is unreachable this raises and the old chunks stay in place.
    emails = get_recent_emails(days_window=EMAIL_DAYS_WINDOW)

    if not emails:
        print("No emails found.")

    chunks = []

    for email in emails:
        date = human_datetime(email.date)
        header = f"[Email]\nFrom: {email.from_}\nDate: {date}\nSubject: {email.subject}"
        body = _strip_html(email.body or "").strip()[:400]

        # Split only the body and repeat the header in every chunk, so each chunk still
        # says who sent it and what it is about (previously the header and the body
        # landed in separate chunks). Every chunk stays within CHUNK_SIZE.
        body_room = max(CHUNK_SIZE - len(header) - len("\nBody: "), 100)
        body_parts = RecursiveCharacterTextSplitter(
            chunk_size=body_room, chunk_overlap=min(CHUNK_OVERLAP, body_room // 4)
        ).split_text(body) if body else []

        for part in body_parts or [""]:
            text = f"{header}\nBody: {part}" if part else header
            doc = Document(page_content=text, metadata={})
            if _apply_chunk_metadata(doc, "emails", "gmail"):
                chunks.append(doc)

    _sync_chunks("emails", chunks)


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