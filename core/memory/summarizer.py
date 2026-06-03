import json
import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent.parent
load_dotenv(BASE_DIR / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
MEMORY_STORE_PATH = Path(__file__).parent / "memory_store.json"
MEMORY_DB_PATH = BASE_DIR / "data" / "memory_db"

PERSONAL_KEYWORDS = {
    "remember", "last time", "i told you", "did i",
    "you know", "recall", "we talked"
}


class MemoryStore(BaseModel):
    facts: list[str] = []


_episode_vs = None


def _episode_vectorstore():
    """Lazy singleton for the chatette_memory ChromaDB collection."""
    global _episode_vs
    if _episode_vs is None:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        MEMORY_DB_PATH.mkdir(parents=True, exist_ok=True)
        _episode_vs = Chroma(
            collection_name="chatette_memory",
            persist_directory=str(MEMORY_DB_PATH),
            embedding_function=OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=EMBEDDING_MODEL,
            ),
        )
    return _episode_vs


def _load_store() -> MemoryStore:
    if MEMORY_STORE_PATH.exists():
        try:
            return MemoryStore.model_validate_json(
                MEMORY_STORE_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            pass
    return MemoryStore()


def _save_store(store: MemoryStore) -> None:
    MEMORY_STORE_PATH.write_text(store.model_dump_json(indent=2), encoding="utf-8")


def run_summarizer(turns: list[dict], llm_invoke) -> None:
    """Two-pass summarizer: update facts in memory_store.json, then store episode in ChromaDB."""
    if not turns:
        return

    dialogue = "\n".join(
        f"User: {t['question']}\nChatette: {t['answer']}" for t in turns
    )

    # Pass 1 — extract/merge facts into memory_store.json
    existing = _load_store()
    existing_text = "\n".join(f"- {f}" for f in existing.facts) or "None"

    facts_prompt = f"""You are a memory assistant. Extract concise personal facts about the user from this conversation.

Existing facts:
{existing_text}

New conversation:
{dialogue}

Rules:
- Merge and deduplicate with existing facts
- Maximum 7 facts total
- Use soft language: "user mentioned", "user said", "seems like"
- Never use absolute language: "user always", "user is", "user has"
- Only include facts about the user's life, preferences, or environment
- Always cross-check with existing facts and with the reality to avoid contradictions
- Return ONLY a JSON array of strings, no explanation

Output:"""

    try:
        raw = llm_invoke(facts_prompt).strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            new_facts = json.loads(raw[start:end])
            new_facts = [str(f).strip() for f in new_facts if str(f).strip()][:7]
            _save_store(MemoryStore(facts=new_facts))
            print(f"[memory] Facts updated: {len(new_facts)} items")
    except Exception as e:
        print(f"[memory] Facts pass failed: {e}")

    # Pass 2 — write episode summary to ChromaDB
    episode_prompt = f"""Write one concise paragraph summarizing this conversation as a memory episode.
Focus on what the user did, asked, or shared. Be factual and brief (2-3 sentences).

Conversation:
{dialogue}

Summary:"""

    try:
        episode_text = llm_invoke(episode_prompt).strip()
        vs = _episode_vectorstore()
        doc_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vs.add_texts(
            texts=[episode_text],
            metadatas=[{"timestamp": datetime.now().isoformat()}],
            ids=[doc_id],
        )
        print(f"[memory] Episode stored: {doc_id}")
    except Exception as e:
        print(f"[memory] Episode pass failed: {e}")


def get_memory_block() -> str:
    """Return the formatted facts block to append to the system prompt."""
    store = _load_store()
    if not store.facts:
        return ""
    lines = "\n".join(f"- {f}" for f in store.facts)
    return f"\n--- Background (use only if directly relevant) ---\n{lines}\n---"


def retrieve_episodes(query: str, n: int = 2) -> str:
    """Similarity search over chatette_memory; returns formatted context block or empty string."""
    try:
        vs = _episode_vectorstore()
        results = vs.similarity_search(query, k=n)
        if not results:
            return ""
        lines = "\n".join(f"- {doc.page_content}" for doc in results)
        return f"\n--- Relevant past context ---\n{lines}\n---"
    except Exception as e:
        print(f"[memory] Episode retrieval failed: {e}")
        return ""


def is_personal_query(question: str) -> bool:
    """Return True when the question contains personal-reference keywords."""
    q = question.lower()
    return any(kw in q for kw in PERSONAL_KEYWORDS)
