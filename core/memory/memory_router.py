import threading
from fastapi import APIRouter
from pydantic import BaseModel

from .summarizer import _load_store, _save_store, MemoryStore, retrieve_episodes, run_summarizer

router = APIRouter(prefix="/memory", tags=["memory"])

# Wired by rag_lw.init_memory_router() after import to avoid circular dependency
_get_turn_history = None
_llm_invoke_ref = None


def init_router(turn_history_getter, llm_invoke):
    global _get_turn_history, _llm_invoke_ref
    _get_turn_history = turn_history_getter
    _llm_invoke_ref = llm_invoke


class MemoryResponse(BaseModel):
    facts: list[str]
    episodes: list[str]


@router.get("", response_model=MemoryResponse)
def get_memory(query: str = ""):
    """Return current facts and, if a query is provided, matching episode excerpts."""
    store = _load_store()
    episodes: list[str] = []
    if query:
        raw_block = retrieve_episodes(query)
        for line in raw_block.splitlines():
            line = line.strip()
            if line.startswith("- "):
                episodes.append(line[2:])
    return MemoryResponse(facts=store.facts, episodes=episodes)


@router.post("/update")
def update_memory(device_id: str = "app"):
    """Manually trigger a summariser run for the given device's buffered turns."""
    if _get_turn_history is None or _llm_invoke_ref is None:
        return {"status": "not_initialized"}
    turns = _get_turn_history(device_id)
    if not turns:
        return {"status": "no_turns"}
    threading.Thread(
        target=run_summarizer,
        args=(turns, _llm_invoke_ref),
        daemon=True,
    ).start()
    return {"status": "summarizer_triggered", "turns": len(turns)}


@router.delete("/facts")
def clear_facts():
    """Wipe all stored facts."""
    _save_store(MemoryStore(facts=[]))
    return {"status": "cleared"}
