import os
import re
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

ENV_PATH = Path(__file__).parent.parent / ".env"

_VALID_MODELS = {"qwen3:8b", "qwen2.5:7b", "gemma3:4b"}


# ==========================
# Pydantic model
# ==========================

class ChatetteSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    server_url: str
    model_selection: str
    email_days_window: int
    calendar_days_ahead: int
    calendar_days_behind: int
    thinking_enabled: bool = False
    # On: Chatette decides per question whether to use personal data (modes ignored).
    auto_rag_selector: bool = False

    @field_validator("email_days_window", "calendar_days_ahead", "calendar_days_behind")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Day window values must be non-negative")
        return v

    @field_validator("model_selection")
    @classmethod
    def must_be_valid_model(cls, v: str) -> str:
        if v.strip() not in _VALID_MODELS:
            raise ValueError(f"model_selection must be one of {_VALID_MODELS}")
        return v.strip()


# ==========================
# Read / Write
# ==========================

def read_settings() -> ChatetteSettings:
    load_dotenv(ENV_PATH, override=True)
    model_selection = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    if model_selection not in _VALID_MODELS:
        model_selection = "qwen3:8b"
    thinking_enabled = os.getenv("THINKING_ENABLED", "false").lower() == "true"
    auto_rag_selector = os.getenv("AUTO_RAG_SELECTOR", "false").lower() == "true"

    try:
        return ChatetteSettings(
            server_url=f"http://{_get_local_ip()}:8000",
            model_selection=model_selection,
            email_days_window=int(os.getenv("EMAIL_DAYS_WINDOW", "14")),
            calendar_days_ahead=int(os.getenv("CALENDAR_DAYS_AHEAD", "14")),
            calendar_days_behind=int(os.getenv("CALENDAR_DAYS_BEHIND", "1")),
            thinking_enabled=thinking_enabled,
            auto_rag_selector=auto_rag_selector,
        )
    except ValidationError as e:
        print(f"⚠️ Settings validation error: {e}")
        raise


def write_settings(settings: ChatetteSettings) -> bool:
    try:
        if not ENV_PATH.exists():
            print(f"❌ .env file not found at {ENV_PATH}")
            return False

        with open(ENV_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        content = _set_env_value(content, "OLLAMA_MODEL", settings.model_selection)
        content = _set_env_value(content, "USE_GROQ", "false")
        content = _set_env_value(
            content, "THINKING_ENABLED", "true" if settings.thinking_enabled else "false"
        )
        content = _set_env_value(
            content, "AUTO_RAG_SELECTOR", "true" if settings.auto_rag_selector else "false"
        )
        content = _set_env_value(
            content, "EMAIL_DAYS_WINDOW", str(settings.email_days_window)
        )
        content = _set_env_value(
            content, "CALENDAR_DAYS_AHEAD", str(settings.calendar_days_ahead)
        )
        content = _set_env_value(
            content, "CALENDAR_DAYS_BEHIND", str(settings.calendar_days_behind)
        )

        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Settings saved to .env")
        return True

    except Exception as e:
        print(f"❌ Failed to write settings: {e}")
        return False


def _set_env_value(content: str, key: str, value: str) -> str:
    pattern = rf"^{re.escape(key)}=.*$"
    replacement = f"{key}={value}"
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        return content.rstrip() + f"\n{replacement}\n"


def _get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.0.4"
