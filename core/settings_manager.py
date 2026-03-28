import os
import re
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

# Path to .env file — one level up from core/
ENV_PATH = Path(__file__).parent.parent / ".env"


# ==========================
# Pydantic model
# ==========================

class ChatetteSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    server_url: str
    model_selection: str
    use_groq: bool
    groq_model: str
    email_days_window: int
    calendar_days_ahead: int
    calendar_days_behind: int

    @field_validator("email_days_window", "calendar_days_ahead", "calendar_days_behind")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Day window values must be non-negative")
        return v

    @field_validator("model_selection", "groq_model")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name must not be empty")
        return v.strip()


# ==========================
# Read / Write
# ==========================

def read_settings() -> ChatetteSettings:
    """Read current settings from .env and return a validated ChatetteSettings."""
    load_dotenv(ENV_PATH, override=True)
    use_groq = os.getenv("USE_GROQ", "false").lower() == "true"
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    model_selection = groq_model if use_groq else "local"

    try:
        return ChatetteSettings(
            server_url=f"http://{_get_local_ip()}:8000",
            model_selection=model_selection,
            use_groq=use_groq,
            groq_model=groq_model,
            email_days_window=int(os.getenv("EMAIL_DAYS_WINDOW", "14")),
            calendar_days_ahead=int(os.getenv("CALENDAR_DAYS_AHEAD", "14")),
            calendar_days_behind=int(os.getenv("CALENDAR_DAYS_BEHIND", "1")),
        )
    except ValidationError as e:
        print(f"⚠️ Settings validation error: {e}")
        raise


def write_settings(settings: ChatetteSettings) -> bool:
    """Write a validated ChatetteSettings to .env file."""
    try:
        if not ENV_PATH.exists():
            print(f"❌ .env file not found at {ENV_PATH}")
            return False

        with open(ENV_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        if settings.model_selection == "local":
            content = _set_env_value(content, "USE_GROQ", "false")
        else:
            content = _set_env_value(content, "USE_GROQ", "true")
            content = _set_env_value(content, "GROQ_MODEL", settings.model_selection)

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
    """Replace or append a key=value in .env content."""
    pattern = rf"^{re.escape(key)}=.*$"
    replacement = f"{key}={value}"
    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        return content.rstrip() + f"\n{replacement}\n"


def _get_local_ip() -> str:
    """Get the local IP address of the PC."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.0.4"