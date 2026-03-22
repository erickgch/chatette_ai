import datetime
from zoneinfo import ZoneInfo


def _local_tz() -> ZoneInfo:
    """Derive local timezone from the system clock at runtime."""
    try:
        import time as _time
        return ZoneInfo(_time.tzname[0])
    except Exception:
        offset = datetime.datetime.now().astimezone().utcoffset()
        return datetime.timezone(offset)


def _parse_and_fix_dt(value: str | None, fallback: datetime.datetime | None = None) -> str:
    """Parse an ISO datetime string; inject local timezone if absent."""
    if not value:
        if fallback is not None:
            return fallback.isoformat()
        raise ValueError("datetime is required but was not provided")
    try:
        dt = datetime.datetime.fromisoformat(value)
    except (ValueError, TypeError):
        raise ValueError(f"Could not parse datetime: '{value}'")
    if dt.tzinfo is None:
        try:
            dt = dt.replace(tzinfo=_local_tz())
        except Exception:
            pass
    return dt.isoformat()
