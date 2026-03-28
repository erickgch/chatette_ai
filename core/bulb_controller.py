import asyncio
import os
import concurrent.futures
from tapo import ApiClient  # type: ignore[import-untyped]


# ── Named colour map (hue, saturation) ───────────────────────────────────────
COLOR_MAP: dict[str, tuple[int, int]] = {
    "red":     (0,   100),
    "orange":  (30,  100),
    "yellow":  (60,  100),
    "green":   (120, 100),
    "teal":    (180, 100),
    "cyan":    (180, 100),
    "blue":    (240, 100),
    "purple":  (270, 100),
    "violet":  (270, 100),
    "pink":    (300, 100),
    "magenta": (300, 100),
}

# ── Named colour-temperature map (Kelvin) ─────────────────────────────────────
COLOR_TEMP_MAP: dict[str, int] = {
    "warm":     2700,
    "candle":   2500,
    "neutral":  4000,
    "daylight": 5000,
    "cool":     6000,
    "cold":     6500,
    "white":    4000,
}


def _env(key: str) -> str | None:
    return os.getenv(key)


def _run(coro):
    """Run an async coroutine safely from sync code, even inside an event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _client():
    email    = _env("TAPO_EMAIL")
    password = _env("TAPO_PASSWORD")
    ip       = _env("TAPO_BULB_IP")
    if not all([email, password, ip]):
        raise RuntimeError("TAPO_EMAIL, TAPO_PASSWORD and TAPO_BULB_IP must be set in .env")
    return await ApiClient(email, password).l535(ip)


# ── Public control functions ──────────────────────────────────────────────────

def turn_on() -> None:
    async def _():
        await (await _client()).on()
    _run(_())


def turn_off() -> None:
    async def _():
        await (await _client()).off()
    _run(_())


def set_brightness(level: int) -> None:
    """level: 1–100"""
    level = max(1, min(100, level))
    async def _():
        await (await _client()).set_brightness(level)
    _run(_())


def set_color_temperature(kelvin: int) -> None:
    """kelvin: 2500–6500"""
    kelvin = max(2500, min(6500, kelvin))
    async def _():
        await (await _client()).set_color_temperature(kelvin)
    _run(_())


def set_color(hue: int, saturation: int) -> None:
    """hue: 0–360, saturation: 0–100"""
    async def _():
        await (await _client()).set_hue_saturation(hue, saturation)
    _run(_())


def get_status() -> dict:
    async def _():
        device = await _client()
        info = await device.get_device_info()
        return info.to_dict()
    return _run(_())
