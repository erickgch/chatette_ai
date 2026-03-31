"""channel_registry.py — Resolves live-TV channel names to HLS stream URLs.

Strategy (in priority order for each channel):
  1. Arte: official Arte API (returns a fresh URL every time)
  2. iptv-org M3U: matched by tvg-id + tvg-language (precise, no false positives)
  3. Hardcoded fallbacks (last resort, may become stale)

Refresh: daily background thread for iptv-org feeds.
Arte URL is fetched fresh on every cast request.
"""

import re
import threading

import requests

# ── M3U sources ───────────────────────────────────────────────────────────────
_M3U_URLS = [
    "https://iptv-org.github.io/iptv/countries/de.m3u",    # ARD, ZDF, Arte, DW
    "https://iptv-org.github.io/iptv/categories/news.m3u",  # France24, Euronews, DW
    "https://iptv-org.github.io/iptv/countries/gb.m3u",     # Euronews EN, France24 EN
]

# ── Channel matching rules ────────────────────────────────────────────────────
# Each entry: (tvg-id substring, channel_key, required tvg-language or None)
# First match per key wins across all M3U files.
_RULES: list[tuple[str, str, str | None]] = [
    ("DasErste.de",    "ard",      None),
    ("ZDF.de",         "zdf",      None),
    ("Arte.de",        "arte",     None),
    ("France24En.fr",  "france24", None),        # English-specific tvg-id
    ("France24.fr",    "france24", "English"),   # fallback: language-filtered match
    ("DW.de",          "dw",       "English"),
    ("EuronewsEN.en",  "euronews", None),        # English Euronews (iptv-org id)
    ("Euronews.en",    "euronews", None),         # alternate English id
    ("EuronewsDE.de",  "euronews", None),        # German as fallback
    ("Euronews.de",    "euronews", None),         # last resort
]

# ── Hardcoded fallbacks ───────────────────────────────────────────────────────
_FALLBACKS: dict[str, str] = {
    "ard":      "https://mcdn.daserste.de/daserste/de/master.m3u8",
    "zdf":      "https://zdf-hls-03.akamaized.net/hls/live/2016498/de/high/master.m3u8",
    "arte":     "https://artesimulcast.akamaized.net/hls/live/2031003/artelive_de/index.m3u8",
    "france24": "https://stream.france24.com/hls/live/2037161/F24_EN_HI_HLS/master.m3u8",
    "dw":       "https://dwamdstream104.akamaized.net/hls/live/2015530/dwstream104/index.m3u8",
    "euronews": "https://euronews-euronews-euronews1-live.freecaster.net/live/euronews1/euronews_ENGLISH_HD.m3u8",
}

_registry: dict[str, str] = dict(_FALLBACKS)
_lock = threading.Lock()

_ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')


# ── Arte official API ─────────────────────────────────────────────────────────

def _fetch_arte_url() -> str | None:
    """Call Arte's player API to get a fresh HLS URL for Arte DE live."""
    try:
        resp = requests.get(
            "https://www.arte.tv/api/player/v2/config/de/LIVE",
            timeout=8,
        )
        resp.raise_for_status()
        streams = (
            resp.json()
            .get("data", {})
            .get("attributes", {})
            .get("streams", [])
        )
        for s in streams:
            if s.get("protocol") == "HLS":
                url = s.get("url")
                if url:
                    print(f"[ChannelRegistry] Arte live URL refreshed from API")
                    return url
    except Exception as e:
        print(f"[ChannelRegistry] Arte API failed: {e}")
    return None


# ── M3U parsing ───────────────────────────────────────────────────────────────

def _parse_attrs(extinf_line: str) -> dict[str, str]:
    return {
        k.lower().replace("-", "_"): v
        for k, v in _ATTR_RE.findall(extinf_line)
    }


def _parse_m3u(text: str, found: dict[str, str]) -> None:
    lines = text.splitlines()
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        if line.startswith("#EXTINF"):
            url_line = lines[i + 1].strip()
            if url_line and not url_line.startswith("#"):
                attrs  = _parse_attrs(line)
                tvg_id = attrs.get("tvg_id", "")
                lang   = attrs.get("tvg_language", "")
                for (id_fragment, key, req_lang) in _RULES:
                    if key in found:
                        continue
                    if id_fragment.lower() in tvg_id.lower():
                        if req_lang is None or req_lang.lower() in lang.lower():
                            found[key] = url_line
                            break
        i += 1


def _fetch_and_parse() -> None:
    found: dict[str, str] = {}
    for m3u_url in _M3U_URLS:
        try:
            resp = requests.get(m3u_url, timeout=20)
            resp.raise_for_status()
            _parse_m3u(resp.text, found)
        except Exception as e:
            print(f"[ChannelRegistry] Failed to fetch {m3u_url}: {e}")

    if found:
        with _lock:
            _registry.update(found)
        print(f"[ChannelRegistry] Updated from iptv-org: {list(found)}")
    else:
        print("[ChannelRegistry] No channels found in M3U feeds — keeping fallbacks")


def _schedule() -> None:
    _fetch_and_parse()
    t = threading.Timer(24 * 3600, _schedule)   # refresh daily
    t.daemon = True
    t.start()


# ── Public API ────────────────────────────────────────────────────────────────

def init() -> None:
    """Call once at app startup."""
    threading.Thread(target=_schedule, daemon=True).start()


def get_url(channel: str) -> str | None:
    key = channel.lower().replace(" ", "").replace("-", "")

    # Arte: always fetch a fresh URL from the official API first
    if key == "arte":
        live = _fetch_arte_url()
        if live:
            return live

    with _lock:
        return _registry.get(key) or _FALLBACKS.get(key)


def probe_url(url: str, timeout: int = 6) -> bool:
    """Return True if the URL responds (2xx/3xx). HEAD may be blocked — try GET too."""
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code < 400:
            return True
        # Some CDNs reject HEAD — try a minimal GET
        r = requests.get(url, timeout=timeout, stream=True)
        r.close()
        return r.status_code < 400
    except Exception:
        return False


def invalidate(channel: str) -> None:
    key = channel.lower().replace(" ", "").replace("-", "")
    with _lock:
        _registry.pop(key, None)
    print(f"[ChannelRegistry] Invalidated '{key}'")
