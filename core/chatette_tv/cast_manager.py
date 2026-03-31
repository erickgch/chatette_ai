"""
cast_manager.py — pychromecast singleton for Chatette TV.

Fully compatible with pychromecast v14.
Uses static IP 192.168.0.7 for reliable connection.

Features:
  - Power on/off via Backdrop (HDMI-CEC)
  - YouTube playback
  - HLS / media playback
  - Volume control
  - Thread-safe singleton
  - Debug logging
"""

import socket
import threading

_lock = threading.RLock()
_cast = None
_yt_controller = None

_CHROMECAST_IP = "192.168.0.7"
_WAKEUP_APP_ID = "CC1AD845"   # Default Media Receiver — triggers CEC Active Source

try:
    import pychromecast
    from pychromecast.controllers.youtube import YouTubeController
    _PYCHROMECAST_AVAILABLE = True
except ImportError:
    _PYCHROMECAST_AVAILABLE = False


def _connect_via_castinfo():
    """Fallback: build a CastInfo directly and bypass mDNS entirely."""
    global _cast, _yt_controller
    try:
        from pychromecast.models import CastInfo, HostServiceInfo
        cast_info = CastInfo(
            services=frozenset([HostServiceInfo(host=_CHROMECAST_IP, port=8009)]),
            uuid=None,
            model_name="Chromecast",
            friendly_name="Chromecast",
            host=_CHROMECAST_IP,
            port=8009,
            cast_type="cast",
            manufacturer="Google Inc.",
        )
        cast = pychromecast.Chromecast(cast_info=cast_info)
        cast.wait(timeout=10)
        yt = YouTubeController()
        cast.register_handler(yt)
        _cast = cast
        _yt_controller = yt
        print(f"[CastManager] Connected via CastInfo to '{cast.name}'")
        return cast
    except Exception as e:
        print(f"[CastManager] CastInfo fallback failed: {e}")
        _cast = None
        return None


def _connect():
    """Connect to the Chromecast at the static IP."""
    global _cast
    with _lock:
        if _cast is not None:
            return _cast

        if not _PYCHROMECAST_AVAILABLE:
            print("[CastManager] pychromecast not installed.")
            return None

        print(f"[CastManager] Connecting to Chromecast at {_CHROMECAST_IP} ...")
        try:
            with socket.create_connection((_CHROMECAST_IP, 8009), timeout=5):
                pass
        except Exception as sock_err:
            print(f"[CastManager] Port 8009 NOT reachable: {sock_err}")
            return None

        return _connect_via_castinfo()


def ensure_connected():
    """Return the Chromecast singleton, reconnecting if needed."""
    global _cast
    with _lock:
        if _cast:
            try:
                _ = _cast.status  # raises if socket is dead
                return _cast
            except Exception:
                _cast = None
        return _connect()


# ── Public API ────────────────────────────────────────────────────────────────

def power_on() -> bool:
    cast = ensure_connected()
    if not cast:
        return False
    try:
        cast.start_app(_WAKEUP_APP_ID)
        return True
    except Exception as e:
        print(f"[CastManager] power_on error: {e}")
        return False


def power_off() -> bool:
    cast = ensure_connected()
    if not cast:
        return False
    try:
        cast.quit_app()
        return True
    except Exception as e:
        print(f"[CastManager] power_off error: {e}")
        return False


def set_volume(level: int) -> int:
    cast = ensure_connected()
    if not cast:
        return -1
    try:
        clamped = max(0, min(100, level))
        cast.set_volume(clamped / 100.0)
        return clamped
    except Exception as e:
        print(f"[CastManager] set_volume error: {e}")
        return -1


def volume_delta(delta: int) -> int:
    cast = ensure_connected()
    if not cast:
        return -1
    try:
        current = int((cast.status.volume_level or 0.5) * 100)
        return set_volume(current + delta)
    except Exception as e:
        print(f"[CastManager] volume_delta error: {e}")
        return -1


def play_hls(url: str, title: str = "") -> bool:
    cast = ensure_connected()
    if not cast:
        return False
    try:
        mc = cast.media_controller
        mc.play_media(url, "application/x-mpegURL", title=title)
        mc.block_until_active(timeout=10)
        return True
    except Exception as e:
        print(f"[CastManager] play_hls error: {e}")
        return False


def play_youtube(video_id: str) -> bool:
    global _yt_controller
    cast = ensure_connected()
    if not cast or _yt_controller is None:
        return False
    try:
        _yt_controller.play_video(video_id)
        return True
    except Exception as e:
        print(f"[CastManager] play_youtube error: {e}")
        return False


def stop() -> bool:
    cast = ensure_connected()
    if not cast:
        return False
    try:
        cast.media_controller.stop()
        return True
    except Exception as e:
        print(f"[CastManager] stop error: {e}")
        return False


def get_status() -> dict:
    cast = ensure_connected()
    if not cast:
        return {"connected": False, "app": None, "volume": None,
                "muted": None, "is_playing": False, "media_title": None}
    try:
        cs = cast.status
        ms = cast.media_controller.status
        return {
            "connected":   True,
            "app":         cs.display_name if cs else None,
            "volume":      int((cs.volume_level or 0) * 100) if cs else None,
            "muted":       cs.volume_muted if cs else None,
            "is_playing":  ms.player_is_playing if ms else False,
            "media_title": ms.title if ms else None,
        }
    except Exception as e:
        print(f"[CastManager] get_status error: {e}")
        return {"connected": False, "app": None, "volume": None,
                "muted": None, "is_playing": False, "media_title": None}