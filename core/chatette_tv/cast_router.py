"""cast_router.py — FastAPI router for all /cast endpoints."""

from fastapi import APIRouter

from . import cast_manager, channel_registry, youtube_search
from .models import (
    CastResponse, ChannelRequest, PowerRequest,
    StatusResponse, VolumeRequest,
    YouTubeSearchRequest, YouTubeCastRequest, YouTubeSearchResponse, YouTubeResult,
)

router = APIRouter()

_CHANNEL_DISPLAY = {
    "ard":      "ARD",
    "zdf":      "ZDF",
    "euronews": "Euronews",
    "arte_fr":  "Arte (FR)",
    "arte_de":  "Arte (DE)",
    "milenio":  "Milenio",
    "tv5monde": "TV5 Monde",
    "nhk":      "NHK World",
}

# Euronews YouTube live fallback (English) — used if HLS stream is down
_EURONEWS_YT_ID = "9Auq9mYxFEE"


@router.post("/tv/power", response_model=CastResponse)
def tv_power(req: PowerRequest):
    ok = cast_manager.power_on() if req.action == "on" else cast_manager.power_off()
    msg = "TV turned on." if req.action == "on" else "TV turned off."
    return CastResponse(ok=ok, message=msg if ok else "Chromecast not reachable.")


@router.post("/tv/volume", response_model=CastResponse)
def tv_volume(req: VolumeRequest):
    if req.level is not None:
        new_level = cast_manager.set_volume(req.level)
    elif req.delta is not None:
        new_level = cast_manager.volume_delta(req.delta)
    else:
        return CastResponse(ok=False, message="Provide level or delta.")
    if new_level < 0:
        return CastResponse(ok=False, message="Chromecast not reachable.")
    return CastResponse(ok=True, message=f"Volume set to {new_level}%.")


@router.post("/youtube/search", response_model=YouTubeSearchResponse)
def youtube_search_endpoint(req: YouTubeSearchRequest):
    data = youtube_search.search_videos(req.query, page_token=req.page_token)
    if data is None:
        return YouTubeSearchResponse(results=[])
    return YouTubeSearchResponse(
        results=[YouTubeResult(**r) for r in data["results"]],
        next_page_token=data["next_page_token"],
    )


@router.post("/youtube", response_model=CastResponse)
def cast_youtube(req: YouTubeCastRequest):
    ok = cast_manager.play_youtube(req.video_id)
    title = req.title or req.video_id
    return CastResponse(ok=ok, message=f"Casting '{title}'." if ok else "Chromecast not reachable.")


@router.post("/channel", response_model=CastResponse)
def cast_channel(req: ChannelRequest):
    print(f"[CastRouter] Channel request: {req.channel}")
    url = channel_registry.get_url(req.channel)
    if not url:
        print(f"[CastRouter] No URL for {req.channel}")
        return CastResponse(ok=False, message=f"No stream URL for {req.channel}.")
    name = _CHANNEL_DISPLAY.get(req.channel, req.channel)
    print(f"[CastRouter] Playing {name} → {url[:60]}…")
    ok = cast_manager.play_hls(url, title=name)
    if not ok and req.channel == "euronews":
        print("[CastRouter] Euronews HLS failed — trying YouTube fallback")
        ok = cast_manager.play_youtube(_EURONEWS_YT_ID)
        if ok:
            return CastResponse(ok=True, message=f"Casting Euronews (YouTube).")
    return CastResponse(ok=ok, message=f"Casting {name}." if ok else "Chromecast not reachable.")


@router.post("/stop", response_model=CastResponse)
def cast_stop():
    ok = cast_manager.stop()
    return CastResponse(ok=ok, message="Playback stopped." if ok else "Chromecast not reachable.")


@router.post("/play", response_model=CastResponse)
def cast_play():
    ok = cast_manager.resume()
    return CastResponse(ok=ok, message="Resumed." if ok else "Chromecast not reachable.")


@router.get("/status", response_model=StatusResponse)
def cast_status():
    return StatusResponse(**cast_manager.get_status())
