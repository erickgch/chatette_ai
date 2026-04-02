from typing import Optional, Literal
from pydantic import BaseModel, Field


class PowerRequest(BaseModel):
    action: Literal["on", "off"]


class VolumeRequest(BaseModel):
    level: Optional[int] = Field(None, ge=0, le=100)
    delta: Optional[int] = Field(None, ge=-100, le=100)


class YouTubeSearchRequest(BaseModel):
    query: str
    page_token: str = ""

class YouTubeCastRequest(BaseModel):
    video_id: str
    title: str = ""

class YouTubeResult(BaseModel):
    video_id: str
    title: str
    thumbnail: str

class YouTubeSearchResponse(BaseModel):
    results: list[YouTubeResult]
    next_page_token: str = ""


class ChannelRequest(BaseModel):
    channel: Literal["ard", "zdf", "euronews", "arte_fr", "arte_de", "milenio", "tv5monde", "nhk"]


class CastResponse(BaseModel):
    ok: bool
    message: str = ""


class StatusResponse(BaseModel):
    connected: bool
    app: Optional[str] = None
    volume: Optional[int] = None
    muted: Optional[bool] = None
    is_playing: bool = False
    media_title: Optional[str] = None
