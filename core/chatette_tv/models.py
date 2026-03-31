from typing import Optional, Literal
from pydantic import BaseModel, Field


class PowerRequest(BaseModel):
    action: Literal["on", "off"]


class VolumeRequest(BaseModel):
    level: Optional[int] = Field(None, ge=0, le=100)
    delta: Optional[int] = Field(None, ge=-100, le=100)


class YouTubeSearchRequest(BaseModel):
    query: str

class YouTubeCastRequest(BaseModel):
    video_id: str
    title: str = ""

class YouTubeResult(BaseModel):
    video_id: str
    title: str
    thumbnail: str

class YouTubeSearchResponse(BaseModel):
    results: list[YouTubeResult]


class ChannelRequest(BaseModel):
    channel: Literal["ard", "arte", "france24", "dw", "euronews", "zdf"]


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
