"""youtube_search.py — YouTube Data API v3 search wrapper."""

import os

import requests

_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

_quota_exceeded = False


def is_available() -> bool:
    return bool(os.getenv("YOUTUBE_API_KEY", "")) and not _quota_exceeded


def search_videos(query: str, max_results: int = 5, page_token: str = "") -> dict | None:
    """Search YouTube. Returns {results: [...], next_page_token: str} or None on error."""
    global _quota_exceeded
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        print("[YouTubeSearch] YOUTUBE_API_KEY not set.")
        return None
    if _quota_exceeded:
        print("[YouTubeSearch] Quota exceeded — search disabled.")
        return None
    try:
        params = {
            "part":       "snippet",
            "q":          query,
            "maxResults": max_results,
            "type":       "video",
            "key":        api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(_SEARCH_URL, params=params, timeout=10)
        if resp.status_code == 403:
            data = resp.json()
            errors = data.get("error", {}).get("errors", [])
            if any(e.get("reason") in ("quotaExceeded", "dailyLimitExceeded") for e in errors):
                _quota_exceeded = True
                print("[YouTubeSearch] Daily quota exceeded — YouTube search disabled until restart.")
                return None
        resp.raise_for_status()
        body = resp.json()
        results = []
        for item in body.get("items", []):
            video_id  = item["id"]["videoId"]
            snippet   = item["snippet"]
            title     = snippet["title"]
            thumbnail = snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
            results.append({"video_id": video_id, "title": title, "thumbnail": thumbnail})
        return {"results": results, "next_page_token": body.get("nextPageToken", "")}
    except Exception as e:
        print(f"[YouTubeSearch] Error: {e}")
        return None
