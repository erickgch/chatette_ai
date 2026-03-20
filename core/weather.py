import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ─── Home location from .env ───────────────────────────
HOME_CITY = os.getenv("HOME_CITY", "")
HOME_LAT  = os.getenv("HOME_LAT", "")
HOME_LON  = os.getenv("HOME_LON", "")

# ─── Open-Meteo endpoints ──────────────────────────────
DWD_API_URL       = "https://api.open-meteo.com/v1/dwd-icon"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"

# ─── WMO weather code descriptions ────────────────────
WMO_CODES = {
    0:  "clear sky",
    1:  "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "light showers", 81: "moderate showers", 82: "heavy showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "heavy thunderstorm with hail",
}


def _wmo(code: int) -> str:
    return WMO_CODES.get(int(code), f"weather code {code}")


def _clean_city_input(city: str) -> str:
    """
    Clean up free-form city input before geocoding.
    Removes filler words, commas, extra spaces.
    """
    import re
    # Remove leading prepositions the user might type naturally
    city = re.sub(r"^(in|for|at|near|around|über|für|für das wetter in)\s+",
                  "", city.strip(), flags=re.IGNORECASE)
    # Remove commas (e.g. "London, Canada" → "London Canada")
    city = city.replace(",", " ")
    # Collapse multiple spaces
    city = re.sub(r"\s+", " ", city).strip()
    return city


def geocode(city: str) -> tuple[float, float, str]:
    """
    Convert a city name to (lat, lon, resolved_name).
    Handles natural language input like "London, Canada" or "in Berlin".
    Raises ValueError if city not found.
    """
    city = _clean_city_input(city)
    if not city:
        raise ValueError("No city name provided")

    # Split into city + country hint if user typed both
    # e.g. "London Canada" → search "London" with country filter
    parts = city.split()
    search_name = city  # try full string first

    try:
        resp = requests.get(
            GEOCODING_API_URL,
            params={"name": search_name, "count": 5, "language": "en", "format": "json"},
            timeout=5
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        # If no results and input has multiple words, try just the first word
        # with the remainder as a country/region hint
        if not results and len(parts) > 1:
            resp2 = requests.get(
                GEOCODING_API_URL,
                params={"name": parts[0], "count": 10, "language": "en", "format": "json"},
                timeout=5
            )
            resp2.raise_for_status()
            all_results = resp2.json().get("results", [])
            # Filter by country/region hint from remaining words
            hint = " ".join(parts[1:]).lower()
            results = [
                r for r in all_results
                if hint in r.get("country", "").lower()
                or hint in r.get("country_code", "").lower()
                or hint in r.get("admin1", "").lower()
            ]
            if not results:
                results = all_results  # fallback: take best match ignoring hint

        if not results:
            raise ValueError(f"City '{city}' not found")

        r = results[0]
        # Build a descriptive name: City, Region, Country
        name_parts = [r.get("name", city)]
        if r.get("admin1"):
            name_parts.append(r["admin1"])
        if r.get("country"):
            name_parts.append(r["country"])
        name = ", ".join(name_parts)
        return float(r["latitude"]), float(r["longitude"]), name

    except requests.RequestException as e:
        raise ValueError(f"Geocoding failed: {e}")


def _resolve_location(city: str) -> tuple[float, float, str]:
    """
    Resolve city to coordinates.
    Uses home location if city is empty/home reference.
    """
    home_refs = {"", "here", "home", "my location", "current location",
                 "da", "hier", "zuhause", "aquí", "mi ubicación"}

    if city.lower().strip() in home_refs:
        if HOME_LAT and HOME_LON and HOME_CITY:
            return float(HOME_LAT), float(HOME_LON), HOME_CITY
        raise ValueError(
            "Home location not set. Add HOME_CITY, HOME_LAT and HOME_LON to your .env"
        )

    return geocode(city)


def get_current_weather(city: str = "") -> dict:
    """
    Fetch current weather conditions.
    Returns a dict with current conditions and location name.
    """
    lat, lon, location_name = _resolve_location(city)

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "current": [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": "Europe/Berlin",
        "models": "icon_seamless",
    }

    resp = requests.get(DWD_API_URL, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    current = data.get("current", {})
    cw = data.get("current_weather", {})

    return {
        "location": location_name,
        "time": current.get("time", cw.get("time", "")),
        "temperature": current.get("temperature_2m", cw.get("temperature")),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "precipitation": current.get("precipitation", 0),
        "condition": _wmo(current.get("weather_code", cw.get("weathercode", 0))),
        "wind_speed": current.get("wind_speed_10m", cw.get("windspeed")),
    }


def get_today_forecast(city: str = "") -> dict:
    """
    Fetch today's hourly forecast (next 24 hours).
    Returns summary with highs, lows, precipitation and conditions.
    """
    lat, lon, location_name = _resolve_location(city)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
        ],
        "forecast_days": 2,
        "timezone": "Europe/Berlin",
        "models": "icon_seamless",
    }

    resp = requests.get(DWD_API_URL, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip_prob = hourly.get("precipitation_probability", [])
    precip = hourly.get("precipitation", [])
    codes = hourly.get("weather_code", [])
    winds = hourly.get("wind_speed_10m", [])

    # Filter to today's hours
    today = datetime.now().strftime("%Y-%m-%d")
    today_indices = [i for i, t in enumerate(times) if t.startswith(today)]

    if not today_indices:
        return {"location": location_name, "error": "No forecast data for today"}

    today_temps   = [temps[i] for i in today_indices if i < len(temps)]
    today_probs   = [precip_prob[i] for i in today_indices if i < len(precip_prob)]
    today_precip  = [precip[i] for i in today_indices if i < len(precip)]
    today_codes   = [codes[i] for i in today_indices if i < len(codes)]
    today_winds   = [winds[i] for i in today_indices if i < len(winds)]

    # Build hourly summary (morning/afternoon/evening)
    hour_now = datetime.now().hour
    segments = {
        "morning":   [i for i in today_indices if 6  <= int(times[i][11:13]) < 12],
        "afternoon": [i for i in today_indices if 12 <= int(times[i][11:13]) < 18],
        "evening":   [i for i in today_indices if 18 <= int(times[i][11:13]) < 24],
    }

    def _seg_summary(indices):
        if not indices:
            return None
        t = [temps[i] for i in indices if i < len(temps)]
        p = [precip_prob[i] for i in indices if i < len(precip_prob)]
        c = [codes[i] for i in indices if i < len(codes)]
        w = [winds[i] for i in indices if i < len(winds)]
        dominant_code = max(set(c), key=c.count) if c else 0
        return {
            "temp_min": round(min(t), 1) if t else None,
            "temp_max": round(max(t), 1) if t else None,
            "condition": _wmo(dominant_code),
            "precip_prob_max": round(max(p), 0) if p else 0,
            "wind_max": round(max(w), 1) if w else 0,
        }

    return {
        "location": location_name,
        "date": today,
        "temp_min": round(min(today_temps), 1) if today_temps else None,
        "temp_max": round(max(today_temps), 1) if today_temps else None,
        "total_precipitation": round(sum(today_precip), 1),
        "max_precip_probability": round(max(today_probs), 0) if today_probs else 0,
        "dominant_condition": _wmo(max(set(today_codes), key=today_codes.count)),
        "max_wind": round(max(today_winds), 1) if today_winds else 0,
        "morning":   _seg_summary(segments["morning"]),
        "afternoon": _seg_summary(segments["afternoon"]),
        "evening":   _seg_summary(segments["evening"]),
    }


def get_weekly_forecast(city: str = "") -> list[dict]:
    """
    Fetch 7-day daily forecast.
    Returns list of daily summaries.
    """
    lat, lon, location_name = _resolve_location(city)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "sunrise",
            "sunset",
        ],
        "forecast_days": 7,
        "timezone": "Europe/Berlin",
        "models": "icon_seamless",
    }

    resp = requests.get(DWD_API_URL, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    dates     = daily.get("time", [])
    codes     = daily.get("weather_code", [])
    t_max     = daily.get("temperature_2m_max", [])
    t_min     = daily.get("temperature_2m_min", [])
    precip    = daily.get("precipitation_sum", [])
    precip_p  = daily.get("precipitation_probability_max", [])
    wind      = daily.get("wind_speed_10m_max", [])
    sunrise   = daily.get("sunrise", [])
    sunset    = daily.get("sunset", [])

    result = []
    for i, date in enumerate(dates):
        # Parse day name
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            day_name = dt.strftime("%A")
        except Exception:
            day_name = date

        result.append({
            "location": location_name,
            "date": date,
            "day": day_name,
            "condition": _wmo(codes[i]) if i < len(codes) else "unknown",
            "temp_max": round(t_max[i], 1) if i < len(t_max) else None,
            "temp_min": round(t_min[i], 1) if i < len(t_min) else None,
            "precipitation": round(precip[i], 1) if i < len(precip) else 0,
            "precip_probability": round(precip_p[i], 0) if i < len(precip_p) else 0,
            "wind_max": round(wind[i], 1) if i < len(wind) else 0,
            "sunrise": sunrise[i][11:16] if i < len(sunrise) else "",
            "sunset": sunset[i][11:16] if i < len(sunset) else "",
        })

    return result


def format_weather_context(weather_data: dict | list, timeframe: str) -> str:
    """
    Convert weather data into a concise text context for the LLM.
    """
    if isinstance(weather_data, list):
        # Weekly forecast
        lines = [f"7-day forecast for {weather_data[0]['location']}:"]
        for day in weather_data:
            precip_note = ""
            if day['precip_probability'] >= 40:
                precip_note = f", {day['precip_probability']}% chance of rain ({day['precipitation']}mm)"
            lines.append(
                f"  {day['day']} {day['date']}: {day['condition']}, "
                f"{day['temp_min']}–{day['temp_max']}°C"
                f"{precip_note}, wind up to {day['wind_max']} km/h"
            )
        return "\n".join(lines)

    elif timeframe in ("now", "current"):
        d = weather_data
        parts = [
            f"Current weather in {d['location']}:",
            f"  Condition: {d['condition']}",
            f"  Temperature: {d['temperature']}°C (feels like {d['feels_like']}°C)" if d.get('feels_like') else f"  Temperature: {d['temperature']}°C",
            f"  Humidity: {d['humidity']}%" if d.get('humidity') else "",
            f"  Wind: {d['wind_speed']} km/h" if d.get('wind_speed') else "",
            f"  Precipitation: {d['precipitation']}mm" if d.get('precipitation') else "",
        ]
        return "\n".join(p for p in parts if p)

    else:
        # Today forecast
        d = weather_data
        lines = [f"Today's forecast for {d['location']} ({d['date']}):"]
        lines.append(f"  Overall: {d['dominant_condition']}, {d['temp_min']}–{d['temp_max']}°C")
        if d['total_precipitation'] > 0:
            lines.append(f"  Rain: {d['total_precipitation']}mm total, up to {d['max_precip_probability']}% probability")
        lines.append(f"  Wind: up to {d['max_wind']} km/h")

        for seg_name in ("morning", "afternoon", "evening"):
            seg = d.get(seg_name)
            if seg:
                rain = f", {seg['precip_prob_max']}% rain" if seg['precip_prob_max'] >= 30 else ""
                lines.append(
                    f"  {seg_name.capitalize()}: {seg['condition']}, "
                    f"{seg['temp_min']}–{seg['temp_max']}°C{rain}"
                )

        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("=== Current weather ===")
    current = get_current_weather()
    print(format_weather_context(current, "now"))

    print("\n=== Today's forecast ===")
    today = get_today_forecast()
    print(format_weather_context(today, "today"))

    print("\n=== Weekly forecast ===")
    weekly = get_weekly_forecast()
    print(format_weather_context(weekly, "week"))