"""
Nearby place search via OpenStreetMap Overpass API — no API key required.
"""
import math
import requests
import os

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_RADIUS_M = 1500  # 1.5 km default search radius

# Maps common English place type keywords → OSM amenity/shop/tourism tags
_TYPE_MAP = {
    "restaurant":     [('amenity', 'restaurant')],
    "cafe":           [('amenity', 'cafe')],
    "coffee":         [('amenity', 'cafe')],
    "coffee shop":    [('amenity', 'cafe')],
    "bar":            [('amenity', 'bar')],
    "pub":            [('amenity', 'pub')],
    "fast food":      [('amenity', 'fast_food')],
    "pizza":          [('amenity', 'restaurant'), ('amenity', 'fast_food')],
    "bakery":         [('shop', 'bakery')],
    "supermarket":    [('shop', 'supermarket')],
    "grocery":        [('shop', 'supermarket'), ('shop', 'convenience')],
    "pharmacy":       [('amenity', 'pharmacy')],
    "hospital":       [('amenity', 'hospital')],
    "doctor":         [('amenity', 'doctors')],
    "dentist":        [('amenity', 'dentist')],
    "bank":           [('amenity', 'bank')],
    "atm":            [('amenity', 'atm')],
    "hotel":          [('tourism', 'hotel')],
    "hostel":         [('tourism', 'hostel')],
    "museum":         [('tourism', 'museum')],
    "park":           [('leisure', 'park')],
    "gym":            [('leisure', 'fitness_centre')],
    "cinema":         [('amenity', 'cinema')],
    "library":        [('amenity', 'library')],
    "gas station":    [('amenity', 'fuel')],
    "petrol":         [('amenity', 'fuel')],
    "post office":    [('amenity', 'post_office')],
    "hair":           [('shop', 'hairdresser')],
    "hairdresser":    [('shop', 'hairdresser')],
    "barber":         [('shop', 'barber')],
    "laundry":        [('shop', 'laundry'), ('amenity', 'laundry')],
}

# Cuisine tag keywords → OSM cuisine values
_CUISINE_MAP = {
    "chinese":    "chinese",
    "italian":    "italian",
    "japanese":   "japanese",
    "sushi":      "sushi",
    "indian":     "indian",
    "thai":       "thai",
    "vietnamese": "vietnamese",
    "mexican":    "mexican",
    "turkish":    "turkish",
    "greek":      "greek",
    "korean":     "korean",
    "french":     "french",
    "spanish":    "spanish",
    "american":   "american",
    "burger":     "burger",
    "kebab":      "kebab",
    "pizza":      "pizza",
    "asian":      "asian",
    "mediterranean": "mediterranean",
    "arabic":     "arabic",
    "lebanese":   "lebanese",
    "persian":    "persian",
}


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_overpass_query(lat: float, lon: float, radius: int,
                          osm_tags: list[tuple[str, str]],
                          cuisine: str | None) -> str:
    """Build an Overpass QL query for the given tags around a point."""
    parts = []
    for key, value in osm_tags:
        cuisine_filter = f'["cuisine"~"{cuisine}",i]' if cuisine and key == "amenity" and value in ("restaurant", "fast_food") else ""
        parts.append(f'node["{key}"="{value}"]{cuisine_filter}(around:{radius},{lat},{lon});')
        parts.append(f'way["{key}"="{value}"]{cuisine_filter}(around:{radius},{lat},{lon});')
    union = "\n  ".join(parts)
    return f"""[out:json][timeout:10];
(
  {union}
);
out center 20;"""


def search_places(place_type: str, cuisine: str | None = None,
                  radius: int = DEFAULT_RADIUS_M) -> list[dict]:
    """
    Search OSM for places near the home location.
    Returns a list of dicts: {name, address, distance_m, tags}.
    """
    lat = float(os.getenv("HOME_LAT", "50.9029"))
    lon = float(os.getenv("HOME_LON", "6.9384"))

    # Resolve type → OSM tags
    type_key = place_type.lower().strip()
    osm_tags = _TYPE_MAP.get(type_key)
    if not osm_tags:
        # Fallback: treat as amenity=restaurant for unknown food queries
        osm_tags = [('amenity', 'restaurant')]

    query = _build_overpass_query(lat, lon, radius, osm_tags, cuisine)

    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "Chatette/1.0 (personal assistant)"},
            timeout=12,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception as e:
        raise RuntimeError(f"Overpass API error: {e}") from e

    results = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            continue

        # Get coordinates (nodes have lat/lon directly; ways have center)
        if el["type"] == "node":
            elat, elon = el.get("lat", lat), el.get("lon", lon)
        else:
            center = el.get("center", {})
            elat, elon = center.get("lat", lat), center.get("lon", lon)

        dist = _haversine_m(lat, lon, elat, elon)

        # Build address string
        addr_parts = [
            tags.get("addr:street", ""),
            tags.get("addr:housenumber", ""),
        ]
        address = " ".join(p for p in addr_parts if p).strip()
        if not address:
            address = tags.get("addr:full", "")

        results.append({
            "name": name,
            "address": address,
            "distance_m": round(dist),
            "cuisine": tags.get("cuisine", ""),
            "opening_hours": tags.get("opening_hours", ""),
            "website": tags.get("website", ""),
            "phone": tags.get("phone", ""),
        })

    # Sort by distance, return top 5 (handler will pick 3)
    results.sort(key=lambda x: x["distance_m"])
    return results[:5]
