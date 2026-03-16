#!/usr/bin/env python3
"""
Weather Intelligence Dashboard - local server with Polymarket-backed data.
Scrapes the public weather predictions page and exposes a small local API.
"""
import http.server
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

PORT = 8090
BIND = "127.0.0.1"
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache
_cache = {"data": None, "ts": 0, "ttl": 300}


def fold_text(value):
    """Lowercase and strip accents for tolerant city matching."""
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


# Canonical Polymarket city keys with accepted aliases.
CITY_ALIASES = {
    "nyc": ["new york", "ny", "jfk", "new york (jfk)", "nova york", "nova iorque"],
    "london": ["london", "heathrow", "london (heathrow)", "londres"],
    "tokyo": ["tokyo", "haneda", "tokyo (haneda)", "toquio", "tokio"],
    "seoul": ["seoul", "seul"],
    "beijing": ["beijing", "pequim"],
    "ankara": ["ankara", "ancara"],
    "milan": ["milan", "milao"],
    "munich": ["munich", "munique"],
    "moscow": ["moscow", "moscou"],
    "sao paulo": ["sao paulo", "sao-paulo", "sp"],
    "wellington": ["wellington"],
    "toronto": ["toronto"],
    "chicago": ["chicago"],
    "dallas": ["dallas"],
    "seattle": ["seattle"],
    "miami": ["miami"],
    "atlanta": ["atlanta"],
    "buenos aires": ["buenos aires"],
}

FOLDED_CITY_ALIASES = {
    city_key: {fold_text(city_key), *(fold_text(alias) for alias in aliases)}
    for city_key, aliases in CITY_ALIASES.items()
}

MONTH_DAY_PATTERN = re.compile(r"^[A-Za-z]+\s+\d{1,2}$")


def parse_event_title(title):
    """Extract canonical question, city, and date fields from a market title."""
    question = (title or "").strip()
    city = ""
    date_label = ""

    match = re.search(r"\bin\s+(.+?)\s+on\s+(.+?)[\?]?$", question, re.IGNORECASE)
    if match:
        city = match.group(1).strip()
        date_label = match.group(2).strip()

    return {"question": question, "city": city, "date_label": date_label}


def normalize_city_query(query):
    """Map a raw city string to a canonical Polymarket city key when possible."""
    folded_query = fold_text(query)
    if not folded_query:
        return ""

    for city_key, aliases in FOLDED_CITY_ALIASES.items():
        if folded_query in aliases:
            return city_key

    for city_key, aliases in FOLDED_CITY_ALIASES.items():
        for alias in aliases:
            if alias in folded_query or folded_query in alias:
                return city_key

    return folded_query


def classify_event_interpretation(question, city, city_key, date_label):
    """Return explicit parsing status and confidence for the current baseline."""
    parse_notes = []
    has_city = bool(city)
    has_date = bool(date_label)
    canonical_city = city_key in CITY_ALIASES if city_key else False
    supported_date = bool(MONTH_DAY_PATTERN.match(date_label or ""))

    if has_city:
        if canonical_city:
            parse_notes.append("city_matched_by_alias")
        else:
            parse_notes.append("city_not_canonical")
    else:
        parse_notes.append("city_not_extracted")

    if has_date:
        parse_notes.append("date_inferred_from_title_only")
        if not supported_date:
            parse_notes.append("date_not_supported_by_current_baseline")
    else:
        parse_notes.append("date_not_extracted")

    if question and has_city and has_date and canonical_city and supported_date:
        parse_status = "valid"
        rule_confidence = "HIGH"
    elif question and (has_city or has_date):
        parse_status = "partial"
        rule_confidence = "MEDIUM"
    else:
        parse_status = "unknown"
        rule_confidence = "LOW"

    return {
        "parse_status": parse_status,
        "parse_notes": parse_notes,
        "rule_confidence": rule_confidence,
    }


def interpret_market_event(title):
    """Build the baseline market interpretation returned by the backend."""
    meta = parse_event_title(title)
    city = meta["city"]
    city_key = normalize_city_query(city)
    interpretation = classify_event_interpretation(meta["question"], city, city_key, meta["date_label"])

    return {
        "question": meta["question"],
        "city": city,
        "city_key": city_key,
        "date": meta["date_label"],
        **interpretation,
    }


def fetch_polymarket_weather():
    """Fetch weather prediction data from the Polymarket page."""
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < _cache["ttl"]:
        return _cache["data"]

    url = "https://polymarket.com/predictions/weather"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[Polymarket] Fetch error: {exc}", flush=True)
        return _cache.get("data") or []

    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not match:
        print("[Polymarket] No __NEXT_DATA__ found", flush=True)
        return _cache.get("data") or []

    try:
        nd = json.loads(match.group(1))
        pages = nd["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"]["data"]["pages"]
        events_raw = []
        for page in pages:
            events_raw.extend(page.get("results", []))
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"[Polymarket] Parse error: {exc}", flush=True)
        return _cache.get("data") or []

    # Canonical API payload:
    # {
    #   question: str,
    #   city: str,
    #   city_key: str,
    #   date: str,
    #   parse_status: "valid" | "partial" | "unknown",
    #   parse_notes: [str],
    #   rule_confidence: "HIGH" | "MEDIUM" | "LOW",
    #   outcomes: [{ label: str, probability: float }],
    #   volume: str,
    #   liquidity: str
    # }
    result = []
    for ev in events_raw:
        title = ev.get("title", "")
        if "temperature" not in title.lower() and "highest" not in title.lower():
            continue

        interpreted = interpret_market_event(title)

        markets = ev.get("markets", [])
        outcomes = []
        for market in markets:
            label = market.get("groupItemTitle", market.get("question", ""))
            price = market.get("bestBid") or market.get("lastTradePrice") or 0
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = 0
            outcomes.append({"label": label, "probability": price})

        outcomes.sort(key=lambda item: item["probability"], reverse=True)

        result.append(
            {
                **interpreted,
                "outcomes": outcomes,
                "volume": ev.get("volume", "0"),
                "liquidity": ev.get("liquidity", "0"),
            }
        )

    _cache["data"] = result
    _cache["ts"] = now
    print(f"[Polymarket] Cached {len(result)} weather events", flush=True)
    return result


def match_city_events(events, query):
    """Match events by city using the canonical city key when possible."""
    city_key = normalize_city_query(query)
    if not city_key:
        return []

    matched = [event for event in events if event.get("city_key") == city_key]
    if matched:
        return matched

    query_folded = fold_text(query)
    return [event for event in events if query_folded in fold_text(event.get("city", ""))]


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/polymarket":
            self.send_json(fetch_polymarket_weather())
            return

        if parsed.path == "/api/polymarket/city":
            params = urllib.parse.parse_qs(parsed.query)
            city_q = params.get("city", [""])[0]
            if not city_q.strip():
                self.send_json([])
                return
            events = fetch_polymarket_weather()
            self.send_json(match_city_events(events, city_q))
            return

        if parsed.path == "/api/geocode":
            params = urllib.parse.parse_qs(parsed.query)
            q = params.get("q", [""])[0].strip()
            if not q:
                self.send_json([])
                return
            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search"
                f"?name={urllib.parse.quote(q)}&count=8&language=pt&format=json"
            )
            try:
                with urllib.request.urlopen(geo_url, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    self.send_json({"results": data.get("results", [])})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
            return

        super().do_GET()

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "max-age=60")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    bind = sys.argv[1] if len(sys.argv) > 1 else BIND
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    print(f"Dashboard: http://{bind}:{port}/dashboard.html", flush=True)
    server = http.server.HTTPServer((bind, port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
