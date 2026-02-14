#!/usr/bin/env python3
"""
Weather Intelligence Dashboard — Server com dados Polymarket.
Scrape da página de previsões do Polymarket (dados embutidos no __NEXT_DATA__).
Cache de 5 minutos. Sem API key necessária.
"""
import http.server
import urllib.request
import urllib.parse
import json
import os
import sys
import time
import re

PORT = 8090
BIND = "100.83.83.44"
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache
_cache = {"data": None, "ts": 0, "ttl": 300}


def fetch_polymarket_weather():
    """Fetch weather prediction data from Polymarket page."""
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < _cache["ttl"]:
        return _cache["data"]

    url = "https://polymarket.com/predictions/weather"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    })

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[Polymarket] Fetch error: {e}", flush=True)
        return _cache.get("data") or []

    # Extract __NEXT_DATA__ JSON
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
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[Polymarket] Parse error: {e}", flush=True)
        return _cache.get("data") or []

    # Process events into clean structure
    result = []
    for ev in events_raw:
        title = ev.get("title", "")
        if "temperature" not in title.lower() and "highest" not in title.lower():
            continue

        # Parse city and date from title
        city_match = re.search(r"in (.+?) on (.+?)[\?]", title)
        city = city_match.group(1) if city_match else ""
        date_str = city_match.group(2).strip() if city_match else ""

        markets = ev.get("markets", [])
        outcomes = []
        for m in markets:
            label = m.get("groupItemTitle", m.get("question", ""))
            # Get best price/probability
            price = m.get("bestBid") or m.get("lastTradePrice") or 0
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = 0
            outcomes.append({"range": label, "probability": price})

        # Sort by probability descending
        outcomes.sort(key=lambda x: x["probability"], reverse=True)

        result.append({
            "title": title,
            "city": city,
            "date": date_str,
            "outcomes": outcomes,
            "volume": ev.get("volume", "0"),
            "liquidity": ev.get("liquidity", "0"),
        })

    _cache["data"] = result
    _cache["ts"] = now
    print(f"[Polymarket] Cached {len(result)} weather events", flush=True)
    return result


# City alias mapping for fuzzy search
CITY_ALIASES = {
    # Polymarket city names as values
    "nyc": ["new york", "ny", "jfk", "new york (jfk)", "nova york", "nova iorque"],
    "london": ["london", "heathrow", "london (heathrow)", "londres"],
    "tokyo": ["tokyo", "haneda", "tokyo (haneda)", "tóquio", "toquio"],
    "seoul": ["seoul", "seul"],
    "beijing": ["beijing", "pequim"],
    "ankara": ["ankara", "ancara"],
    "wellington": ["wellington"],
    "toronto": ["toronto"],
    "chicago": ["chicago"],
    "dallas": ["dallas"],
    "seattle": ["seattle"],
    "miami": ["miami"],
    "atlanta": ["atlanta"],
    "buenos aires": ["buenos aires"],
}

def match_city_events(events, query):
    """Match events by city with fuzzy alias support."""
    query = query.lower().strip()
    
    # Direct substring match first
    direct = [e for e in events if query in e["city"].lower()]
    if direct:
        return direct
    
    # Try alias lookup: find which Polymarket city this query maps to
    target_city = None
    for poly_city, aliases in CITY_ALIASES.items():
        if query in aliases or query == poly_city:
            target_city = poly_city
            break
    
    if target_city:
        return [e for e in events if target_city in e["city"].lower()]
    
    # Fallback: partial match on any alias
    for poly_city, aliases in CITY_ALIASES.items():
        for alias in aliases:
            if alias in query or query in alias:
                return [e for e in events if poly_city in e["city"].lower()]
    
    return []


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/polymarket":
            events = fetch_polymarket_weather()
            self.send_json(events)
            return

        if parsed.path == "/api/polymarket/city":
            params = urllib.parse.parse_qs(parsed.query)
            city_q = params.get("city", [""])[0].lower().strip()
            if not city_q:
                self.send_json([])
                return
            events = fetch_polymarket_weather()
            # Fuzzy matching with aliases
            matched = match_city_events(events, city_q)
            self.send_json(matched)
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
