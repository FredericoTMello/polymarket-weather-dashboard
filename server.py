#!/usr/bin/env python3
"""
Weather Intelligence Dashboard - local server with Polymarket-backed data.
Scrapes the public weather predictions page and exposes a small local API.
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import logging
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

_cache: dict[str, Any] = {"data": None, "ts": 0.0, "ttl": 300}
_control_lock = threading.Lock()
_control_logs: list[str] = []
_control_tasks: dict[str, dict[str, Any]] = {}
_task_seq = 0
_monitor_proc: subprocess.Popen[str] | None = None


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
    return data


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


def filter_market_events(events, query="", city_query="", limit=None):
    """Filter market events for market-first entry without changing the payload."""
    filtered = list(events)

    if city_query:
        filtered = match_city_events(filtered, city_query)

    if query:
        query_folded = fold_text(query)
        filtered = [
            event
            for event in filtered
            if query_folded in fold_text(event.get("question", ""))
            or query_folded in fold_text(event.get("city", ""))
            or query_folded in fold_text(event.get("date", ""))
            or query_folded in fold_text(event.get("city_key", ""))
        ]

    if limit is not None:
        try:
            limit_value = max(0, int(limit))
        except (TypeError, ValueError):
            limit_value = None
        if limit_value is not None:
            filtered = filtered[:limit_value]

    return filtered


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "application/javascript",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if parsed.path == "/api/polymarket/markets":
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            city_q = params.get("city", [""])[0].strip()
            limit = params.get("limit", [""])[0].strip() or None
            events = fetch_polymarket_weather()
            self.send_json(filter_market_events(events, query=query, city_query=city_q, limit=limit))
            return

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

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_json_body()

        if path == "/api/control/monitor/start":
            ok, msg = _start_monitor_process()
            self.send_json({"ok": ok, "message": msg, "status": _control_status()})
            return

        if path == "/api/control/monitor/stop":
            ok, msg = _stop_monitor_process()
            self.send_json({"ok": ok, "message": msg, "status": _control_status()})
            return

        if path == "/api/control/monitor/once":
            task_id = _start_task("monitor_once", ["monitor", "--once"])
            self.send_json({"ok": True, "task_id": task_id, "status": _control_status()})
            return
        if path in ("/api/control/monitor/once/relaxed", "/api/control/monitor/relaxed"):
            task_id = _start_task("monitor_once_relaxed", ["monitor", "--once", "--relaxed-risk"])
            self.send_json({"ok": True, "task_id": task_id, "status": _control_status()})
            return

        if path == "/api/control/validate":
            days = int(body.get("days", 730))
            train_ratio = float(body.get("train_ratio", 0.7))
            apply_best = bool(body.get("apply_best", False))
            args = ["validate", "--days", str(max(365, days)), "--train-ratio", str(max(0.5, min(0.9, train_ratio)))]
            if apply_best:
                args.append("--apply-best")
            task_id = _start_task("validate", args)
            self.send_json({"ok": True, "task_id": task_id, "status": _control_status()})
            return

        self.send_json({"ok": False, "error": "Unknown control endpoint"}, status=404)

    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "max-age=60")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):
        pass

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def run_web(bind: str, port: int) -> None:
    print(f"Dashboard: http://{bind}:{port}/dashboard.html", flush=True)
    server = http.server.HTTPServer((bind, port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket weather orchestrator")
    sub = parser.add_subparsers(dest="mode")

    mon = sub.add_parser("monitor", help="Run monitoring loop")
    mon.add_argument("--once", action="store_true", help="Run one cycle and exit")
    mon.add_argument(
        "--relaxed-risk",
        action="store_true",
        help="Relax filters to inspect more candidates (best with --once)",
    )

    web = sub.add_parser("web", help="Run dashboard HTTP server")
    web.add_argument("--bind", default=DEFAULT_BIND)
    web.add_argument("--port", type=int, default=DEFAULT_PORT)

    cal = sub.add_parser("calibrate", help="Calibrate trading thresholds from historical data")
    cal.add_argument("--days", type=int, default=365)
    cal.add_argument("--apply", action="store_true", help="Persist best params into config.py")

    val = sub.add_parser("validate", help="Out-of-sample temporal validation")
    val.add_argument("--days", type=int, default=730)
    val.add_argument("--train-ratio", type=float, default=0.7)
    val.add_argument("--apply-best", action="store_true", help="Apply train-selected params after validation")

    aud = sub.add_parser("audit", help="Statistical audit first, then market comparison")
    aud.add_argument("--days", type=int, default=730)
    aud.add_argument("--train-ratio", type=float, default=0.7)

    parser.set_defaults(mode="monitor")
    return parser.parse_args()


def apply_calibration_to_config(best) -> None:
    path = os.path.join(DASHBOARD_DIR, "config.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def _replace(name: str, value: str) -> str:
        import re
        pattern = rf"^{name}\s*=\s*.*$"
        repl = f"{name} = {value}"
        return re.sub(pattern, repl, text, flags=re.MULTILINE)

    updated = text
    for name, value in [
        ("SAFETY_MARGIN_C", f"{best.safety_margin_c:.2f}"),
        ("MIN_EDGE_TO_TRADE", f"{best.min_edge_to_trade:.2f}"),
        ("KELLY_FRACTION", f"{best.kelly_fraction:.2f}"),
    ]:
        import re
        pattern = rf"^{name}\s*=\s*.*$"
        repl = f"{name} = {value}"
        updated = re.sub(pattern, repl, updated, flags=re.MULTILINE)

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "web":
        run_web(bind=args.bind, port=args.port)
    elif args.mode == "calibrate":
        best = calibrate_parameters(days=max(180, int(args.days)))
        if not best:
            print("Calibration failed: insufficient historical data.")
            raise SystemExit(1)
        print(
            "Best params | "
            f"SAFETY_MARGIN_C={best.safety_margin_c:.2f} "
            f"MIN_EDGE_TO_TRADE={best.min_edge_to_trade:.2f} "
            f"KELLY_FRACTION={best.kelly_fraction:.2f} | "
            f"precision={best.metrics.precision:.3f} recall={best.metrics.recall:.3f} trades={best.metrics.trade_count}"
        )
        if getattr(args, "apply", False):
            apply_calibration_to_config(best)
            print("Applied calibrated params to config.py")
    elif args.mode == "validate":
        report = validate_oos(days=max(365, int(args.days)), train_ratio=max(0.5, min(0.9, float(args.train_ratio))))
        if not report:
            print("Validation failed: insufficient historical data.")
            raise SystemExit(1)
        bp = report.best_params
        print(
            "Best(train) params | "
            f"SAFETY_MARGIN_C={bp.safety_margin_c:.2f} "
            f"MIN_EDGE_TO_TRADE={bp.min_edge_to_trade:.2f} "
            f"KELLY_FRACTION={bp.kelly_fraction:.2f}"
        )
        print(
            "Train metrics | "
            f"precision={report.train_metrics.precision:.3f} "
            f"recall={report.train_metrics.recall:.3f} "
            f"trades={report.train_metrics.trade_count}"
        )
        print(
            "Test metrics (OOS) | "
            f"precision={report.test_metrics.precision:.3f} "
            f"recall={report.test_metrics.recall:.3f} "
            f"trades={report.test_metrics.trade_count}"
        )
        print(
            "Walk-forward metrics | "
            f"precision={report.walk_forward_metrics.precision:.3f} "
            f"recall={report.walk_forward_metrics.recall:.3f} "
            f"trades={report.walk_forward_metrics.trade_count}"
        )
        if getattr(args, "apply_best", False):
            apply_calibration_to_config(bp)
            print("Applied train-selected params to config.py")
    elif args.mode == "audit":
        report = run_statistical_audit(days=max(365, int(args.days)), train_ratio=max(0.55, min(0.9, float(args.train_ratio))))
        print(
            f"Audit | days={report.days} train_ratio={report.train_ratio:.2f} "
            f"reliable={report.summary['reliable_cities']}/{report.summary['total_cities']}"
        )
        print(report.summary["north"])
        print("-" * 118)
        print(
            f"{'City':22} {'Rel':4} {'MAE(test)':>10} {'RMSE(test)':>11} {'Brier':>8} "
            f"{'Cov90':>8} {'TempNow':>9} {'PM_Impl':>9} {'EdgeC':>8}"
        )
        print("-" * 118)
        for c in report.cities:
            temp_now = "n/a" if c.current_estimated_max_c is None else f"{c.current_estimated_max_c:.1f}"
            pm_imp = "n/a" if c.polymarket_implied_c is None else f"{c.polymarket_implied_c:.1f}"
            edge = "n/a" if c.current_edge_c is None else f"{c.current_edge_c:+.1f}"
            print(
                f"{c.city_key:22} {('yes' if c.model_reliable else 'no'):4} "
                f"{c.test_mae_c:10.2f} {c.test_rmse_c:11.2f} {c.test_brier:8.3f} "
                f"{c.interval90_coverage:8.2f} {temp_now:>9} {pm_imp:>9} {edge:>8}"
            )
        print("-" * 118)
        print(f"ready_for_market_step={report.summary['ready_for_market_step']}")
    else:
        run_monitor_loop(
            once=bool(getattr(args, "once", False)),
            relaxed_risk=bool(getattr(args, "relaxed_risk", False)),
        )
