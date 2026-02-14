#!/usr/bin/env python3
"""
Polymarket weather orchestrator.

Modes:
- monitor: quant loop (default)
- web: legacy HTTP dashboard proxy for dashboard.html
"""
from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import config
from calibration import calibrate_parameters
from poly_client import MarketSnapshot, PolymarketClient
from stats_engine import MarketEvaluation, StatsEngine
from weather_aggregator import AggregateResult, WeatherAggregator


DEFAULT_PORT = 8090
DEFAULT_BIND = "127.0.0.1"
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

_cache: dict[str, Any] = {"data": None, "ts": 0.0, "ttl": 300}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("poly-weather")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream)
    return logger


def resolve_city_coords(city_name: str, cache: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    key = city_name.strip().lower()
    if not key:
        return None
    if key in cache:
        return cache[key]

    alias_key = config.CITY_ALIASES.get(key)
    if alias_key and alias_key in config.CITY_COORDS:
        cache[key] = config.CITY_COORDS[alias_key]
        return cache[key]

    # Fallback geocode for new Polymarket cities.
    q = urllib.parse.quote(city_name)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "polymarket-weather-dashboard/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        if not results:
            return None
        lat = float(results[0]["latitude"])
        lon = float(results[0]["longitude"])
        cache[key] = (lat, lon)
        return cache[key]
    except Exception:
        return None


async def fetch_weather_batch(
    aggregator: WeatherAggregator,
    city_coords: dict[str, tuple[float, float]],
) -> dict[str, AggregateResult]:
    tasks = {city: asyncio.create_task(aggregator.aggregate_daily_high(lat, lon)) for city, (lat, lon) in city_coords.items()}
    out: dict[str, AggregateResult] = {}
    for city, task in tasks.items():
        try:
            out[city] = await task
        except Exception:
            continue
    return out


def log_opportunity(logger: logging.Logger, ev: MarketEvaluation) -> None:
    if not ev.best_outcome:
        return
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "city": ev.city,
        "date": ev.date_text,
        "title": ev.title,
        "action": ev.recommended_action,
        "direction": ev.directional_signal,
        "real_temp_c": round(ev.real_temp_c, 3),
        "implied_temp_c": None if ev.implied_temp_c is None else round(ev.implied_temp_c, 3),
        "outcome": ev.best_outcome.label,
        "edge": round(ev.best_outcome.edge, 6),
        "market_probability": round(ev.best_outcome.market_probability, 6),
        "fair_probability": round(ev.best_outcome.fair_probability, 6),
        "expected_roi": round(ev.best_outcome.expected_roi, 6),
        "kelly_fraction": round(ev.best_outcome.kelly_fraction, 6),
    }
    with open(config.OPPORTUNITY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    logger.info(
        "OPPORTUNITY city=%s date=%s action=%s edge=%.2f%% roi=%.2f%% kelly=%.2f%%",
        ev.city,
        ev.date_text,
        ev.recommended_action,
        100 * ev.best_outcome.edge,
        100 * ev.best_outcome.expected_roi,
        100 * ev.best_outcome.kelly_fraction,
    )


def print_console_dashboard(
    evaluations: list[MarketEvaluation],
    scanned_count: int,
    skipped_count: int,
    elapsed_s: float,
) -> None:
    print()
    print("=" * 106)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Markets={scanned_count}  Signals={len(evaluations)}  Skipped={skipped_count}  Elapsed={elapsed_s:.1f}s"
    )
    print("-" * 106)
    print(f"{'City':16} {'Date':13} {'Action':20} {'Edge%':>7} {'ROI%':>7} {'Kelly%':>7} {'RealC':>7} {'ImplC':>7}")
    print("-" * 106)
    for ev in evaluations[:25]:
        b = ev.best_outcome
        if not b:
            continue
        impl = "n/a" if ev.implied_temp_c is None else f"{ev.implied_temp_c:.1f}"
        print(
            f"{ev.city[:16]:16} {ev.date_text[:13]:13} {ev.recommended_action[:20]:20} "
            f"{100*b.edge:7.1f} {100*b.expected_roi:7.1f} {100*b.kelly_fraction:7.1f} "
            f"{ev.real_temp_c:7.1f} {impl:>7}"
        )
    if not evaluations:
        print("No actionable opportunities in this cycle.")
    print("=" * 106)


def run_monitor_loop(once: bool = False) -> None:
    logger = setup_logger()
    poly = PolymarketClient(timeout_seconds=config.REQUEST_TIMEOUT_SECONDS)
    aggregator = WeatherAggregator(timeout_seconds=config.REQUEST_TIMEOUT_SECONDS)
    engine = StatsEngine()
    coord_cache: dict[str, tuple[float, float]] = {}

    while True:
        cycle_start = time.time()
        actionable: list[MarketEvaluation] = []
        skipped = 0

        try:
            markets = poly.fetch_temperature_markets()
        except Exception as exc:
            logger.exception("Failed to fetch Polymarket markets: %s", exc)
            markets = []

        city_coords: dict[str, tuple[float, float]] = {}
        for m in markets:
            coords = resolve_city_coords(m.city, coord_cache)
            if coords:
                city_coords[m.city] = coords

        weather_map = asyncio.run(fetch_weather_batch(aggregator, city_coords)) if city_coords else {}

        for market in markets:
            if market.liquidity < config.MIN_MARKET_LIQUIDITY:
                skipped += 1
                continue

            agg = weather_map.get(market.city)
            if not agg or agg.median_c is None:
                skipped += 1
                continue

            if len(agg.cleaned_samples) < config.MIN_SOURCES_FOR_SIGNAL:
                skipped += 1
                continue

            sigma = float(agg.stats.get("cleaned", {}).get("stdev", 1.5) or 1.5)
            ev = engine.evaluate_market(market, real_temp_c=agg.median_c, sigma_c=sigma, poly_client=poly)
            if ev.recommended_action == "SKIP":
                skipped += 1
                continue

            actionable.append(ev)
            log_opportunity(logger, ev)

        actionable.sort(key=lambda x: (x.best_outcome.edge if x.best_outcome else 0.0), reverse=True)
        print_console_dashboard(
            evaluations=actionable,
            scanned_count=len(markets),
            skipped_count=skipped,
            elapsed_s=time.time() - cycle_start,
        )

        if once:
            break
        time.sleep(config.MONITOR_INTERVAL_SECONDS)


def market_to_api_event(market: MarketSnapshot) -> dict[str, Any]:
    outcomes = [{"range": o.label, "probability": o.probability} for o in market.outcomes]
    return {
        "title": market.title,
        "question": market.title,
        "city": market.city,
        "date": market.date_text,
        "outcomes": outcomes,
        "volume": market.volume,
        "liquidity": market.liquidity,
    }


def fetch_polymarket_weather_cached() -> list[dict[str, Any]]:
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < _cache["ttl"]:
        return _cache["data"]

    client = PolymarketClient(timeout_seconds=config.REQUEST_TIMEOUT_SECONDS)
    markets = client.fetch_temperature_markets()
    data = [market_to_api_event(m) for m in markets]
    _cache["data"] = data
    _cache["ts"] = now
    return data


def match_city_events(events: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    q = query.lower().strip()
    if not q:
        return []
    return [e for e in events if q in str(e.get("city", "")).lower()]


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/polymarket":
            self.send_json(fetch_polymarket_weather_cached())
            return

        if parsed.path == "/api/polymarket/city":
            params = urllib.parse.parse_qs(parsed.query)
            city_q = params.get("city", [""])[0]
            events = fetch_polymarket_weather_cached()
            self.send_json(match_city_events(events, city_q))
            return

        if parsed.path == "/api/geocode":
            params = urllib.parse.parse_qs(parsed.query)
            q = params.get("q", [""])[0].strip()
            if not q:
                self.send_json([])
                return
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(q)}&count=8&language=pt&format=json"
            try:
                with urllib.request.urlopen(geo_url, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.send_json({"results": data.get("results", [])})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
            return

        super().do_GET()

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

    web = sub.add_parser("web", help="Run dashboard HTTP server")
    web.add_argument("--bind", default=DEFAULT_BIND)
    web.add_argument("--port", type=int, default=DEFAULT_PORT)

    cal = sub.add_parser("calibrate", help="Calibrate trading thresholds from historical data")
    cal.add_argument("--days", type=int, default=365)
    cal.add_argument("--apply", action="store_true", help="Persist best params into config.py")

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
            f"precision={best.precision:.3f} recall={best.recall:.3f} trades={best.trade_count}"
        )
        if getattr(args, "apply", False):
            apply_calibration_to_config(best)
            print("Applied calibrated params to config.py")
    else:
        run_monitor_loop(once=bool(getattr(args, "once", False)))
