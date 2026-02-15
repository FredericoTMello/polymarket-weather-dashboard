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
import subprocess
import sys
import time
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import config
from audit_engine import run_statistical_audit
from calibration import calibrate_parameters, validate_oos
from poly_client import MarketSnapshot, PolymarketClient
from stats_engine import MarketEvaluation, StatsEngine
from weather_aggregator import AggregateResult, WeatherAggregator


DEFAULT_PORT = 8090
DEFAULT_BIND = "127.0.0.1"
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

_cache: dict[str, Any] = {"data": None, "ts": 0.0, "ttl": 300}
_control_lock = threading.Lock()
_control_logs: list[str] = []
_control_tasks: dict[str, dict[str, Any]] = {}
_task_seq = 0
_monitor_proc: subprocess.Popen[str] | None = None


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("poly-weather")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream)
    return logger


def _control_log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _control_lock:
        _control_logs.append(line)
        if len(_control_logs) > 300:
            del _control_logs[:120]


def _start_task(name: str, args: list[str]) -> str:
    global _task_seq
    with _control_lock:
        _task_seq += 1
        task_id = f"task-{_task_seq}"
        _control_tasks[task_id] = {
            "id": task_id,
            "name": name,
            "status": "running",
            "args": args,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "exit_code": None,
        }

    def worker() -> None:
        cmd = [sys.executable, "server.py"] + args
        _control_log(f"Started {name}: {' '.join(args)}")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=DASHBOARD_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _control_log(f"{name}> {line.rstrip()}")
            code = proc.wait()
        except Exception as exc:
            code = -1
            _control_log(f"{name} failed: {exc}")

        with _control_lock:
            t = _control_tasks.get(task_id)
            if t:
                t["status"] = "finished" if code == 0 else "failed"
                t["exit_code"] = code
                t["finished_at"] = datetime.now(timezone.utc).isoformat()
        _control_log(f"Finished {name} with exit={code}")

    threading.Thread(target=worker, daemon=True).start()
    return task_id


def _start_monitor_process() -> tuple[bool, str]:
    global _monitor_proc
    with _control_lock:
        if _monitor_proc and _monitor_proc.poll() is None:
            return False, "Monitor loop already running."
        try:
            _monitor_proc = subprocess.Popen(
                [sys.executable, "server.py", "monitor"],
                cwd=DASHBOARD_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc = _monitor_proc
        except Exception as exc:
            return False, f"Failed to start monitor loop: {exc}"

    def stream() -> None:
        assert proc is not None
        if proc.stdout:
            for line in proc.stdout:
                _control_log(f"monitor> {line.rstrip()}")
        code = proc.wait()
        _control_log(f"Monitor loop exited with code {code}")
        with _control_lock:
            global _monitor_proc
            if _monitor_proc is proc:
                _monitor_proc = None

    threading.Thread(target=stream, daemon=True).start()
    return True, "Monitor loop started."


def _stop_monitor_process() -> tuple[bool, str]:
    global _monitor_proc
    with _control_lock:
        proc = _monitor_proc
        _monitor_proc = None
    if not proc or proc.poll() is not None:
        return False, "Monitor loop is not running."
    proc.terminate()
    return True, "Monitor loop stopped."


def _control_status() -> dict[str, Any]:
    with _control_lock:
        monitor_running = bool(_monitor_proc and _monitor_proc.poll() is None)
        tasks = list(_control_tasks.values())[-20:]
        logs = _control_logs[-120:]
    return {"monitor_running": monitor_running, "tasks": tasks, "logs": logs}


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


def is_focus_city(city_name: str) -> bool:
    city = city_name.lower().strip()
    if not city:
        return False
    return any(focus in city for focus in config.FOCUS_MARKET_CITIES)


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
    if not ev.best_outcome and not ev.best_package:
        return
    is_pack = ev.recommended_action.startswith("BUY PACK") and ev.best_package is not None
    edge = ev.best_package.edge_sum if is_pack else (ev.best_outcome.edge if ev.best_outcome else 0.0)
    roi = ev.best_package.expected_roi if is_pack else (ev.best_outcome.expected_roi if ev.best_outcome else 0.0)
    kelly = 0.0 if is_pack else (ev.best_outcome.kelly_fraction if ev.best_outcome else 0.0)
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "city": ev.city,
        "date": ev.date_text,
        "title": ev.title,
        "action": ev.recommended_action,
        "direction": ev.directional_signal,
        "real_temp_c": round(ev.real_temp_c, 3),
        "implied_temp_c": None if ev.implied_temp_c is None else round(ev.implied_temp_c, 3),
        "outcome": None if is_pack else (ev.best_outcome.label if ev.best_outcome else None),
        "package_labels": ev.best_package.labels if is_pack else None,
        "package_weights": ev.best_package.suggested_weights if is_pack else None,
        "edge": round(edge, 6),
        "expected_roi": round(roi, 6),
        "kelly_fraction": round(kelly, 6),
    }
    if ev.best_outcome:
        payload["market_probability"] = round(ev.best_outcome.market_probability, 6)
        payload["fair_probability"] = round(ev.best_outcome.fair_probability, 6)
    with open(config.OPPORTUNITY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    logger.info(
        "OPPORTUNITY city=%s date=%s action=%s edge=%.2f%% roi=%.2f%% kelly=%.2f%%",
        ev.city,
        ev.date_text,
        ev.recommended_action,
        100 * edge,
        100 * roi,
        100 * kelly,
    )


def print_console_dashboard(
    rows: list[dict[str, Any]],
    scanned_count: int,
    skipped_count: int,
    elapsed_s: float,
) -> None:
    print()
    print("=" * 116)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Markets={scanned_count}  Signals={len(rows)}  Skipped={skipped_count}  Elapsed={elapsed_s:.1f}s"
    )
    print("-" * 116)
    print(
        f"{'City':16} {'Rel':4} {'Date':13} {'Action':20} "
        f"{'Edge%':>7} {'Net%':>7} {'RiskC':>7} {'ROI%':>7} {'Kelly%':>7} {'RealC':>7} {'ImplC':>7}"
    )
    print("-" * 116)
    for row in rows[:25]:
        ev: MarketEvaluation = row["evaluation"]
        b = ev.best_outcome
        pack = ev.best_package
        if not b and not pack:
            continue
        rel = row.get("reliable")
        rel_txt = "yes" if rel is True else ("no" if rel is False else "n/a")
        is_pack = ev.recommended_action.startswith("BUY PACK") and pack is not None
        edge = pack.edge_sum if is_pack else (b.edge if b else 0.0)
        roi = pack.expected_roi if is_pack else (b.expected_roi if b else 0.0)
        kelly = 0.0 if is_pack else (b.kelly_fraction if b else 0.0)
        net_edge = row.get("net_edge", edge)
        station_gap_c = row.get("station_gap_c", 0.0)
        impl = "n/a" if ev.implied_temp_c is None else f"{ev.implied_temp_c:.1f}"
        print(
            f"{ev.city[:16]:16} {rel_txt:4} {ev.date_text[:13]:13} {ev.recommended_action[:20]:20} "
            f"{100*edge:7.1f} {100*net_edge:7.1f} {station_gap_c:7.2f} {100*roi:7.1f} {100*kelly:7.1f} "
            f"{ev.real_temp_c:7.1f} {impl:>7}"
        )
    if not rows:
        print("No actionable opportunities in this cycle.")
    print("=" * 116)


def run_monitor_loop(once: bool = False, relaxed_risk: bool = False) -> None:
    logger = setup_logger()
    poly = PolymarketClient(timeout_seconds=config.REQUEST_TIMEOUT_SECONDS)
    aggregator = WeatherAggregator(timeout_seconds=config.REQUEST_TIMEOUT_SECONDS)
    engine = StatsEngine(
        safety_margin_c=(max(0.2, config.SAFETY_MARGIN_C * 0.5) if relaxed_risk else config.SAFETY_MARGIN_C),
        min_edge_to_trade=(max(0.005, config.MIN_EDGE_TO_TRADE * 0.25) if relaxed_risk else config.MIN_EDGE_TO_TRADE),
        kelly_fraction=config.KELLY_FRACTION,
        max_kelly_fraction=config.MAX_KELLY_FRACTION,
    )
    coord_cache: dict[str, tuple[float, float]] = {}
    reliability_map: dict[str, bool] = {}
    min_sources_for_signal = 1 if relaxed_risk else config.MIN_SOURCES_FOR_SIGNAL
    min_net_edge_to_trade = -0.50 if relaxed_risk else config.MIN_NET_EDGE_TO_TRADE
    station_penalty_scale = 0.35 if relaxed_risk else 1.0

    try:
        audit = run_statistical_audit(days=730, train_ratio=0.7)
        reliability_map = {c.city_key: c.model_reliable for c in audit.cities}
    except Exception as exc:
        logger.warning("Reliability audit unavailable at startup: %s", exc)

    while True:
        cycle_start = time.time()
        actionable_rows: list[dict[str, Any]] = []
        skipped = 0

        try:
            markets = poly.fetch_temperature_markets()
        except Exception as exc:
            logger.exception("Failed to fetch Polymarket markets: %s", exc)
            markets = []

        markets = [m for m in markets if is_focus_city(m.city)]

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

            if len(agg.cleaned_samples) < min_sources_for_signal:
                skipped += 1
                continue

            sigma = float(agg.stats.get("cleaned", {}).get("stdev", 1.5) or 1.5)
            ev = engine.evaluate_market(market, real_temp_c=agg.median_c, sigma_c=sigma, poly_client=poly)
            if ev.recommended_action == "SKIP":
                if not relaxed_risk:
                    skipped += 1
                    continue
                if ev.best_package:
                    ev.recommended_action = f"WATCH PACK {' | '.join(ev.best_package.labels)}"
                elif ev.best_outcome:
                    ev.recommended_action = f"WATCH {ev.best_outcome.label}"
                else:
                    skipped += 1
                    continue

            city_key = _to_coord_key(market.city)
            reliable = reliability_map.get(city_key)
            is_pack = ev.recommended_action.startswith("BUY PACK") and ev.best_package is not None
            if is_pack:
                exec_cost = _estimate_package_execution_cost(
                    market,
                    ev.best_package.labels,
                    ev.best_package.suggested_weights,
                )
                raw_edge = ev.best_package.edge_sum
            else:
                exec_cost = _estimate_execution_cost(market, ev.best_outcome.label if ev.best_outcome else "")
                raw_edge = ev.best_outcome.edge if ev.best_outcome else 0.0
            station_gap_c, station_penalty = _estimate_station_risk_penalty(market.city, agg)
            station_penalty *= station_penalty_scale
            net_edge = raw_edge - exec_cost - station_penalty
            if net_edge < min_net_edge_to_trade:
                skipped += 1
                continue

            actionable_rows.append(
                {
                    "evaluation": ev,
                    "reliable": reliable,
                    "exec_cost": exec_cost,
                    "net_edge": net_edge,
                    "station_gap_c": station_gap_c,
                    "station_penalty": station_penalty,
                }
            )
            if ev.recommended_action.startswith("BUY"):
                log_opportunity(logger, ev)

        actionable_rows.sort(key=lambda x: x.get("net_edge", 0.0), reverse=True)
        print_console_dashboard(
            rows=actionable_rows,
            scanned_count=len(markets),
            skipped_count=skipped,
            elapsed_s=time.time() - cycle_start,
        )

        if once:
            break
        time.sleep(config.MONITOR_INTERVAL_SECONDS)


def _to_coord_key(city_name: str) -> str:
    city = city_name.lower().strip()
    alias = config.CITY_ALIASES.get(city)
    if alias:
        return alias
    if "london" in city:
        return "london_city"
    if "new york" in city or "laguardia" in city or "nyc" in city:
        return "new_york_laguardia"
    if "seoul" in city or "incheon" in city or "seul" in city:
        return "seoul_incheon"
    return city.replace(" ", "_")


def _estimate_execution_cost(market: MarketSnapshot, outcome_label: str) -> float:
    """
    Cost in probability points (0..1) used to derive net edge:
    - half-spread from visible bid/ask when available
    - fallback conservative micro-cost when book is sparse
    """
    if not outcome_label:
        return 0.015
    quote = next((o for o in market.outcomes if o.label == outcome_label), None)
    if quote and quote.best_bid is not None and quote.best_ask is not None and quote.best_ask >= quote.best_bid:
        return max(0.0, (quote.best_ask - quote.best_bid) / 2.0)
    return 0.015


def _estimate_package_execution_cost(
    market: MarketSnapshot,
    labels: list[str],
    weights: list[float] | None,
) -> float:
    if not labels:
        return 0.02
    if not weights or len(weights) != len(labels):
        weights = [1.0 / len(labels)] * len(labels)
    total = 0.0
    for label, w in zip(labels, weights):
        total += _estimate_execution_cost(market, label) * max(0.0, float(w))
    # Add a small coordination penalty for multi-leg execution.
    return total + 0.003


def _estimate_station_risk_penalty(city_name: str, agg: AggregateResult) -> tuple[float, float]:
    """
    Returns:
    - station_gap_c: model-vs-station proxy temperature gap in Celsius
    - penalty: edge penalty in probability points (0..1)
    """
    source_temps = agg.stats.get("source_temps", {})
    if not isinstance(source_temps, dict) or not source_temps:
        return 0.0, 0.0

    proxy_vals: list[float] = []
    model_vals: list[float] = []
    for source, value in source_temps.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if source in config.STATION_PROXY_SOURCES:
            proxy_vals.append(v)
        else:
            model_vals.append(v)

    if proxy_vals and model_vals:
        station_proxy = sum(proxy_vals) / len(proxy_vals)
        model_consensus = sum(model_vals) / len(model_vals)
        gap = abs(station_proxy - model_consensus)
    else:
        # Fallback: dispersion is a proxy for uncertainty around station alignment.
        gap = float(agg.stats.get("cleaned", {}).get("stdev", 0.0) or 0.0)

    city_bias = float(config.CITY_STATION_BIAS_C.get(_to_coord_key(city_name), 0.0))
    station_gap_c = gap + abs(city_bias)
    penalty = min(config.STATION_RISK_MAX_PENALTY, station_gap_c * config.STATION_RISK_PENALTY_PER_C)
    return station_gap_c, penalty


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

        if parsed.path == "/api/control/status":
            self.send_json(_control_status())
            return

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
