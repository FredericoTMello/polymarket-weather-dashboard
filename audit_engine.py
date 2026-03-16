"""
Statistical audit first, market second.

Goal:
1) Validate whether temperature modeling is reliable enough.
2) Only then compare with live Polymarket implied prices.
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import config
from poly_client import PolymarketClient
from weather_aggregator import WeatherAggregator


@dataclass(slots=True)
class CityAudit:
    city_key: str
    data_points: int
    missing_pct: float
    train_mae_c: float
    test_mae_c: float
    train_rmse_c: float
    test_rmse_c: float
    test_brier: float
    test_logloss: float
    interval90_coverage: float
    model_reliable: bool
    current_estimated_max_c: float | None
    polymarket_implied_c: float | None
    current_edge_c: float | None


@dataclass(slots=True)
class AuditReport:
    days: int
    train_ratio: float
    cities: list[CityAudit]
    summary: dict[str, Any]


def _fetch_daily_max(lat: float, lon: float, start_date: str, end_date: str) -> tuple[list[str], list[float | None]]:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        + urllib.parse.urlencode(
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_max",
                "timezone": "UTC",
                "temperature_unit": "celsius",
            }
        )
    )
    req = urllib.request.Request(url, headers={"User-Agent": "polymarket-weather-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    days = payload.get("daily", {}).get("time", []) or []
    vals = payload.get("daily", {}).get("temperature_2m_max", []) or []
    out: list[float | None] = []
    for v in vals:
        try:
            out.append(float(v) if v is not None else None)
        except (TypeError, ValueError):
            out.append(None)
    return days, out


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _rmse(errs: list[float]) -> float:
    return math.sqrt(_mean([e * e for e in errs])) if errs else 0.0


def _split_idx(n: int, ratio: float) -> int:
    return max(40, min(n - 20, int(n * ratio)))


def _run_backtest(series: list[float], split: int) -> dict[str, float]:
    train_abs: list[float] = []
    train_err: list[float] = []
    test_abs: list[float] = []
    test_err: list[float] = []
    test_brier: list[float] = []
    test_logloss: list[float] = []
    cov_hits = 0
    cov_n = 0

    for t in range(30, len(series)):
        hist3 = series[t - 3 : t]
        hist7 = series[t - 7 : t]
        hist14 = series[t - 14 : t]
        hist30 = series[t - 30 : t]
        y = series[t]

        pred = 0.5 * _mean(hist3) + 0.5 * _mean(hist7)
        sigma = max(1.0, statistics.pstdev(hist30) if len(hist30) > 1 else 1.0)
        err = pred - y
        abs_err = abs(err)

        # Binary auxiliary target: "above rolling implied center"
        implied = _mean(hist14)
        z = (pred - implied) / sigma
        p = _clamp01(1.0 / (1.0 + math.exp(-z)))
        event = 1.0 if y > implied else 0.0

        if t < split:
            train_abs.append(abs_err)
            train_err.append(err)
            continue

        test_abs.append(abs_err)
        test_err.append(err)
        test_brier.append((p - event) ** 2)
        p_eps = min(1 - 1e-9, max(1e-9, p))
        test_logloss.append(-(event * math.log(p_eps) + (1 - event) * math.log(1 - p_eps)))

        lo = pred - 1.64 * sigma
        hi = pred + 1.64 * sigma
        cov_n += 1
        if lo <= y <= hi:
            cov_hits += 1

    return {
        "train_mae": _mean(train_abs),
        "test_mae": _mean(test_abs),
        "train_rmse": _rmse(train_err),
        "test_rmse": _rmse(test_err),
        "test_brier": _mean(test_brier),
        "test_logloss": _mean(test_logloss),
        "coverage90": (cov_hits / cov_n) if cov_n else 0.0,
    }


def _is_reliable(m: dict[str, float]) -> bool:
    # Conservative gates for daily max temperature modeling.
    return (
        m["test_mae"] <= 2.8
        and m["test_rmse"] <= 3.6
        and 0.72 <= m["coverage90"] <= 0.96
        and m["test_brier"] <= 0.24
    )


async def _current_estimates() -> dict[str, float | None]:
    agg = WeatherAggregator()
    out: dict[str, float | None] = {}
    tasks = {}
    for key, (lat, lon) in config.CITY_COORDS.items():
        tasks[key] = asyncio.create_task(agg.aggregate_daily_high(lat, lon))
    for key, task in tasks.items():
        try:
            r = await task
            out[key] = r.median_c
        except Exception:
            out[key] = None
    return out


def _market_implied_by_city() -> dict[str, float]:
    client = PolymarketClient(timeout_seconds=config.REQUEST_TIMEOUT_SECONDS)
    mkts = client.fetch_temperature_markets()
    out: dict[str, float] = {}
    for m in mkts:
        c = m.city.lower().strip()
        if c in out:
            continue
        implied = client.implied_temperature_c(m.outcomes)
        if implied is not None:
            out[c] = implied
    return out


def run_statistical_audit(days: int = 730, train_ratio: float = 0.7) -> AuditReport:
    end = date.today()
    start = end - timedelta(days=days + 35)
    start_s = start.isoformat()
    end_s = end.isoformat()

    current_est = asyncio.run(_current_estimates())
    market_imp = _market_implied_by_city()

    cities: list[CityAudit] = []
    reliable_n = 0
    for key, (lat, lon) in config.CITY_COORDS.items():
        _, vals = _fetch_daily_max(lat, lon, start_s, end_s)
        total = len(vals)
        miss = sum(v is None for v in vals)
        series = [v for v in vals if v is not None]
        if len(series) < 120:
            city = CityAudit(
                city_key=key,
                data_points=len(series),
                missing_pct=(miss / total) if total else 1.0,
                train_mae_c=0.0,
                test_mae_c=0.0,
                train_rmse_c=0.0,
                test_rmse_c=0.0,
                test_brier=0.0,
                test_logloss=0.0,
                interval90_coverage=0.0,
                model_reliable=False,
                current_estimated_max_c=current_est.get(key),
                polymarket_implied_c=None,
                current_edge_c=None,
            )
            cities.append(city)
            continue

        split = _split_idx(len(series), train_ratio)
        m = _run_backtest(series, split)
        ok = _is_reliable(m)
        if ok:
            reliable_n += 1

        # Map coord-key to market city alias for current comparison.
        market_city_candidates = {
            "london_city": ["london"],
            "new_york_laguardia": ["new york", "nyc"],
            "seoul_incheon": ["seoul"],
        }.get(key, [key.replace("_", " ")])
        pm_imp = None
        for c in market_city_candidates:
            if c in market_imp:
                pm_imp = market_imp[c]
                break

        cur_est = current_est.get(key)
        cur_edge = (cur_est - pm_imp) if (cur_est is not None and pm_imp is not None) else None

        cities.append(
            CityAudit(
                city_key=key,
                data_points=len(series),
                missing_pct=(miss / total) if total else 0.0,
                train_mae_c=m["train_mae"],
                test_mae_c=m["test_mae"],
                train_rmse_c=m["train_rmse"],
                test_rmse_c=m["test_rmse"],
                test_brier=m["test_brier"],
                test_logloss=m["test_logloss"],
                interval90_coverage=m["coverage90"],
                model_reliable=ok,
                current_estimated_max_c=cur_est,
                polymarket_implied_c=pm_imp,
                current_edge_c=cur_edge,
            )
        )

    summary = {
        "reliable_cities": reliable_n,
        "total_cities": len(cities),
        "ready_for_market_step": reliable_n >= max(1, len(cities) - 1),
        "north": "1) validar estatistica de temperatura 2) cruzar com Polymarket 3) decidir entrada",
    }
    return AuditReport(days=days, train_ratio=train_ratio, cities=cities, summary=summary)

