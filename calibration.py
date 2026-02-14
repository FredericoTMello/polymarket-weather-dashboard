"""
Simple parameter calibration via historical walk-forward backtest.

This calibrates trading thresholds to reduce false signals using weather-only proxies:
- market-implied temp proxy: rolling mean of previous 14 days
- model forecast proxy: weighted average of previous 3 days
- realized temp: actual daily max at day t
"""
from __future__ import annotations

import json
import math
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta

import config


@dataclass(slots=True)
class CalibratedParams:
    safety_margin_c: float
    min_edge_to_trade: float
    kelly_fraction: float
    precision: float
    recall: float
    trade_count: int
    score: float


def fetch_daily_max_series(lat: float, lon: float, start_date: str, end_date: str) -> list[float]:
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
    arr = payload.get("daily", {}).get("temperature_2m_max", [])
    return [float(x) for x in arr if x is not None]


def calibrate_parameters(days: int = 365) -> CalibratedParams | None:
    end = date.today()
    start = end - timedelta(days=days + 40)
    start_s = start.isoformat()
    end_s = end.isoformat()

    city_series: dict[str, list[float]] = {}
    for _, key in sorted((v, k) for k, v in config.CITY_ALIASES.items() if v in config.CITY_COORDS):
        pass
    # De-duplicate by coord key
    used = set()
    for coord_key, (lat, lon) in config.CITY_COORDS.items():
        if coord_key in used:
            continue
        used.add(coord_key)
        series = fetch_daily_max_series(lat, lon, start_s, end_s)
        if len(series) >= 120:
            city_series[coord_key] = series

    if not city_series:
        return None

    margin_grid = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    edge_grid = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    kelly_grid = [0.10, 0.20, 0.25, 0.33, 0.40]

    best: CalibratedParams | None = None

    for margin in margin_grid:
        for min_edge in edge_grid:
            for kelly in kelly_grid:
                tp = fp = fn = trades = 0

                for series in city_series.values():
                    # Warmup required for rolling windows.
                    for t in range(30, len(series)):
                        hist14 = series[t - 14:t]
                        hist7 = series[t - 7:t]
                        hist3 = series[t - 3:t]
                        hist30 = series[t - 30:t]
                        real_t = series[t]
                        if len(hist14) < 14 or len(hist7) < 7 or len(hist3) < 3:
                            continue

                        implied = statistics.fmean(hist14)
                        # Forecast proxy: recent momentum + weekly baseline.
                        forecast = 0.5 * statistics.fmean(hist3) + 0.5 * statistics.fmean(hist7)
                        sigma = max(1.0, statistics.pstdev(hist30) if len(hist30) > 1 else 1.0)

                        delta_pred = forecast - implied
                        p_fair_heat = 1.0 / (1.0 + math.exp(-(delta_pred / sigma)))
                        edge_proxy = abs(p_fair_heat - 0.5) * 2.0  # 0..1
                        if edge_proxy < min_edge:
                            # No trade predicted.
                            if real_t > implied + margin or real_t < implied - margin:
                                fn += 1
                            continue

                        signal = "HOLD"
                        if delta_pred > margin:
                            signal = "BUY_HEAT"
                        elif delta_pred < -margin:
                            signal = "BUY_COLD"

                        if signal == "HOLD":
                            if real_t > implied + margin or real_t < implied - margin:
                                fn += 1
                            continue

                        trades += 1
                        realized_heat = real_t > implied
                        correct = (signal == "BUY_HEAT" and realized_heat) or (signal == "BUY_COLD" and not realized_heat)
                        if correct:
                            tp += 1
                        else:
                            fp += 1

                precision = tp / trades if trades else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                # Bias towards fewer false positives while keeping activity.
                score = (precision * 0.75 + recall * 0.25) * math.log1p(trades) * (1.0 + 0.2 * kelly)

                candidate = CalibratedParams(
                    safety_margin_c=margin,
                    min_edge_to_trade=min_edge,
                    kelly_fraction=kelly,
                    precision=precision,
                    recall=recall,
                    trade_count=trades,
                    score=score,
                )
                if best is None or candidate.score > best.score:
                    best = candidate

    return best

