"""
Calibration + out-of-sample validation for trade thresholds.
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
class BacktestMetrics:
    precision: float
    recall: float
    trade_count: int
    tp: int
    fp: int
    fn: int
    score: float


@dataclass(slots=True)
class CalibratedParams:
    safety_margin_c: float
    min_edge_to_trade: float
    kelly_fraction: float
    metrics: BacktestMetrics


@dataclass(slots=True)
class ValidationReport:
    best_params: CalibratedParams
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    walk_forward_metrics: BacktestMetrics
    train_ratio: float
    series_count: int
    days: int


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


def _load_city_series(days: int) -> dict[str, list[float]]:
    end = date.today()
    start = end - timedelta(days=days + 40)
    start_s = start.isoformat()
    end_s = end.isoformat()

    out: dict[str, list[float]] = {}
    for coord_key, (lat, lon) in config.CITY_COORDS.items():
        series = fetch_daily_max_series(lat, lon, start_s, end_s)
        if len(series) >= 140:
            out[coord_key] = series
    return out


def _simulate_series(
    series: list[float],
    margin: float,
    min_edge: float,
    start_idx: int,
    end_idx: int,
) -> tuple[int, int, int, int]:
    tp = fp = fn = trades = 0
    s = max(30, start_idx)
    e = min(end_idx, len(series))
    for t in range(s, e):
        hist14 = series[t - 14:t]
        hist7 = series[t - 7:t]
        hist3 = series[t - 3:t]
        hist30 = series[t - 30:t]
        real_t = series[t]
        if len(hist14) < 14 or len(hist7) < 7 or len(hist3) < 3:
            continue

        implied = statistics.fmean(hist14)
        forecast = 0.5 * statistics.fmean(hist3) + 0.5 * statistics.fmean(hist7)
        sigma = max(1.0, statistics.pstdev(hist30) if len(hist30) > 1 else 1.0)

        delta_pred = forecast - implied
        p_fair_heat = 1.0 / (1.0 + math.exp(-(delta_pred / sigma)))
        edge_proxy = abs(p_fair_heat - 0.5) * 2.0
        if edge_proxy < min_edge:
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
    return tp, fp, fn, trades


def _metrics(tp: int, fp: int, fn: int, trades: int, kelly: float) -> BacktestMetrics:
    precision = tp / trades if trades else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    score = (precision * 0.75 + recall * 0.25) * math.log1p(trades) * (1.0 + 0.2 * kelly)
    return BacktestMetrics(
        precision=precision,
        recall=recall,
        trade_count=trades,
        tp=tp,
        fp=fp,
        fn=fn,
        score=score,
    )


def _grid() -> tuple[list[float], list[float], list[float]]:
    return (
        [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
        [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        [0.10, 0.20, 0.25, 0.33, 0.40],
    )


def calibrate_parameters(days: int = 365) -> CalibratedParams | None:
    series_map = _load_city_series(days)
    if not series_map:
        return None
    best, _, _ = _calibrate_on_split(series_map, train_ratio=1.0)
    return best


def _calibrate_on_split(series_map: dict[str, list[float]], train_ratio: float) -> tuple[CalibratedParams, BacktestMetrics, BacktestMetrics]:
    margins, min_edges, kellys = _grid()
    best: CalibratedParams | None = None
    best_train: BacktestMetrics | None = None
    best_test: BacktestMetrics | None = None

    for margin in margins:
        for min_edge in min_edges:
            for kelly in kellys:
                tr_tp = tr_fp = tr_fn = tr_trades = 0
                te_tp = te_fp = te_fn = te_trades = 0
                for series in series_map.values():
                    split = len(series) if train_ratio >= 1.0 else max(60, int(len(series) * train_ratio))
                    tp, fp, fn, trades = _simulate_series(series, margin, min_edge, start_idx=30, end_idx=split)
                    tr_tp += tp
                    tr_fp += fp
                    tr_fn += fn
                    tr_trades += trades

                    if split < len(series):
                        tp, fp, fn, trades = _simulate_series(series, margin, min_edge, start_idx=split, end_idx=len(series))
                        te_tp += tp
                        te_fp += fp
                        te_fn += fn
                        te_trades += trades

                train_m = _metrics(tr_tp, tr_fp, tr_fn, tr_trades, kelly)
                test_m = _metrics(te_tp, te_fp, te_fn, te_trades, kelly)
                candidate = CalibratedParams(
                    safety_margin_c=margin,
                    min_edge_to_trade=min_edge,
                    kelly_fraction=kelly,
                    metrics=train_m,
                )
                if best is None or candidate.metrics.score > best.metrics.score:
                    best = candidate
                    best_train = train_m
                    best_test = test_m

    assert best is not None and best_train is not None and best_test is not None
    return best, best_train, best_test


def _walk_forward_validate(
    series_map: dict[str, list[float]],
    best: CalibratedParams,
    train_window: int = 180,
    test_window: int = 30,
    step: int = 30,
) -> BacktestMetrics:
    total_tp = total_fp = total_fn = total_trades = 0
    for series in series_map.values():
        if len(series) < train_window + test_window + 30:
            continue
        pos = train_window
        while pos + test_window <= len(series):
            # Train segment is intentionally unused here because params are fixed.
            tp, fp, fn, trades = _simulate_series(
                series,
                margin=best.safety_margin_c,
                min_edge=best.min_edge_to_trade,
                start_idx=pos,
                end_idx=pos + test_window,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_trades += trades
            pos += step
    return _metrics(total_tp, total_fp, total_fn, total_trades, best.kelly_fraction)


def validate_oos(days: int = 730, train_ratio: float = 0.7) -> ValidationReport | None:
    series_map = _load_city_series(days)
    if not series_map:
        return None

    best, train_m, test_m = _calibrate_on_split(series_map, train_ratio=train_ratio)
    wf_m = _walk_forward_validate(series_map, best)
    return ValidationReport(
        best_params=best,
        train_metrics=train_m,
        test_metrics=test_m,
        walk_forward_metrics=wf_m,
        train_ratio=train_ratio,
        series_count=len(series_map),
        days=days,
    )

