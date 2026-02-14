"""
Async weather data aggregator with robust median + outlier filtering.

Task 1 scope:
- multi-provider async request orchestration
- production-safe error handling and timeouts
- robust stats (median + robust z-score / MAD)
"""
from __future__ import annotations

import asyncio
import json
import math
import statistics
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import config


JsonDict = dict[str, Any]
ParserFn = Callable[[JsonDict], float | None]
UrlBuilderFn = Callable[[float, float], str]


@dataclass(slots=True)
class WeatherSource:
    name: str
    build_url: UrlBuilderFn
    parser: ParserFn
    enabled: bool = True
    requires_key: bool = False


@dataclass(slots=True)
class WeatherSample:
    source: str
    temp_c: float
    fetched_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class AggregateResult:
    lat: float
    lon: float
    median_c: float | None
    cleaned_samples: list[WeatherSample]
    dropped_outliers: list[WeatherSample]
    raw_samples: list[WeatherSample]
    stats: JsonDict


class WeatherAggregator:
    def __init__(
        self,
        timeout_seconds: float = config.REQUEST_TIMEOUT_SECONDS,
        max_concurrency: int = config.MAX_CONCURRENT_REQUESTS,
        zscore_threshold: float = config.DEFAULT_ZSCORE_THRESHOLD,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.zscore_threshold = zscore_threshold
        self._sem = asyncio.Semaphore(max_concurrency)
        self.sources: list[WeatherSource] = self._build_sources()

    async def aggregate_daily_high(self, lat: float, lon: float) -> AggregateResult:
        tasks = [self._fetch_source(src, lat, lon) for src in self.sources if src.enabled]
        raw: list[WeatherSample] = [s for s in await asyncio.gather(*tasks) if s is not None]
        cleaned, dropped = self._remove_outliers(raw, self.zscore_threshold)
        median_c = statistics.median([s.temp_c for s in cleaned]) if cleaned else None
        stats = self._build_stats(raw, cleaned, dropped)

        return AggregateResult(
            lat=lat,
            lon=lon,
            median_c=median_c,
            cleaned_samples=cleaned,
            dropped_outliers=dropped,
            raw_samples=raw,
            stats=stats,
        )

    async def _fetch_source(self, source: WeatherSource, lat: float, lon: float) -> WeatherSample | None:
        url = source.build_url(lat, lon)
        try:
            async with self._sem:
                data = await asyncio.wait_for(asyncio.to_thread(self._http_get_json, url), timeout=self.timeout_seconds)
        except (asyncio.TimeoutError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return None

        try:
            value = source.parser(data)
        except Exception:
            return None

        if value is None or not math.isfinite(value):
            return None
        return WeatherSample(source=source.name, temp_c=float(value))

    @staticmethod
    def _http_get_json(url: str) -> JsonDict:
        req = urllib.request.Request(url, headers={"User-Agent": "polymarket-weather-dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                raise urllib.error.HTTPError(url, resp.status, f"HTTP {resp.status}", resp.headers, None)
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object response")
        return data

    @staticmethod
    def _remove_outliers(samples: list[WeatherSample], z_threshold: float) -> tuple[list[WeatherSample], list[WeatherSample]]:
        if len(samples) < 4:
            return samples[:], []

        values = [s.temp_c for s in samples]
        med = statistics.median(values)
        abs_dev = [abs(v - med) for v in values]
        mad = statistics.median(abs_dev)

        # MAD = 0 means almost all values are identical; do not over-filter.
        if mad == 0:
            return samples[:], []

        cleaned: list[WeatherSample] = []
        dropped: list[WeatherSample] = []
        for sample in samples:
            robust_z = 0.6745 * (sample.temp_c - med) / mad
            if abs(robust_z) <= z_threshold:
                cleaned.append(sample)
            else:
                dropped.append(sample)
        return cleaned, dropped

    @staticmethod
    def _build_stats(raw: list[WeatherSample], cleaned: list[WeatherSample], dropped: list[WeatherSample]) -> JsonDict:
        def _summary(values: Iterable[float]) -> JsonDict:
            vals = list(values)
            if not vals:
                return {"count": 0}
            return {
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }

        return {
            "raw": _summary(s.temp_c for s in raw),
            "cleaned": _summary(s.temp_c for s in cleaned),
            "dropped_count": len(dropped),
            "sources_used": [s.source for s in cleaned],
            "sources_dropped": [s.source for s in dropped],
        }

    def _build_sources(self) -> list[WeatherSource]:
        sources: list[WeatherSource] = [
            WeatherSource(
                name="open_meteo",
                build_url=lambda lat, lon: (
                    "https://api.open-meteo.com/v1/forecast?"
                    + urllib.parse.urlencode(
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "daily": "temperature_2m_max",
                            "forecast_days": 1,
                            "timezone": "UTC",
                            "temperature_unit": "celsius",
                        }
                    )
                ),
                parser=lambda d: _safe_get(d, "daily", "temperature_2m_max", index=0),
            ),
            WeatherSource(
                name="weatherapi",
                enabled=bool(config.WEATHERAPI_KEY),
                requires_key=True,
                build_url=lambda lat, lon: (
                    "https://api.weatherapi.com/v1/forecast.json?"
                    + urllib.parse.urlencode(
                        {
                            "key": config.WEATHERAPI_KEY,
                            "q": f"{lat},{lon}",
                            "days": 1,
                            "aqi": "no",
                            "alerts": "no",
                        }
                    )
                ),
                parser=lambda d: _safe_get(d, "forecast", "forecastday", index=0, child="day", leaf="maxtemp_c"),
            ),
            WeatherSource(
                name="openweather",
                enabled=bool(config.OPENWEATHER_KEY),
                requires_key=True,
                build_url=lambda lat, lon: (
                    "https://api.openweathermap.org/data/2.5/forecast?"
                    + urllib.parse.urlencode(
                        {
                            "lat": lat,
                            "lon": lon,
                            "appid": config.OPENWEATHER_KEY,
                            "units": "metric",
                        }
                    )
                ),
                parser=_parse_openweather_max,
            ),
            WeatherSource(
                name="tomorrow_io",
                enabled=bool(config.TOMORROW_API_KEY),
                requires_key=True,
                build_url=lambda lat, lon: (
                    "https://api.tomorrow.io/v4/weather/forecast?"
                    + urllib.parse.urlencode(
                        {
                            "location": f"{lat},{lon}",
                            "timesteps": "1d",
                            "units": "metric",
                            "apikey": config.TOMORROW_API_KEY,
                        }
                    )
                ),
                parser=lambda d: _safe_get(d, "timelines", "daily", index=0, child="values", leaf="temperatureMax"),
            ),
        ]

        # Placeholder sources for scale testing (disabled by default).
        for i in range(1, 31):
            sources.append(
                WeatherSource(
                    name=f"placeholder_provider_{i:02d}",
                    enabled=False,
                    build_url=lambda lat, lon, i=i: f"https://placeholder{i}.invalid/weather?lat={lat}&lon={lon}",
                    parser=lambda _: None,
                )
            )
        return sources


def _safe_get(data: JsonDict, k1: str, k2: str, *, index: int, child: str | None = None, leaf: str | None = None) -> float | None:
    block = data.get(k1, {})
    if not isinstance(block, dict):
        return None
    arr = block.get(k2, [])
    if not isinstance(arr, list) or len(arr) <= index:
        return None
    item = arr[index]
    if child is not None:
        if not isinstance(item, dict):
            return None
        item = item.get(child)
    if leaf is not None:
        if not isinstance(item, dict):
            return None
        item = item.get(leaf)
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def _parse_openweather_max(data: JsonDict) -> float | None:
    # OpenWeather 5-day endpoint is 3-hourly. Use max as daily-high proxy.
    rows = data.get("list", [])
    if not isinstance(rows, list) or not rows:
        return None
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        main = row.get("main", {})
        if not isinstance(main, dict):
            continue
        t = main.get("temp_max")
        try:
            values.append(float(t))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None

