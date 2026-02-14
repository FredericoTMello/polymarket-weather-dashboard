"""
Polymarket client for weather temperature markets.

Task 2 scope:
- load active weather markets
- read contract prices + liquidity/volume
- optionally fetch CLOB order books for each outcome token
- convert contract price (0..1) to implied probability
- estimate implied temperature from outcome bins
"""
from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OutcomeQuote:
    label: str
    probability: float
    token_id: str | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    last_trade: float | None = None
    order_book: dict[str, Any] | None = None


@dataclass(slots=True)
class MarketSnapshot:
    title: str
    city: str
    date_text: str
    volume: float
    liquidity: float
    outcomes: list[OutcomeQuote] = field(default_factory=list)


class PolymarketClient:
    WEATHER_PAGE_URL = "https://polymarket.com/predictions/weather"
    CLOB_BOOK_URL = "https://clob.polymarket.com/book?token_id={token_id}"

    def __init__(self, timeout_seconds: float = 10.0, max_workers: int = 12) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self._headers = {
            "User-Agent": "Mozilla/5.0 (compatible; polymarket-weather-dashboard/1.0)",
            "Accept": "application/json,text/html,*/*",
        }

    def fetch_temperature_markets(self) -> list[MarketSnapshot]:
        raw_events = self._fetch_weather_events_raw()
        return [m for m in (self._to_snapshot(e) for e in raw_events) if m is not None]

    def enrich_order_books(self, markets: list[MarketSnapshot]) -> None:
        jobs: list[tuple[OutcomeQuote, str]] = []
        for market in markets:
            for out in market.outcomes:
                if out.token_id:
                    jobs.append((out, out.token_id))

        if not jobs:
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {pool.submit(self.fetch_order_book, token_id): outcome for outcome, token_id in jobs}
            for fut in as_completed(future_map):
                out = future_map[fut]
                try:
                    book = fut.result()
                except Exception:
                    continue
                out.order_book = book
                out.best_bid = _best_price(book, side="bids")
                out.best_ask = _best_price(book, side="asks")

    def fetch_order_book(self, token_id: str) -> dict[str, Any]:
        url = self.CLOB_BOOK_URL.format(token_id=token_id)
        return self._http_get_json(url)

    @staticmethod
    def implied_probability(price_0_to_1: float) -> float:
        if not math.isfinite(price_0_to_1):
            raise ValueError("price must be finite")
        return min(1.0, max(0.0, float(price_0_to_1)))

    def implied_temperature_c(self, outcomes: list[OutcomeQuote]) -> float | None:
        weighted_sum = 0.0
        weight_sum = 0.0

        for out in outcomes:
            p = self.implied_probability(out.probability)
            bounds = _parse_temp_range_to_c(out.label)
            if bounds is None:
                continue

            lo, hi = bounds
            midpoint = _range_midpoint(lo, hi)
            if midpoint is None:
                continue

            weighted_sum += p * midpoint
            weight_sum += p

        if weight_sum <= 0:
            return None
        return weighted_sum / weight_sum

    def _fetch_weather_events_raw(self) -> list[dict[str, Any]]:
        html = self._http_get_text(self.WEATHER_PAGE_URL)
        next_data = self._extract_next_data_json(html)
        pages = (
            next_data.get("props", {})
            .get("pageProps", {})
            .get("dehydratedState", {})
            .get("queries", [{}])[0]
            .get("state", {})
            .get("data", {})
            .get("pages", [])
        )
        events: list[dict[str, Any]] = []
        for page in pages:
            events.extend(page.get("results", []))
        return events

    def _to_snapshot(self, event: dict[str, Any]) -> MarketSnapshot | None:
        title = str(event.get("title", ""))
        if not re.search(r"temperature|highest", title, flags=re.IGNORECASE):
            return None

        city, date_text = _parse_city_and_date(title)
        outcomes: list[OutcomeQuote] = []

        for market in event.get("markets", []):
            label = str(market.get("groupItemTitle") or market.get("question") or "").strip()
            if not label:
                continue

            best_bid = _coerce_float(market.get("bestBid"))
            best_ask = _coerce_float(market.get("bestAsk"))
            last_trade = _coerce_float(market.get("lastTradePrice"))
            prob = _first_not_none(best_bid, last_trade, 0.0)
            token_id = _coerce_token_id(market)

            outcomes.append(
                OutcomeQuote(
                    label=label,
                    probability=self.implied_probability(prob),
                    token_id=token_id,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    last_trade=last_trade,
                )
            )

        outcomes.sort(key=lambda x: x.probability, reverse=True)

        return MarketSnapshot(
            title=title,
            city=city,
            date_text=date_text,
            volume=_coerce_float(event.get("volume")) or 0.0,
            liquidity=_coerce_float(event.get("liquidity")) or 0.0,
            outcomes=outcomes,
        )

    def _http_get_text(self, url: str) -> str:
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            if resp.status != 200:
                raise urllib.error.HTTPError(url, resp.status, f"HTTP {resp.status}", resp.headers, None)
            return resp.read().decode("utf-8", errors="replace")

    def _http_get_json(self, url: str) -> dict[str, Any]:
        text = self._http_get_text(url)
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object")
        return payload

    @staticmethod
    def _extract_next_data_json(html: str) -> dict[str, Any]:
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, flags=re.DOTALL)
        if not match:
            raise ValueError("Polymarket __NEXT_DATA__ payload not found")
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict):
            raise ValueError("__NEXT_DATA__ payload has invalid format")
        return payload


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _coerce_token_id(market: dict[str, Any]) -> str | None:
    for key in ("tokenId", "clobTokenId", "outcomeTokenId"):
        if market.get(key):
            return str(market.get(key))

    maybe = market.get("clobTokenIds")
    if isinstance(maybe, list) and maybe:
        return str(maybe[0])
    return None


def _parse_city_and_date(title: str) -> tuple[str, str]:
    m = re.search(r"in (.+?) on (.+?)\?", title, flags=re.IGNORECASE)
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()


def _range_midpoint(lo: float, hi: float) -> float | None:
    if math.isinf(lo) and math.isinf(hi):
        return None
    if math.isinf(lo):
        return hi - 1.0
    if math.isinf(hi):
        return lo + 1.0
    return (lo + hi) / 2.0


def _f_to_c(v: float) -> float:
    return (v - 32.0) * 5.0 / 9.0


def _parse_temp_range_to_c(label: str) -> tuple[float, float] | None:
    is_f = bool(re.search(r"°?\s*F", label, flags=re.IGNORECASE))

    def conv(x: float) -> float:
        return _f_to_c(x) if is_f else x

    m = re.search(r"([-\d.]+)\s*°?\s*[FC]?\s*(?:or below|ou menos)", label, flags=re.IGNORECASE)
    if m:
        return float("-inf"), conv(float(m.group(1)))

    m = re.search(r"([-\d.]+)\s*°?\s*[FC]?\s*(?:or higher|ou mais)", label, flags=re.IGNORECASE)
    if m:
        return conv(float(m.group(1))), float("inf")

    m = re.search(r"([-\d.]+)\s*[-–]\s*([-\d.]+)", label)
    if m:
        lo = conv(float(m.group(1)))
        hi = conv(float(m.group(2)))
        return (min(lo, hi), max(lo, hi))

    m = re.search(r"([-\d.]+)\s*°?\s*[FC]", label, flags=re.IGNORECASE)
    if m:
        v = conv(float(m.group(1)))
        return v - 0.5, v + 0.5

    return None


def _best_price(book: dict[str, Any], side: str) -> float | None:
    # CLOB book usually exposes arrays in descending bid / ascending ask order.
    levels = book.get(side)
    if not isinstance(levels, list) or not levels:
        return _coerce_float(book.get("best_bid" if side == "bids" else "best_ask"))

    first = levels[0]
    if isinstance(first, dict):
        return _coerce_float(first.get("price"))
    return None
