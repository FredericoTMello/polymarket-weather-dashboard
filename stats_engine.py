"""
Stats/arbitrage engine for Polymarket weather contracts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import config
from poly_client import MarketSnapshot, OutcomeQuote, PolymarketClient


@dataclass(slots=True)
class OutcomeEvaluation:
    label: str
    market_probability: float
    fair_probability: float
    edge: float
    expected_roi: float
    kelly_fraction: float


@dataclass(slots=True)
class MarketEvaluation:
    title: str
    city: str
    date_text: str
    real_temp_c: float
    implied_temp_c: float | None
    safety_margin_c: float
    directional_signal: str
    recommended_action: str
    best_outcome: OutcomeEvaluation | None
    outcomes: list[OutcomeEvaluation] = field(default_factory=list)


class StatsEngine:
    def __init__(
        self,
        safety_margin_c: float = config.SAFETY_MARGIN_C,
        min_edge_to_trade: float = config.MIN_EDGE_TO_TRADE,
        kelly_fraction: float = config.KELLY_FRACTION,
        max_kelly_fraction: float = config.MAX_KELLY_FRACTION,
    ) -> None:
        self.safety_margin_c = safety_margin_c
        self.min_edge_to_trade = min_edge_to_trade
        self.kelly_fraction = kelly_fraction
        self.max_kelly_fraction = max_kelly_fraction

    def evaluate_market(
        self,
        market: MarketSnapshot,
        real_temp_c: float,
        sigma_c: float,
        poly_client: PolymarketClient | None = None,
    ) -> MarketEvaluation:
        sigma = max(0.8, float(sigma_c))
        poly_client = poly_client or PolymarketClient()
        implied_temp_c = poly_client.implied_temperature_c(market.outcomes)
        directional_signal = self.directional_signal(real_temp_c, implied_temp_c)

        evaluations: list[OutcomeEvaluation] = []
        for out in market.outcomes:
            parsed = _parse_range_to_c(out.label)
            if parsed is None:
                continue
            lo, hi = parsed
            fair_prob = _prob_range_normal(real_temp_c, sigma, lo, hi)
            market_prob = min(1.0, max(0.0, out.probability))
            edge = fair_prob - market_prob
            expected_roi = _expected_roi(fair_prob, market_prob)
            kelly = self.kelly_position_fraction(fair_prob, market_prob)

            evaluations.append(
                OutcomeEvaluation(
                    label=out.label,
                    market_probability=market_prob,
                    fair_probability=fair_prob,
                    edge=edge,
                    expected_roi=expected_roi,
                    kelly_fraction=kelly,
                )
            )

        evaluations.sort(key=lambda x: x.edge, reverse=True)
        best = evaluations[0] if evaluations else None
        action = self._choose_action(best, directional_signal)

        return MarketEvaluation(
            title=market.title,
            city=market.city,
            date_text=market.date_text,
            real_temp_c=real_temp_c,
            implied_temp_c=implied_temp_c,
            safety_margin_c=self.safety_margin_c,
            directional_signal=directional_signal,
            recommended_action=action,
            best_outcome=best,
            outcomes=evaluations,
        )

    def directional_signal(self, real_temp_c: float, implied_temp_c: float | None) -> str:
        if implied_temp_c is None:
            return "NO_SIGNAL"
        if real_temp_c > implied_temp_c + self.safety_margin_c:
            return "BUY_HEAT"
        if real_temp_c < implied_temp_c - self.safety_margin_c:
            return "BUY_COLD"
        return "HOLD"

    def kelly_position_fraction(self, fair_probability: float, market_probability: float) -> float:
        p = min(1.0, max(0.0, fair_probability))
        price = min(1.0, max(0.0, market_probability))
        if price <= 0 or price >= 1:
            return 0.0
        # For a $1 binary share bought at `price`, full Kelly is (p - price) / (1 - price).
        full = (p - price) / (1 - price)
        scaled = max(0.0, full) * self.kelly_fraction
        return min(self.max_kelly_fraction, scaled)

    def _choose_action(self, best: OutcomeEvaluation | None, directional_signal: str) -> str:
        if best is None:
            return "SKIP"
        if directional_signal == "HOLD":
            return "SKIP"
        if best.edge < self.min_edge_to_trade:
            return "SKIP"
        if best.kelly_fraction <= 0:
            return "SKIP"
        return f"BUY {best.label}"


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _prob_range_normal(mu: float, sigma: float, lo: float, hi: float) -> float:
    if sigma <= 0:
        return 0.0
    a = -1e9 if math.isinf(lo) and lo < 0 else lo
    b = 1e9 if math.isinf(hi) and hi > 0 else hi
    p = _normal_cdf((b - mu) / sigma) - _normal_cdf((a - mu) / sigma)
    return min(1.0, max(0.0, p))


def _expected_roi(fair_probability: float, market_probability: float) -> float:
    p = min(1.0, max(0.0, fair_probability))
    price = min(1.0, max(0.0, market_probability))
    if price <= 0:
        return 0.0
    return (p / price) - 1.0


def _f_to_c(v: float) -> float:
    return (v - 32.0) * 5.0 / 9.0


def _parse_range_to_c(label: str) -> tuple[float, float] | None:
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

