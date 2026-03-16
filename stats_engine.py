"""
Stats/arbitrage engine for Polymarket weather contracts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import config
from poly_client import MarketSnapshot, PolymarketClient


@dataclass(slots=True)
class OutcomeEvaluation:
    label: str
    market_probability: float
    fair_probability: float
    edge: float
    expected_roi: float
    kelly_fraction: float


@dataclass(slots=True)
class PackageEvaluation:
    labels: list[str]
    market_probability_sum: float
    fair_probability_sum: float
    edge_sum: float
    expected_roi: float
    suggested_weights: list[float]


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
    best_package: PackageEvaluation | None
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
        best_package = _build_three_bin_package(evaluations, real_temp_c)
        action = self._choose_action(best, directional_signal, best_package)

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
            best_package=best_package,
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
        full = (p - price) / (1 - price)
        scaled = max(0.0, full) * self.kelly_fraction
        return min(self.max_kelly_fraction, scaled)

    def _choose_action(
        self,
        best: OutcomeEvaluation | None,
        directional_signal: str,
        package: PackageEvaluation | None,
    ) -> str:
        if package and package.edge_sum >= (self.min_edge_to_trade * 1.5):
            return f"BUY PACK {' | '.join(package.labels)}"
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
    is_f = bool(re.search(r"[°º]?\s*F", label, flags=re.IGNORECASE))

    def conv(x: float) -> float:
        return _f_to_c(x) if is_f else x

    m = re.search(r"([-\d.]+)\s*[°º]?\s*[FC]?\s*(?:or below|ou menos)", label, flags=re.IGNORECASE)
    if m:
        return float("-inf"), conv(float(m.group(1)))

    m = re.search(r"([-\d.]+)\s*[°º]?\s*[FC]?\s*(?:or higher|ou mais)", label, flags=re.IGNORECASE)
    if m:
        return conv(float(m.group(1))), float("inf")

    m = re.search(r"([-\d.]+)\s*[-–]\s*([-\d.]+)", label)
    if m:
        lo = conv(float(m.group(1)))
        hi = conv(float(m.group(2)))
        return (min(lo, hi), max(lo, hi))

    m = re.search(r"([-\d.]+)\s*[°º]?\s*[FC]", label, flags=re.IGNORECASE)
    if m:
        v = conv(float(m.group(1)))
        return v - 0.5, v + 0.5

    return None


def _range_midpoint(lo: float, hi: float) -> float | None:
    if math.isinf(lo) and math.isinf(hi):
        return None
    if math.isinf(lo):
        return hi - 1.0
    if math.isinf(hi):
        return lo + 1.0
    return (lo + hi) / 2.0


def _build_three_bin_package(outcomes: list[OutcomeEvaluation], mu_c: float) -> PackageEvaluation | None:
    pts: list[tuple[float, OutcomeEvaluation]] = []
    for o in outcomes:
        parsed = _parse_range_to_c(o.label)
        if not parsed:
            continue
        mid = _range_midpoint(parsed[0], parsed[1])
        if mid is None:
            continue
        pts.append((mid, o))

    if len(pts) < 3:
        return None
    pts.sort(key=lambda x: x[0])

    best_win: list[tuple[float, OutcomeEvaluation]] | None = None
    best_score = -1e9
    for i in range(len(pts) - 2):
        w = pts[i : i + 3]
        gaps = [w[j + 1][0] - w[j][0] for j in range(2)]
        if any(g > 3.0 for g in gaps):
            continue
        fair_sum = sum(x[1].fair_probability for x in w)
        market_sum = sum(x[1].market_probability for x in w)
        edge_sum = fair_sum - market_sum
        center = sum(x[0] for x in w) / 3.0
        score = edge_sum - abs(center - mu_c) * 0.01
        if score > best_score:
            best_score = score
            best_win = w

    if not best_win:
        return None

    labels = [x[1].label for x in best_win]
    fair_sum = sum(x[1].fair_probability for x in best_win)
    market_sum = sum(x[1].market_probability for x in best_win)
    if market_sum <= 0:
        return None
    edge_sum = fair_sum - market_sum
    expected_roi = (fair_sum / market_sum) - 1.0

    pos_edges = [max(0.0, x[1].edge) for x in best_win]
    if sum(pos_edges) > 0:
        weights = [e / sum(pos_edges) for e in pos_edges]
    else:
        weights = [x[1].fair_probability for x in best_win]
        s = sum(weights)
        weights = [w / s for w in weights] if s > 0 else [1 / 3, 1 / 3, 1 / 3]

    return PackageEvaluation(
        labels=labels,
        market_probability_sum=market_sum,
        fair_probability_sum=fair_sum,
        edge_sum=edge_sum,
        expected_roi=expected_roi,
        suggested_weights=weights,
    )
