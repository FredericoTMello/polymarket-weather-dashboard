import test from "node:test";
import assert from "node:assert/strict";

import {
  convertRangeToC,
  normalizeAnalyticalLabel,
  parseRange,
  probInRange,
  resolveTargetDate,
} from "../analytics-core.mjs";

function approxEqual(actual, expected, tolerance = 1e-3) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `expected ${actual} to be within ${tolerance} of ${expected}`);
}

test("parseRange handles Celsius closed intervals", () => {
  assert.deepEqual(parseRange("12-14°C"), { lo: 12, hi: 14 });
});

test("parseRange handles Fahrenheit lower bucket", () => {
  const range = parseRange("50°F or below");
  assert.equal(range.lo, -Infinity);
  approxEqual(range.hi, 10);
});

test("parseRange handles single-value Celsius buckets", () => {
  assert.deepEqual(parseRange("18°C"), { lo: 17.5, hi: 18.5 });
});

test("convertRangeToC converts Fahrenheit labels to Celsius labels", () => {
  assert.equal(convertRangeToC("50-59°F"), "10 a 15°C");
  assert.equal(convertRangeToC("68°F or higher"), "20°C ou mais");
});

test("resolveTargetDate keeps current year for future dates", () => {
  const now = new Date("2026-03-16T12:00:00Z");
  const result = resolveTargetDate("march 20", now);

  assert.equal(result.targetDate, "2026-03-20");
  assert.equal(result.targetDateObj.toISOString().slice(0, 10), "2026-03-20");
});

test("resolveTargetDate rolls to next year for elapsed dates", () => {
  const now = new Date("2026-12-20T12:00:00Z");
  const result = resolveTargetDate("march 16", now);

  assert.equal(result.targetDate, "2027-03-16");
  assert.equal(result.targetDateObj.toISOString().slice(0, 10), "2027-03-16");
});

test("resolveTargetDate rejects unsupported labels", () => {
  assert.equal(resolveTargetDate("tomorrow"), null);
  assert.equal(resolveTargetDate(""), null);
});

test("probInRange approximates the standard normal mass between -1 and 1", () => {
  approxEqual(probInRange(0, 1, -1, 1), 0.682689, 5e-3);
});

test("probInRange handles upper tails", () => {
  approxEqual(probInRange(0, 1, 1, Infinity), 0.158655, 5e-3);
});

test("normalizeAnalyticalLabel softens legacy wording", () => {
  assert.equal(normalizeAnalyticalLabel("Modelos alinhados (Δ 1.0°C)"), "Leitura heuristica mais estavel (Δ 1.0°C)");
  assert.equal(normalizeAnalyticalLabel("Modelos divergentes (Δ 2.2°C)"), "Leitura heuristica com divergencia relevante (Δ 2.2°C)");
  assert.equal(normalizeAnalyticalLabel("Sem mercado ativo para leitura"), "Sem base suficiente para leitura de mercado");
});
