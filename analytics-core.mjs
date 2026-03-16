export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function formatTemp(value) {
  return value != null ? `${Math.round(value)}°C` : "—";
}

export function fToC(value) {
  return (value - 32) * 5 / 9;
}

export function convertRangeToC(range) {
  const text = String(range || "");
  if (!text.includes("°F") && !/\d+°?F/.test(text)) return text;

  return text
    .replace(/(-?\d+(?:\.\d+)?)\s*(?:-\s*(-?\d+(?:\.\d+)?))?\s*°?F/g, (match, t1, t2) => {
      const c1 = Math.round(fToC(parseFloat(t1)));
      if (t2) {
        const c2 = Math.round(fToC(parseFloat(t2)));
        return `${c1} a ${c2}°C`;
      }
      return `${c1}°C`;
    })
    .replace("or below", "ou menos")
    .replace("or higher", "ou mais");
}

export function parseRange(range) {
  const text = String(range || "");
  const isF = text.includes("°F") || /\d+°?F/.test(text);
  const convert = (value) => (isF ? fToC(value) : value);

  let match = text.match(/([\d.-]+)\s*°?[FC]?\s*(?:or below|ou menos)/i);
  if (match) return { lo: -Infinity, hi: convert(parseFloat(match[1])) };

  match = text.match(/([\d.-]+)\s*°?[FC]?\s*(?:or higher|ou mais)/i);
  if (match) return { lo: convert(parseFloat(match[1])), hi: Infinity };

  match = text.match(/([\d.-]+)\s*[-–]\s*([\d.-]+)/);
  if (match) return { lo: convert(parseFloat(match[1])), hi: convert(parseFloat(match[2])) };

  match = text.match(/([\d.-]+)\s*°?[FC]/);
  if (match) {
    const value = convert(parseFloat(match[1]));
    return { lo: value - 0.5, hi: value + 0.5 };
  }

  return null;
}

export function resolveTargetDate(dateLabel, now = new Date()) {
  const monthNames = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
  ];

  const parts = String(dateLabel || "").toLowerCase().split(/\s+/);
  if (parts.length !== 2) return null;

  const monthIdx = monthNames.indexOf(parts[0]);
  const dayNum = parseInt(parts[1], 10);
  if (monthIdx < 0 || !dayNum) return null;

  const currentYear = now.getFullYear();
  let targetDate = `${currentYear}-${String(monthIdx + 1).padStart(2, "0")}-${String(dayNum).padStart(2, "0")}`;
  let targetDateObj = new Date(targetDate);

  if (targetDateObj <= now) {
    targetDate = `${currentYear + 1}-${String(monthIdx + 1).padStart(2, "0")}-${String(dayNum).padStart(2, "0")}`;
    targetDateObj = new Date(targetDate);
  }

  return { targetDate, targetDateObj };
}

export function normCDF(value) {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = value < 0 ? -1 : 1;
  const normalized = Math.abs(value) / Math.SQRT2;
  const t = 1 / (1 + p * normalized);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-normalized * normalized));
  return 0.5 * (1 + sign * y);
}

export function probInRange(mu, sigma, lo, hi) {
  return normCDF((hi - mu) / sigma) - normCDF((lo - mu) / sigma);
}

export function normalizeAnalyticalLabel(message) {
  return String(message || "")
    .replace("Modelos alinhados", "Leitura heuristica mais estavel")
    .replace("Modelos divergentes", "Leitura heuristica com divergencia relevante")
    .replace("Sem mercado ativo para leitura", "Sem base suficiente para leitura de mercado");
}
