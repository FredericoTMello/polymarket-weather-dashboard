import {
  convertRangeToC,
  escapeHtml,
  fToC,
  formatTemp,
  normalizeAnalyticalLabel,
  parseRange,
  probInRange,
  resolveTargetDate,
} from "./analytics-core.mjs";

const CONFIG = {
  marketStations: [
    { id: "london", name: "London (Heathrow)", country: "United Kingdom", region: "England", lat: 51.47, lon: -0.4543 },
    { id: "new_york", name: "New York (JFK)", country: "United States", region: "New York", lat: 40.6413, lon: -73.7781 },
    { id: "tokyo", name: "Tokyo (Haneda)", country: "Japan", region: "Tokyo", lat: 35.5494, lon: 139.7798 },
  ],
  forecastModels: ["gfs_seamless", "ecmwf_ifs025", "ukmo_seamless"],
  modelLabels: { gfs_seamless: "NOAA GFS", ecmwf_ifs025: "ECMWF IFS", ukmo_seamless: "UKMO" },
  modelColors: { gfs_seamless: "#22c55e", ecmwf_ifs025: "#3b82f6", ukmo_seamless: "#f97316" },
  horizonPenaltyK: 0.05,
};

const State = {
  activeCity: null,
  activeMarket: null,
  selectedCity: null,
  selectedMarket: null,
  searchTimeout: null,
  marketSearchTimeout: null,
  isLoadingAnalysis: false,
  chart: null,
  errorCache: {},
};

const DOM = {
  marketInput: document.getElementById("marketInput"),
  marketDropdown: document.getElementById("marketDropdown"),
  marketButton: document.getElementById("marketBtn"),
  input: document.getElementById("cityInput"),
  dropdown: document.getElementById("dropdown"),
  addButton: document.getElementById("addBtn"),
  cards: document.getElementById("cards"),
  status: document.getElementById("analysisStatus"),
};

const DEFAULT_STATUS =
  "Selecione um mercado do Polymarket para abrir a analise principal. A busca por cidade continua disponivel como fallback temporario.";

const Helpers = {
  escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  },
  formatTemp(value) {
    return value != null ? `${Math.round(value)}°C` : "—";
  },
  fToC(value) {
    return (value - 32) * 5 / 9;
  },
  convertRangeToC(range) {
    if (!range.includes("°F") && !/\d+°?F/.test(range)) return range;
    return range
      .replace(/([\d.-]+)\s*(?:-\s*([\d.-]+)\s*)?°?F/g, (match, t1, t2) => {
        const c1 = Math.round(Helpers.fToC(parseFloat(t1)));
        if (t2) {
          const c2 = Math.round(Helpers.fToC(parseFloat(t2)));
          return `${c1} a ${c2}°C`;
        }
        return `${c1}°C`;
      })
      .replace("or below", "ou menos")
      .replace("or higher", "ou mais");
  },
};

const Parsing = {
  parseRange(range) {
    const isF = range.includes("°F") || /\d+°?F/.test(range);
    const convert = (value) => (isF ? Helpers.fToC(value) : value);

    let match = range.match(/([\d.-]+)\s*°?[FC]?\s*(?:or below|ou menos)/i);
    if (match) return { lo: -Infinity, hi: convert(parseFloat(match[1])) };

    match = range.match(/([\d.-]+)\s*°?[FC]?\s*(?:or higher|ou mais)/i);
    if (match) return { lo: convert(parseFloat(match[1])), hi: Infinity };

    match = range.match(/([\d.-]+)\s*[-–]\s*([\d.-]+)/);
    if (match) return { lo: convert(parseFloat(match[1])), hi: convert(parseFloat(match[2])) };

    match = range.match(/([\d.-]+)\s*°?[FC]/);
    if (match) {
      const value = convert(parseFloat(match[1]));
      return { lo: value - 0.5, hi: value + 0.5 };
    }

    return null;
  },

  resolveTargetDate(dateLabel, now = new Date()) {
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
  },
};

Helpers.escapeHtml = escapeHtml;
Helpers.formatTemp = formatTemp;
Helpers.fToC = fToC;
Helpers.convertRangeToC = convertRangeToC;
Parsing.parseRange = parseRange;
Parsing.resolveTargetDate = resolveTargetDate;

const WeatherProvider = {
  async searchCities(query) {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    const data = await response.json();
    return Array.isArray(data) ? data : data.results || [];
  },

  async fetchBundle(city) {
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth() + 1;
    const startDate = `${year - 10}-01-01`;
    const endDate = `${year}-${String(month).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;

    const multiUrl = `https://api.open-meteo.com/v1/forecast?latitude=${city.lat}&longitude=${city.lon}&daily=temperature_2m_max,temperature_2m_min&current_weather=true&timezone=auto&forecast_days=14&temperature_unit=celsius&models=${CONFIG.forecastModels.join(",")}`;
    const historicalUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${city.lat}&longitude=${city.lon}&start_date=${startDate}&end_date=${endDate}&daily=temperature_2m_mean&timezone=auto&temperature_unit=celsius`;

    const [multiRes, histRes] = await Promise.allSettled([
      fetch(multiUrl).then((response) => response.json()),
      fetch(historicalUrl).then((response) => response.json()),
    ]);

    const multi = multiRes.status === "fulfilled" ? multiRes.value : null;
    const hist = histRes.status === "fulfilled" ? histRes.value : null;
    const daily = multi?.daily || {};
    const dates = daily.time || [];
    const perModel = {};

    for (const model of CONFIG.forecastModels) {
      perModel[model] = {
        max: daily[`temperature_2m_max_${model}`] || [],
        min: daily[`temperature_2m_min_${model}`] || [],
      };
    }

    const current = multi?.current_weather?.temperature ?? null;
    const todayMaxes = [];
    const todayMins = [];

    for (const model of CONFIG.forecastModels) {
      const max = perModel[model].max[0];
      const min = perModel[model].min[0];
      if (max != null) todayMaxes.push(max);
      if (min != null) todayMins.push(min);
    }

    let historicalMean = null;
    if (hist?.daily?.time && hist?.daily?.temperature_2m_mean) {
      const temps = [];
      hist.daily.time.forEach((time, index) => {
        const date = new Date(time);
        if (date.getMonth() + 1 === month) {
          const value = hist.daily.temperature_2m_mean[index];
          if (value != null) temps.push(value);
        }
      });
      if (temps.length) historicalMean = temps.reduce((sum, value) => sum + value, 0) / temps.length;
    }

    const modelMeans = CONFIG.forecastModels
      .map((model) => {
        const max = perModel[model].max[0];
        const min = perModel[model].min[0];
        return max != null && min != null ? (max + min) / 2 : null;
      })
      .filter((value) => value != null);

    return {
      city,
      dates,
      current,
      historicalMean,
      perModel,
      modelMeans,
      avgMax: todayMaxes.length ? todayMaxes.reduce((sum, value) => sum + value, 0) / todayMaxes.length : null,
      avgMin: todayMins.length ? todayMins.reduce((sum, value) => sum + value, 0) / todayMins.length : null,
    };
  },

  async fetchModelError(city) {
    const key = `${city.lat.toFixed(2)},${city.lon.toFixed(2)}`;
    if (State.errorCache[key]) return State.errorCache[key];

    try {
      const now = new Date();
      const end = now.toISOString().slice(0, 10);
      const start = new Date(now - 30 * 86400000).toISOString().slice(0, 10);
      const actualUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${city.lat}&longitude=${city.lon}&start_date=${start}&end_date=${end}&daily=temperature_2m_max&timezone=auto`;
      const forecastUrl = `https://api.open-meteo.com/v1/forecast?latitude=${city.lat}&longitude=${city.lon}&past_days=30&daily=temperature_2m_max&timezone=auto&forecast_days=1`;

      const [actualRes, forecastRes] = await Promise.all([
        fetch(actualUrl).then((response) => response.json()),
        fetch(forecastUrl).then((response) => response.json()),
      ]);

      const actualDates = actualRes.daily?.time || [];
      const actualMax = actualRes.daily?.temperature_2m_max || [];
      const forecastDates = forecastRes.daily?.time || [];
      const forecastMax = forecastRes.daily?.temperature_2m_max || [];
      const errors = [];

      for (let index = 0; index < forecastDates.length; index += 1) {
        const actualIndex = actualDates.indexOf(forecastDates[index]);
        if (actualIndex >= 0 && actualMax[actualIndex] != null && forecastMax[index] != null) {
          errors.push(forecastMax[index] - actualMax[actualIndex]);
        }
      }

      if (errors.length < 5) {
        State.errorCache[key] = null;
        return null;
      }

      const mean = errors.reduce((sum, value) => sum + value, 0) / errors.length;
      const variance = errors.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (errors.length - 1);
      const sigma = Math.sqrt(variance);
      const result = { bias: mean, sigma, baseSigma: sigma, n: errors.length };

      State.errorCache[key] = result;
      return result;
    } catch (error) {
      console.warn("Error calc failed:", error);
      State.errorCache[key] = null;
      return null;
    }
  },
};

const MarketProvider = {
  async listMarkets({ query = "", city = "", limit = 8 } = {}) {
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    if (city) params.set("city", city);
    if (limit) params.set("limit", String(limit));
    const response = await fetch(`/api/polymarket/markets?${params.toString()}`);
    return response.json();
  },

  async fetchByCity(city) {
    const response = await fetch(`/api/polymarket/city?city=${encodeURIComponent(city.name)}`);
    return response.json();
  },
};

const AnalysisEngine = {
  normCDF(value) {
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
  },

  probInRange(mu, sigma, lo, hi) {
    return this.normCDF((hi - mu) / sigma) - this.normCDF((lo - mu) / sigma);
  },

  buildWeatherAnalysis(bundle) {
    const statsHtml = [
      `<div class="stat">Max: ${Helpers.formatTemp(bundle.avgMax)}</div>`,
      `<div class="stat">Min: ${Helpers.formatTemp(bundle.avgMin)}</div>`,
      ...CONFIG.forecastModels.map((model) => {
        const max = bundle.perModel[model].max[0];
        const min = bundle.perModel[model].min[0];
        return `<div class="stat">${CONFIG.modelLabels[model]}: ${Helpers.formatTemp(max)}/${Helpers.formatTemp(min)}</div>`;
      }),
    ].join("");

    let anomalyHtml = "";
    if (bundle.current != null && bundle.historicalMean != null) {
      const anomaly = bundle.current - bundle.historicalMean;
      const cssClass = anomaly > 0 ? "anomaly-hot" : "anomaly-cold";
      const sign = anomaly > 0 ? "+" : "";
      anomalyHtml = `<div class="anomaly-badge ${cssClass}">Anomalia: ${sign}${anomaly.toFixed(1)}°C vs media hist. (${bundle.historicalMean.toFixed(1)}°C)</div>`;
    }

    let divergenceHtml = "";
    if (bundle.modelMeans.length >= 2) {
      const spread = Math.max(...bundle.modelMeans) - Math.min(...bundle.modelMeans);
      if (spread >= 3) {
        divergenceHtml = `<div class="divergence-badge divergence-high">Modelos divergem em ${spread.toFixed(1)}°C</div>`;
      } else if (spread >= 1.5) {
        divergenceHtml = `<div class="divergence-badge divergence-high">Divergencia moderada: ${spread.toFixed(1)}°C entre modelos</div>`;
      } else {
        divergenceHtml = `<div class="divergence-badge divergence-low">Modelos alinhados (Δ ${spread.toFixed(1)}°C)</div>`;
      }
    }

    let consensusHtml = "";
    if (bundle.historicalMean != null && bundle.modelMeans.length >= 2) {
      const above = bundle.modelMeans.filter((value) => value > bundle.historicalMean).length;
      const total = bundle.modelMeans.length;
      if (above === total) {
        consensusHtml = `<div class="consensus-badge consensus-all">${total}/${total} modelos acima da media</div>`;
      } else if (above === 0) {
        consensusHtml = `<div class="consensus-badge consensus-none">${total}/${total} modelos abaixo da media</div>`;
      } else {
        consensusHtml = `<div class="consensus-badge consensus-mixed">Modelos divididos (${above}/${total} acima)</div>`;
      }
    }

    const spread = bundle.modelMeans.length >= 2 ? Math.max(...bundle.modelMeans) - Math.min(...bundle.modelMeans) : null;
    return {
      statsHtml,
      insightsHtml: `${anomalyHtml}${divergenceHtml}${consensusHtml}`,
      weatherSummary:
        bundle.historicalMean != null
          ? `Faixa prevista ${Helpers.formatTemp(bundle.avgMin)} a ${Helpers.formatTemp(bundle.avgMax)} · media hist. ${Math.round(bundle.historicalMean)}°C`
          : `Faixa prevista ${Helpers.formatTemp(bundle.avgMin)} a ${Helpers.formatTemp(bundle.avgMax)} · historico indisponivel`,
      confidenceSummary:
        spread == null
          ? "Leitura heuristica com dispersao insuficiente"
          : spread <= 1.5
            ? `Modelos alinhados (Δ ${spread.toFixed(1)}°C)`
            : `Modelos divergentes (Δ ${spread.toFixed(1)}°C)`,
      chart: {
        labels: bundle.dates.map((date) => date.slice(5)),
        datasets: [
          ...CONFIG.forecastModels.map((model) => ({
            label: CONFIG.modelLabels[model],
            data: bundle.dates.map((_, index) => {
              const max = bundle.perModel[model].max[index];
              const min = bundle.perModel[model].min[index];
              return max != null && min != null ? (max + min) / 2 : null;
            }),
            borderColor: CONFIG.modelColors[model],
            backgroundColor: "transparent",
            tension: 0.3,
            borderWidth: 2,
            spanGaps: true,
            pointRadius: 2,
          })),
          ...(bundle.historicalMean != null
            ? [
                {
                  label: "Media historica",
                  data: bundle.dates.map(() => bundle.historicalMean),
                  borderColor: "#ef4444",
                  backgroundColor: "transparent",
                  borderDash: [6, 4],
                  borderWidth: 1.5,
                  pointRadius: 0,
                  spanGaps: true,
                },
              ]
            : []),
        ],
      },
      modelData: {
        dates: bundle.dates,
        models: Object.fromEntries(CONFIG.forecastModels.map((model) => [model, { max: bundle.perModel[model].max }])),
      },
      sourcesHtml: `
        <tr><th>Fonte</th><th>Dado</th><th>Endpoint</th></tr>
        ${CONFIG.forecastModels
          .map((model) => `<tr><td class="src-name">${CONFIG.modelLabels[model]}</td><td>Previsao 14 dias</td><td>api.open-meteo.com/v1/forecast (${model})</td></tr>`)
          .join("")}
        <tr><td class="src-name">ERA5 Historical</td><td>Media mensal (ultimos 10 anos)</td><td>archive-api.open-meteo.com/v1/archive</td></tr>
        <tr><td class="src-name">Polymarket</td><td>Mercados de temperatura</td><td>polymarket.com/predictions/weather (proxy)</td></tr>
      `,
    };
  },

  buildMarketAnalysis(events, modelData, errorStats, cityName) {
    if (!events.length) {
      return {
        marketSummary: "Nenhum mercado aberto encontrado",
        confidenceSummary: "Sem base suficiente para leitura de mercado",
        html: `<div class="poly-section"><div class="poly-title">Polymarket</div><div class="poly-none">Nenhum mercado de temperatura aberto para ${Helpers.escapeHtml(cityName)}</div></div>`,
      };
    }

    const uniqueDates = [...new Set(events.map((event) => event.date).filter(Boolean))].sort();
    let html = '<div class="poly-section">';
    html += '<div class="poly-title">Polymarket - Leitura heuristica de mercado</div>';
    html += '<div class="poly-meta" style="margin-bottom:10px">Leitura experimental baseada em modelos, data inferida e timezone automatico. Nao equivale a validacao completa do SPEC.</div>';
    if (errorStats) {
      html += `<div style="font-size:11px;color:var(--muted);margin-bottom:8px;padding:4px 8px;background:rgba(255,255,255,0.02);border-radius:6px">Erro do modelo (30d): bias=${errorStats.bias > 0 ? "+" : ""}${errorStats.bias.toFixed(2)}°C · σ=${errorStats.sigma.toFixed(2)}°C · n=${errorStats.n}</div>`;
    }
    if (uniqueDates.length > 1) {
      html += '<div style="margin-bottom:10px">';
      html += '<label style="font-size:12px;color:var(--muted);margin-right:8px">Data:</label>';
      html += `<select class="date-filter" style="padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:12px"><option value="">Todas</option>${uniqueDates.map((date) => `<option value="${Helpers.escapeHtml(date)}">${Helpers.escapeHtml(date)}</option>`).join("")}</select>`;
      html += "</div>";
    }

    let confidenceSummary = "Leitura heuristica com limites atuais";

    for (const event of events) {
      const outcomes = event.outcomes || [];
      const allZero = outcomes.every((outcome) => outcome.probability < 0.01);
      const oneHundred = outcomes.some((outcome) => outcome.probability > 0.99);
      if (allZero || oneHundred) continue;

      html += `<div class="poly-market" data-date="${Helpers.escapeHtml(event.date || "")}">`;
      html += `<div class="poly-question">${Helpers.escapeHtml(event.question || "")}</div>`;

      let avgPredC = null;
      let modelSpread = null;
      let marketRuleConfidence = "LOW";
      let targetDate = null;
      const resolvedDate = Parsing.resolveTargetDate(event.date);
      if (resolvedDate) {
        targetDate = resolvedDate.targetDate;
        if (resolvedDate.targetDateObj > new Date()) {
          const index = (modelData.dates || []).indexOf(targetDate);
          if (index >= 0) {
            const preds = CONFIG.forecastModels
              .map((model) => modelData.models?.[model]?.max?.[index])
              .filter((value) => value != null);
            if (preds.length) {
              avgPredC = preds.reduce((sum, value) => sum + value, 0) / preds.length;
              modelSpread = Math.max(...preds) - Math.min(...preds);
              marketRuleConfidence = "HIGH";
            } else {
              marketRuleConfidence = "MEDIUM";
            }
          } else {
            marketRuleConfidence = "MEDIUM";
          }
        }
      }

      if (avgPredC != null && errorStats && targetDate) {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const horizonDays = Math.max(1, Math.ceil((new Date(targetDate) - today) / 86400000));
        const mu = avgPredC - errorStats.bias;
        const sigmaHorizonPenalty = errorStats.baseSigma * horizonDays * CONFIG.horizonPenaltyK;
        const sigmaFinal = Math.sqrt(errorStats.sigma ** 2 + (modelSpread || 0) ** 2 + sigmaHorizonPenalty ** 2);

        let bestEdge = -999;
        let bestOutcome = "";
        html += '<table class="edge-table"><tr><th>Temp</th><th>Prob Real</th><th>Prob Mercado</th><th>Edge</th><th>Sinal</th></tr>';
        for (const outcome of outcomes) {
          const parsed = Parsing.parseRange(outcome.label);
          if (!parsed) continue;
          const probReal = this.probInRange(mu, sigmaFinal, parsed.lo, parsed.hi);
          const probMarket = outcome.probability;
          const edge = probReal - probMarket;
          let signal = "skip";
          let rowClass = "";
          if (edge >= 0.1) {
            signal = "EDGE";
            rowClass = "edge-row-green";
          } else if (edge >= 0.05) {
            signal = "fraco";
            rowClass = "edge-row-yellow";
          } else if (edge <= -0.1) {
            signal = "contra";
            rowClass = "edge-row-red";
          }

          if (edge > bestEdge) {
            bestEdge = edge;
            bestOutcome = Helpers.convertRangeToC(outcome.label);
          }

          html += `<tr class="${rowClass}"><td>${Helpers.escapeHtml(Helpers.convertRangeToC(outcome.label))}</td><td>${(probReal * 100).toFixed(0)}%</td><td>${(probMarket * 100).toFixed(0)}%</td><td>${edge > 0 ? "+" : ""}${(edge * 100).toFixed(0)}%</td><td>${signal}</td></tr>`;
        }
        html += "</table>";

        if (marketRuleConfidence === "LOW") {
          confidenceSummary = "Leitura insuficiente: regra temporal parcial";
          html += '<div class="edge-verdict"><span class="verdict-skip">Leitura insuficiente: data estimada ou fora do horizonte atual</span></div>';
        } else if (bestEdge >= 0.1 && marketRuleConfidence === "HIGH") {
          confidenceSummary = `Sinal heuristico mais consistente (${marketRuleConfidence})`;
          html += `<div class="edge-verdict"><span class="verdict-go">Sinal heuristico: <strong>${Helpers.escapeHtml(bestOutcome)}</strong> com desvio estimado de +${(bestEdge * 100).toFixed(0)}%</span></div>`;
        } else if (bestEdge >= 0.05) {
          confidenceSummary = `Sinal heuristico fraco (${marketRuleConfidence})`;
          html += `<div class="edge-verdict"><span class="verdict-weak">Sinal heuristico fraco (+${(bestEdge * 100).toFixed(0)}%)</span></div>`;
        } else {
          confidenceSummary = `Sem desvio heuristico relevante (${marketRuleConfidence})`;
          html += '<div class="edge-verdict"><span class="verdict-skip">Sem desvio heuristico relevante na leitura atual</span></div>';
        }
      } else {
        html += '<div class="poly-outcomes">';
        outcomes.slice(0, 6).forEach((outcome, index) => {
          const cssClass = index === 0 ? "poly-outcome-top" : "poly-outcome-yes";
          html += `<div class="poly-outcome ${cssClass}">${Helpers.escapeHtml(Helpers.convertRangeToC(outcome.label))}: ${(outcome.probability * 100).toFixed(0)}%</div>`;
        });
        html += "</div>";
        if (avgPredC != null) {
          html += `<div class="poly-compare">Modelos preveem ${avgPredC.toFixed(1)}°C - sem dados suficientes para edge completo</div>`;
        }
      }

      html += "</div>";
    }

    html += "</div>";
    return {
      marketSummary: `${events.length} contrato(s) de temperatura encontrado(s)`,
      confidenceSummary,
      html,
    };
  },
};

// Keep the existing rendering pipeline, but override market rendering so the UI
// separates contract interpretation from forecast support.
AnalysisEngine.buildMarketAnalysis = function buildMarketAnalysis(events, modelData, errorStats, cityName, selectedMarket = null) {
  const selectedQuestion = String(selectedMarket?.question || "").trim().toLowerCase();
  const selectedDate = String(selectedMarket?.date || "").trim().toLowerCase();
  const selectedCityKey = String(selectedMarket?.city_key || "").trim().toLowerCase();
  const scopedEvents = selectedMarket
    ? events.filter((event) => {
        const eventQuestion = String(event.question || "").trim().toLowerCase();
        const eventDate = String(event.date || "").trim().toLowerCase();
        const eventCityKey = String(event.city_key || "").trim().toLowerCase();
        if (selectedQuestion && eventQuestion && eventQuestion === selectedQuestion) return true;
        return Boolean(selectedCityKey && selectedDate && eventCityKey === selectedCityKey && eventDate === selectedDate);
      })
    : events;
  const marketEvents = scopedEvents.length ? scopedEvents : events;

  if (!marketEvents.length) {
    return {
      marketSummary: "Nenhum mercado aberto encontrado",
      confidenceSummary: "Sem suporte meteorologico para leitura de mercado",
      html: `<div class="poly-section"><div class="poly-title">Polymarket</div><div class="poly-none">Nenhum mercado de temperatura aberto para ${Helpers.escapeHtml(cityName)}</div></div>`,
    };
  }

  const parseNoteLabels = {
    city_matched_by_alias: { label: "Cidade reconhecida por alias", css: "poly-note" },
    city_not_canonical: { label: "Cidade fora do baseline atual", css: "poly-note-warn" },
    city_not_extracted: { label: "Cidade nao extraida do titulo", css: "poly-note-risk" },
    date_inferred_from_title_only: { label: "Data inferida do titulo", css: "poly-note" },
    date_not_supported_by_current_baseline: { label: "Formato de data fora do baseline", css: "poly-note-warn" },
    date_not_extracted: { label: "Data nao extraida do titulo", css: "poly-note-risk" },
  };
  const parseStatusLabels = {
    valid: "Interpretado no baseline",
    partial: "Parcial no baseline",
    unknown: "Insuficiente no baseline",
  };
  const parseStatusBadgeClasses = {
    valid: "status-pill status-valid",
    partial: "status-pill status-partial",
    unknown: "status-pill status-unknown",
  };
  const confidenceBadge = {
    HIGH: "confidence-high",
    MEDIUM: "confidence-medium",
    LOW: "confidence-low",
  };
  const interpretationCounts = { valid: 0, partial: 0, unknown: 0 };
  let forecastReadyCount = 0;
  let forecastLimitedCount = 0;

  const uniqueDates = [...new Set(marketEvents.map((event) => event.date).filter(Boolean))].sort();
  const statusCounts = marketEvents.reduce((accumulator, event) => {
    const key = event.parse_status || "unknown";
    accumulator[key] = (accumulator[key] || 0) + 1;
    return accumulator;
  }, { valid: 0, partial: 0, unknown: 0 });
  let html = '<div class="poly-section">';
  html += '<div class="poly-title">Polymarket - Leitura heuristica de mercado</div>';
  html += '<div class="poly-meta" style="margin-bottom:10px">Leitura experimental baseada em modelos, data inferida e timezone automatico. Nao equivale a validacao completa do SPEC.</div>';
  if (selectedMarket) {
    html += `<div class="poly-meta" style="margin-bottom:8px">Contrato selecionado: ${Helpers.escapeHtml(selectedMarket.question || "mercado sem titulo")}.</div>`;
  }
  if (errorStats) {
    html += `<div style="font-size:11px;color:var(--muted);margin-bottom:8px;padding:4px 8px;background:rgba(255,255,255,0.02);border-radius:6px">Erro do modelo (30d): bias=${errorStats.bias > 0 ? "+" : ""}${errorStats.bias.toFixed(2)}°C · σ=${errorStats.sigma.toFixed(2)}°C · n=${errorStats.n}</div>`;
  }
  if (uniqueDates.length > 1) {
    html += '<div style="margin-bottom:10px">';
    html += '<label style="font-size:12px;color:var(--muted);margin-right:8px">Data:</label>';
    html += `<select class="date-filter" style="padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:12px"><option value="">Todas</option>${uniqueDates.map((date) => `<option value="${Helpers.escapeHtml(date)}">${Helpers.escapeHtml(date)}</option>`).join("")}</select>`;
    html += "</div>";
  }
  html += '<div class="poly-section-summary">';
  html += `<span class="poly-badge poly-badge-valid">${statusCounts.valid} validos</span>`;
  html += `<span class="poly-badge poly-badge-partial">${statusCounts.partial} parciais</span>`;
  html += `<span class="poly-badge poly-badge-unknown">${statusCounts.unknown} insuficientes</span>`;
  html += '<span class="poly-badge">Leitura heuristica</span>';
  html += "</div>";

  let confidenceSummary = "Suporte meteorologico ainda nao consolidado";

  for (const event of marketEvents) {
    const parseStatus = event.parse_status || "unknown";
    const parseNotes = Array.isArray(event.parse_notes) ? event.parse_notes : [];
    const ruleConfidence = event.rule_confidence || "LOW";
    interpretationCounts[parseStatus] = (interpretationCounts[parseStatus] || 0) + 1;

    const outcomes = event.outcomes || [];
    const allZero = outcomes.every((outcome) => outcome.probability < 0.01);
    const oneHundred = outcomes.some((outcome) => outcome.probability > 0.99);
    if (allZero || oneHundred) continue;

    html += `<div class="poly-market" data-date="${Helpers.escapeHtml(event.date || "")}">`;
    html += `<div class="poly-question">${Helpers.escapeHtml(event.question || "")}</div>`;

    let avgPredC = null;
    let modelSpread = null;
    let targetDate = null;
    let forecastSupportLabel = "Sem suporte meteorologico suficiente para este contrato";
    let forecastSupportClass = "support-none";
    const resolvedDate = Parsing.resolveTargetDate(event.date);
    if (resolvedDate) {
      targetDate = resolvedDate.targetDate;
      if (resolvedDate.targetDateObj > new Date()) {
        const index = (modelData.dates || []).indexOf(targetDate);
        if (index >= 0) {
          const preds = CONFIG.forecastModels
            .map((model) => modelData.models?.[model]?.max?.[index])
            .filter((value) => value != null);
          if (preds.length) {
            avgPredC = preds.reduce((sum, value) => sum + value, 0) / preds.length;
            modelSpread = Math.max(...preds) - Math.min(...preds);
            if (errorStats) {
              forecastSupportLabel = "Forecast e erro historico cobrem a data inferida";
              forecastSupportClass = "support-ready";
              forecastReadyCount += 1;
            } else {
              forecastSupportLabel = "Forecast encontrado, mas sem erro historico para esta cidade";
              forecastSupportClass = "support-limited";
              forecastLimitedCount += 1;
            }
          } else {
            forecastSupportLabel = "Data inferida sem valores de forecast suficientes";
            forecastSupportClass = "support-limited";
            forecastLimitedCount += 1;
          }
        } else {
          forecastSupportLabel = "Data inferida fora do horizonte atual de forecast";
          forecastSupportClass = "support-limited";
          forecastLimitedCount += 1;
        }
      } else {
        forecastSupportLabel = "Data inferida ja ficou fora do baseline atual";
      }
    } else {
      forecastSupportLabel = "Sem data inferida util para cruzar forecast";
    }

    html += '<div class="poly-status-grid">';
    html += `<div class="poly-status-item"><span class="poly-status-label">Estado do contrato</span><span class="poly-status-value"><span class="${parseStatusBadgeClasses[parseStatus] || parseStatusBadgeClasses.unknown}">${Helpers.escapeHtml(parseStatusLabels[parseStatus] || parseStatusLabels.unknown)}</span> <span class="confidence-badge ${confidenceBadge[ruleConfidence] || confidenceBadge.LOW}">${Helpers.escapeHtml(ruleConfidence)}</span></span></div>`;
    html += `<div class="poly-status-item"><span class="poly-status-label">Suporte meteorologico</span><span class="poly-status-value ${forecastSupportClass}">${Helpers.escapeHtml(forecastSupportLabel)}</span></div>`;
    html += "</div>";

    if (parseStatus !== "valid" && parseNotes.length) {
      html += '<div class="poly-status-notes">';
      parseNotes.forEach((note) => {
        const meta = parseNoteLabels[note] || { label: note, css: "poly-note" };
        html += `<span class="${meta.css}">${Helpers.escapeHtml(meta.label)}</span>`;
      });
      html += "</div>";
    }

    if (parseStatus === "valid" && avgPredC != null && errorStats && targetDate) {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const horizonDays = Math.max(1, Math.ceil((new Date(targetDate) - today) / 86400000));
      const mu = avgPredC - errorStats.bias;
      const sigmaHorizonPenalty = errorStats.baseSigma * horizonDays * CONFIG.horizonPenaltyK;
      const sigmaFinal = Math.sqrt(errorStats.sigma ** 2 + (modelSpread || 0) ** 2 + sigmaHorizonPenalty ** 2);

      let bestEdge = -999;
      let bestOutcome = "";
      html += '<table class="edge-table"><tr><th>Temp</th><th>Prob Real</th><th>Prob Mercado</th><th>Edge</th><th>Sinal</th></tr>';
      for (const outcome of outcomes) {
        const parsed = Parsing.parseRange(outcome.label);
        if (!parsed) continue;
        const probReal = this.probInRange(mu, sigmaFinal, parsed.lo, parsed.hi);
        const probMarket = outcome.probability;
        const edge = probReal - probMarket;
        let signal = "skip";
        let rowClass = "";
        if (edge >= 0.1) {
          signal = "EDGE";
          rowClass = "edge-row-green";
        } else if (edge >= 0.05) {
          signal = "fraco";
          rowClass = "edge-row-yellow";
        } else if (edge <= -0.1) {
          signal = "contra";
          rowClass = "edge-row-red";
        }

        if (edge > bestEdge) {
          bestEdge = edge;
          bestOutcome = Helpers.convertRangeToC(outcome.label);
        }

        html += `<tr class="${rowClass}"><td>${Helpers.escapeHtml(Helpers.convertRangeToC(outcome.label))}</td><td>${(probReal * 100).toFixed(0)}%</td><td>${(probMarket * 100).toFixed(0)}%</td><td>${edge > 0 ? "+" : ""}${(edge * 100).toFixed(0)}%</td><td>${signal}</td></tr>`;
      }
      html += "</table>";

      if (bestEdge >= 0.1 && ruleConfidence === "HIGH") {
        confidenceSummary = "Forecast cobre ao menos um contrato validado";
        html += `<div class="edge-verdict"><span class="verdict-go">Sinal heuristico: <strong>${Helpers.escapeHtml(bestOutcome)}</strong> com desvio estimado de +${(bestEdge * 100).toFixed(0)}%</span></div>`;
      } else if (bestEdge >= 0.05) {
        confidenceSummary = "Forecast pronto, mas com sinal heuristico fraco";
        html += `<div class="edge-verdict"><span class="verdict-weak">Sinal heuristico fraco (+${(bestEdge * 100).toFixed(0)}%)</span></div>`;
      } else {
        confidenceSummary = "Forecast pronto sem desvio heuristico relevante";
        html += '<div class="edge-verdict"><span class="verdict-skip">Sem desvio heuristico relevante na leitura atual</span></div>';
      }
    } else {
      html += '<div class="poly-outcomes">';
      outcomes.slice(0, 6).forEach((outcome, index) => {
        const cssClass = index === 0 ? "poly-outcome-top" : "poly-outcome-yes";
        html += `<div class="poly-outcome ${cssClass}">${Helpers.escapeHtml(Helpers.convertRangeToC(outcome.label))}: ${(outcome.probability * 100).toFixed(0)}%</div>`;
      });
      html += "</div>";
      if (parseStatus !== "valid") {
        html += '<div class="poly-compare">Leitura descritiva apenas. O contrato ainda nao sustenta uma leitura heuristica forte.</div>';
      } else if (avgPredC != null) {
        html += `<div class="poly-compare">Modelos apontam ${avgPredC.toFixed(1)}°C, mas o suporte meteorologico ainda esta incompleto.</div>`;
      } else {
        html += '<div class="poly-compare">Sem suporte meteorologico suficiente para estimar este contrato agora.</div>';
      }
    }

    html += "</div>";
  }

  html += "</div>";
  const validCount = interpretationCounts.valid || 0;
  const partialCount = interpretationCounts.partial || 0;
  const unknownCount = interpretationCounts.unknown || 0;
  if (!validCount) {
    confidenceSummary = "Sem contrato validado no baseline atual";
  } else if (forecastReadyCount > 0) {
    confidenceSummary = "Forecast cobre ao menos um contrato validado";
  } else if (forecastLimitedCount > 0) {
    confidenceSummary = "Contratos validados, mas com suporte meteorologico parcial";
  } else {
    confidenceSummary = "Contratos presentes sem suporte meteorologico suficiente";
  }

  return {
    marketSummary: selectedMarket
      ? validCount
        ? "Contrato pronto para leitura no baseline"
        : partialCount
          ? "Contrato parcial no baseline atual"
          : "Contrato ainda insuficiente no baseline"
      : `${marketEvents.length} contrato(s) · ${validCount} validos · ${partialCount} parciais · ${unknownCount} insuficientes`,
    confidenceSummary,
    html,
  };
};

const View = {
  setStatus(message) {
    if (DOM.status) DOM.status.textContent = message;
  },

  setAnalysisBusy(isBusy) {
    State.isLoadingAnalysis = isBusy;
    if (DOM.marketButton) DOM.marketButton.disabled = isBusy;
    if (DOM.addButton) DOM.addButton.disabled = isBusy;
  },

  renderDropdownMessage(dropdown, message, cssClass = "") {
    dropdown.innerHTML = `<div class="dropdown-empty ${cssClass}">${Helpers.escapeHtml(message)}</div>`;
    dropdown.style.display = "block";
  },

  getContractStatusMeta(parseStatus) {
    if (parseStatus === "valid") return "Contrato valido no baseline atual";
    if (parseStatus === "partial") return "Contrato parcial no baseline atual";
    return "Contrato insuficiente no baseline atual";
  },

  getContractStatusClass(parseStatus) {
    if (parseStatus === "valid") return "contract-status-valid";
    if (parseStatus === "partial") return "contract-status-partial";
    return "contract-status-unknown";
  },

  normalizeAnalyticalLabel(message) {
    return String(message || "")
      .replace("Modelos alinhados", "Leitura heuristica mais estavel")
      .replace("Modelos divergentes", "Leitura heuristica com divergencia relevante")
      .replace("Sem mercado ativo para leitura", "Sem base suficiente para leitura de mercado");
  },

  renderEmptyState() {
    DOM.cards.innerHTML = '<div class="empty-state">Nenhuma analise ativa. Escolha um mercado do Polymarket para abrir a leitura principal. A busca por cidade continua disponivel como fallback temporario.</div>';
  },

  renderAnalysisCard(city) {
    const market = State.activeMarket;
    const contractStatusKey = market?.parse_status || "unknown";
    const contractStatus = market ? this.getContractStatusMeta(market?.parse_status) : "Fallback por cidade";
    const contractStatusClass = this.getContractStatusClass(contractStatusKey);
    const contractLocation = [market?.city || city.name, market?.date].filter(Boolean).join(" · ");
    const contractOutcomes = (market?.outcomes || []).slice(0, 4);
    const contractHero = market
      ? `
        <div class="contract-hero">
          <div class="contract-kicker">Contrato em analise</div>
          <div class="contract-question">${Helpers.escapeHtml(market.question || "Mercado sem titulo")}</div>
          <div class="contract-meta">
            <span class="contract-pill">${Helpers.escapeHtml(contractLocation || city.name)}</span>
            <span class="contract-pill ${contractStatusClass}">${Helpers.escapeHtml(contractStatus)}</span>
            <span class="contract-pill">${Helpers.escapeHtml(market.rule_confidence || "LOW")}</span>
          </div>
          <div class="contract-outcomes">
            ${contractOutcomes.length
              ? contractOutcomes
                  .map((outcome, index) => `<span class="contract-outcome ${index === 0 ? "contract-outcome-top" : ""}">${Helpers.escapeHtml(Helpers.convertRangeToC(outcome.label))}: ${(outcome.probability * 100).toFixed(0)}%</span>`)
                  .join("")
              : '<span class="contract-pill contract-pill-muted">Outcomes ainda indisponiveis</span>'}
          </div>
        </div>
      `
      : `
        <div class="contract-hero">
          <div class="contract-kicker">Modo fallback temporario</div>
          <div class="contract-question">${Helpers.escapeHtml(city.name)}</div>
          <div class="contract-meta">
            <span class="contract-pill contract-pill-fallback">Analise aberta por cidade</span>
            <span class="contract-pill contract-pill-muted">Use a busca principal para entrar por contrato do Polymarket</span>
          </div>
        </div>
      `;

    DOM.cards.innerHTML = `
      <div class="card" id="${city.id}">
        <div class="card-header">
          <div class="card-header-left">
            <div class="panel-title">${market ? "Analise orientada por contrato" : "Analise aberta por cidade"}</div>
            <div class="city-name">${Helpers.escapeHtml(market?.question || city.name)}</div>
            <div class="city-country">${Helpers.escapeHtml([city.name, city.region, city.country].filter(Boolean).join(", "))} · ${city.lat.toFixed(2)}°, ${city.lon.toFixed(2)}°</div>
          </div>
          <div class="card-btns">
            <button class="btn-remove" id="clearAnalysisBtn" title="Limpar analise">Limpar</button>
          </div>
        </div>
        ${contractHero}
        <div class="summary-grid">
          <div class="summary-item"><span class="summary-label">Contrato ativo</span><span class="summary-value" id="market-summary-${city.id}">Buscando contexto do contrato...</span></div>
          <div class="summary-item"><span class="summary-label">Interpretacao do contrato</span><span class="summary-value" id="contract-status-${city.id}">${Helpers.escapeHtml(contractStatus)}</span></div>
          <div class="summary-item"><span class="summary-label">Clima de suporte</span><span class="summary-value" id="weather-summary-${city.id}">Compilando sinais...</span></div>
          <div class="summary-item"><span class="summary-label">Suporte meteorologico</span><span class="summary-value" id="confidence-summary-${city.id}">Aguardando sinais e limites atuais...</span></div>
        </div>
        <div class="section-block">
          <div class="panel-title">Evidencia meteorologica</div>
          <div class="stats"><div class="stat">Temperatura atual: <span id="temp-${city.id}">carregando...</span></div></div>
          <div class="stats" id="stats-${city.id}"></div>
          <div class="insights" id="insights-${city.id}"></div>
        </div>
        <div class="section-block">
          <div class="panel-title">Leitura do contrato</div>
          <div id="poly-${city.id}"><div class="loading">Carregando leitura do contrato...</div></div>
        </div>
        <div class="section-block">
          <div class="panel-title">Modelos e historico de suporte</div>
          <div class="chart-container"><canvas id="chart-${city.id}"></canvas></div>
        </div>
        <div class="section-block">
          <div class="panel-title">Fontes e cobertura</div>
          <table class="sources-table" id="sources-${city.id}">
            <tr><th>Fonte</th><th>Dado</th><th>Endpoint</th></tr>
            <tr><td colspan="3" class="loading">Buscando dados de multiplas fontes...</td></tr>
          </table>
        </div>
      </div>
    `;

    document.getElementById("clearAnalysisBtn")?.addEventListener("click", () => {
      if (State.chart) {
        State.chart.destroy();
        State.chart = null;
      }
      State.activeCity = null;
      State.activeMarket = null;
      State.selectedMarket = null;
      State.selectedCity = null;
      DOM.marketInput.value = "";
      DOM.input.value = "";
      this.renderEmptyState();
      this.setStatus(DEFAULT_STATUS);
    });
  },

  renderWeather(city, analysis) {
    document.getElementById(`temp-${city.id}`).textContent = city.currentText;
    document.getElementById(`stats-${city.id}`).innerHTML = analysis.statsHtml;
    document.getElementById(`insights-${city.id}`).innerHTML = analysis.insightsHtml;
    document.getElementById(`weather-summary-${city.id}`).textContent = analysis.weatherSummary;
    document.getElementById(`confidence-summary-${city.id}`).textContent = this.normalizeAnalyticalLabel(analysis.confidenceSummary);
    document.getElementById(`sources-${city.id}`).innerHTML = analysis.sourcesHtml;

    if (State.chart) State.chart.destroy();
    const ctx = document.getElementById(`chart-${city.id}`).getContext("2d");
    State.chart = new Chart(ctx, {
      type: "line",
      data: { labels: analysis.chart.labels, datasets: analysis.chart.datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#9ca3af", font: { size: 11 }, boxWidth: 12, padding: 8 } },
          tooltip: { backgroundColor: "#1f2937", titleColor: "#f3f4f6", bodyColor: "#d1d5db", borderColor: "#374151", borderWidth: 1 },
        },
        scales: {
          x: { ticks: { color: "#6b7280", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.03)" } },
          y: { ticks: { color: "#6b7280", font: { size: 10 }, callback: (value) => `${value}°` }, grid: { color: "rgba(255,255,255,0.05)" } },
        },
      },
    });
  },

  renderMarket(city, marketAnalysis) {
    document.getElementById(`market-summary-${city.id}`).textContent = marketAnalysis.marketSummary;
    document.getElementById(`confidence-summary-${city.id}`).textContent = this.normalizeAnalyticalLabel(marketAnalysis.confidenceSummary);
    document.getElementById(`poly-${city.id}`).innerHTML = marketAnalysis.html;

    const filter = document.querySelector(`#${city.id} .date-filter`);
    if (filter) {
      filter.addEventListener("change", (event) => {
        const selectedDate = event.target.value;
        document.querySelectorAll(`#${city.id} .poly-market`).forEach((market) => {
          market.style.display = !selectedDate || market.dataset.date === selectedDate ? "" : "none";
        });
      });
    }
  },
};

AnalysisEngine.probInRange = probInRange;
View.normalizeAnalyticalLabel = normalizeAnalyticalLabel;

async function loadCity(city) {
  const selectedMarket = State.activeMarket;
  View.setAnalysisBusy(true);
  View.renderAnalysisCard(city);
  if (selectedMarket) {
    View.setStatus(`Abrindo o contrato ${selectedMarket.question || "selecionado"} e carregando suporte meteorologico em ${city.name}...`);
  } else {
    View.setStatus(`Abrindo a analise de ${city.name} via fallback por cidade...`);
  }

  try {
    const weatherBundle = await WeatherProvider.fetchBundle(city);
    const weatherAnalysis = AnalysisEngine.buildWeatherAnalysis(weatherBundle);
    city.currentText = weatherBundle.current != null ? `${Math.round(weatherBundle.current)}°C` : "—";
    View.renderWeather(city, weatherAnalysis);
    View.setStatus(selectedMarket ? `Clima de suporte carregado. Cruzando mercado e forecast para ${city.name}...` : `Clima e historico carregados para ${city.name}. Buscando mercados relacionados...`);

    const [markets, errorStats] = await Promise.all([MarketProvider.fetchByCity(city), WeatherProvider.fetchModelError(city)]);
    const marketAnalysis = AnalysisEngine.buildMarketAnalysis(markets, weatherAnalysis.modelData, errorStats, city.name, selectedMarket);
    View.renderMarket(city, marketAnalysis);
    View.setStatus(selectedMarket ? `Analise pronta para o contrato ${selectedMarket.question || "selecionado"}.` : `Analise pronta para ${city.name}.`);
  } catch (error) {
    console.error("Error loading city data:", error);
    const tempEl = document.getElementById(`temp-${city.id}`);
    if (tempEl) tempEl.textContent = "Erro";
    const weatherSummaryEl = document.getElementById(`weather-summary-${city.id}`);
    const confidenceSummaryEl = document.getElementById(`confidence-summary-${city.id}`);
    if (weatherSummaryEl) weatherSummaryEl.textContent = "Falha ao carregar previsoes e historico.";
    if (confidenceSummaryEl) confidenceSummaryEl.textContent = "Leitura indisponivel.";
    const polyEl = document.getElementById(`poly-${city.id}`);
    if (polyEl) polyEl.innerHTML = '<div class="poly-none">Falha ao carregar a leitura do contrato.</div>';
    View.setStatus(`Falha ao carregar a analise de ${city.name}.`);
  } finally {
    View.setAnalysisBusy(false);
  }
}

function setActiveCity(baseCity) {
  const city = { ...baseCity, id: `c${Date.now()}` };
  State.activeCity = city;
  State.activeMarket = null;
  State.selectedCity = null;
  State.selectedMarket = null;
  DOM.marketInput.value = "";
  DOM.input.value = "";
  loadCity(city);
}

function formatMarketLocation(market) {
  const pieces = [market.city, market.date].filter(Boolean);
  return pieces.join(" · ") || "Contrato sem cidade/data canônicas";
}

function getCityFromStations(cityKey) {
  if (!cityKey) return null;
  const station = CONFIG.marketStations.find((item) => item.id === cityKey);
  return station ? { ...station } : null;
}

async function resolveCityForMarket(market) {
  const stationMatch = getCityFromStations(market.city_key);
  if (stationMatch) return stationMatch;

  const query = market.city || market.city_key || "";
  if (!query) return null;

  const results = await WeatherProvider.searchCities(query);
  const first = results[0];
  if (!first) return null;

  return {
    id: market.city_key || `market-${Date.now()}`,
    name: first.name,
    country: first.country || "",
    region: first.admin1 || "",
    lat: first.latitude,
    lon: first.longitude,
  };
}

function renderMarketSuggestions(markets) {
  if (!markets.length) {
    DOM.marketDropdown.style.display = "none";
    return;
  }

  DOM.marketDropdown.innerHTML = "";
  markets.forEach((market) => {
    const item = document.createElement("div");
    item.className = "dropdown-item";
    const location = formatMarketLocation(market);
    item.innerHTML = `<strong>${Helpers.escapeHtml(market.question || "Mercado sem titulo")}</strong> <small>${Helpers.escapeHtml(location)}</small>`;
    item.addEventListener("click", async () => {
      State.selectedMarket = market;
      DOM.marketInput.value = market.question || market.city || "";
      DOM.marketDropdown.style.display = "none";
      View.setStatus(`Abrindo o contrato ${market.question || "selecionado"}...`);
      try {
        await setActiveMarket(market);
      } catch (error) {
        console.error("Error activating market from suggestion:", error);
        View.setStatus("Falha ao carregar o mercado selecionado.");
      }
    });
    DOM.marketDropdown.appendChild(item);
  });
  DOM.marketDropdown.style.display = "block";
}

async function fetchMarkets(query = "") {
  View.renderDropdownMessage(DOM.marketDropdown, query ? "Buscando mercados..." : "Carregando mercados em destaque...", "dropdown-loading");
  try {
    const results = await MarketProvider.listMarkets({ query, limit: 8 });
    const items = Array.isArray(results) ? results : [];
    if (!items.length) {
      View.renderDropdownMessage(DOM.marketDropdown, "Nenhum mercado encontrado para essa busca.");
      return [];
    }
    renderMarketSuggestions(items);
    return items;
  } catch (error) {
    console.error("Market search error:", error);
    View.renderDropdownMessage(DOM.marketDropdown, "Falha ao buscar mercados agora. Tente novamente.");
    return [];
  }
}

async function fetchCities(query) {
  View.renderDropdownMessage(DOM.dropdown, "Buscando cidades...", "dropdown-loading");
  try {
    const results = await WeatherProvider.searchCities(query);
    if (!results.length) {
      View.renderDropdownMessage(DOM.dropdown, "Nenhuma cidade encontrada para essa busca.");
      return;
    }
    DOM.dropdown.innerHTML = "";
    results.forEach((result) => {
      const item = document.createElement("div");
      item.className = "dropdown-item";
      const region = [result.admin1, result.country].filter(Boolean).join(", ");
      item.innerHTML = `<strong>${Helpers.escapeHtml(result.name)}</strong> <small>${Helpers.escapeHtml(region)} (${result.latitude.toFixed(2)}°, ${result.longitude.toFixed(2)}°)</small>`;
      item.addEventListener("click", () => {
        State.selectedCity = {
          name: result.name,
          country: result.country || "",
          region: result.admin1 || "",
          lat: result.latitude,
          lon: result.longitude,
        };
        DOM.input.value = `${result.name}, ${region}`;
        DOM.dropdown.style.display = "none";
      });
      DOM.dropdown.appendChild(item);
    });
    DOM.dropdown.style.display = "block";
  } catch (error) {
    console.error("Geocoding error:", error);
    View.renderDropdownMessage(DOM.dropdown, "Falha ao buscar cidades agora. Tente novamente.");
  }
}

async function setActiveMarket(market) {
  View.setAnalysisBusy(true);
  const city = await resolveCityForMarket(market);
  if (!city) {
    View.setAnalysisBusy(false);
    View.setStatus("Nao foi possivel resolver a cidade base do contrato selecionado.");
    return;
  }

  const resolvedCity = { ...city, id: `c${Date.now()}` };
  State.activeCity = resolvedCity;
  State.activeMarket = market;
  State.selectedMarket = null;
  State.selectedCity = null;
  DOM.marketInput.value = market.question || "";
  DOM.input.value = "";
  DOM.marketDropdown.style.display = "none";
  DOM.dropdown.style.display = "none";
  loadCity(resolvedCity);
}

function addCity() {
  if (!State.selectedCity) {
    alert("Selecione uma cidade da lista de sugestoes.");
    return;
  }
  if (State.activeCity && !State.activeMarket && State.activeCity.name === State.selectedCity.name && State.activeCity.lat === State.selectedCity.lat) {
    View.setStatus(`A analise atual ja esta focada em ${State.selectedCity.name}.`);
    State.selectedCity = null;
    DOM.input.value = "";
    return;
  }
  setActiveCity(State.selectedCity);
}

async function addMarket() {
  if (!State.selectedMarket) {
    const firstItem = DOM.marketDropdown.querySelector(".dropdown-item");
    if (firstItem) {
      firstItem.click();
      return;
    }

    const query = DOM.marketInput.value.trim();
    if (query) {
      const results = await fetchMarkets(query);
      if (results.length) {
        const refreshedFirstItem = DOM.marketDropdown.querySelector(".dropdown-item");
        if (refreshedFirstItem) {
          refreshedFirstItem.click();
          return;
        }
      }
    }

    alert("Selecione um mercado da lista de sugestoes.");
    return;
  }

  const nextMarket = State.selectedMarket;
  if (State.activeMarket && State.activeMarket.question === nextMarket.question) {
    View.setStatus(`O contrato ${nextMarket.question} ja esta em analise.`);
    State.selectedMarket = null;
    DOM.marketInput.value = "";
    return;
  }

  await setActiveMarket(nextMarket);
}

function bindEvents() {
  DOM.marketInput.addEventListener("input", () => {
    const query = DOM.marketInput.value.trim();
    State.selectedMarket = null;
    if (State.marketSearchTimeout) clearTimeout(State.marketSearchTimeout);
    if (query.length < 2) {
      DOM.marketDropdown.style.display = "none";
      return;
    }
    State.marketSearchTimeout = setTimeout(() => fetchMarkets(query), 250);
  });

  DOM.marketInput.addEventListener("focus", () => {
    const query = DOM.marketInput.value.trim();
    if (query.length >= 2) {
      fetchMarkets(query);
      return;
    }
    fetchMarkets("");
  });

  DOM.marketInput.addEventListener("click", () => {
    const query = DOM.marketInput.value.trim();
    if (query.length >= 2) {
      fetchMarkets(query);
      return;
    }
    fetchMarkets("");
  });

  DOM.marketInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      DOM.marketDropdown.style.display = "none";
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      if (State.selectedMarket) {
        addMarket().catch((error) => {
          console.error("Error activating market with Enter:", error);
          View.setStatus("Falha ao carregar o mercado selecionado.");
        });
        return;
      }

      const firstItem = DOM.marketDropdown.querySelector(".dropdown-item");
      if (firstItem) {
        firstItem.click();
        return;
      }

      fetchMarkets(DOM.marketInput.value.trim());
    }
  });

  DOM.input.addEventListener("input", () => {
    const query = DOM.input.value.trim();
    State.selectedCity = null;
    if (State.searchTimeout) clearTimeout(State.searchTimeout);
    if (query.length < 2) {
      DOM.dropdown.style.display = "none";
      return;
    }
    State.searchTimeout = setTimeout(() => fetchCities(query), 300);
  });

  DOM.input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") DOM.dropdown.style.display = "none";
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-wrapper")) {
      DOM.dropdown.style.display = "none";
      DOM.marketDropdown.style.display = "none";
    }
  });

  DOM.marketButton.addEventListener("click", () => {
    addMarket().catch((error) => {
      console.error("Error activating market:", error);
      View.setStatus("Falha ao carregar o mercado selecionado.");
    });
  });
  DOM.addButton.addEventListener("click", addCity);
}

function init() {
  bindEvents();
  View.renderEmptyState();
  View.setStatus(DEFAULT_STATUS);
}

init();
