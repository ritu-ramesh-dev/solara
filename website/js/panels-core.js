// "Your panels": the logic behind the page, with no DOM so it can be tested.
// Photos -> per-photo loss readings -> daily soiling trend -> Module 3's
// cost-benefit rule over a real rain and sunshine forecast -> a wash date.

export const LIMITS = {
  minPhotos: 4,
  maxPhotos: 40,
  minDays: 3,             // a soiling trend needs readings from at least 3 days
  minSide: 192,           // the training photographs were 192 x 192 pixels
  maxBytes: 20 * 1024 * 1024,
  types: ["image/jpeg", "image/png", "image/webp"],
  staleDays: 14,          // newest photo older than this: warn, project forward
};

// Module 3 settings (module3_scheduling/config.py and its tuned horizon).
export const SOILING = {
  residualPct: 0.3,       // dust left after a manual wash
  defaultRatePct: 0.4,    // Indian field average on dry days
  rateBounds: [0.05, 1.5],
  maxLossPct: 60,
  rainFullMm: 5, rainFullClean: 0.9,
  rainPartialMm: 1, rainPartialClean: 0.4,
  horizonDays: 13,        // Solara's tuned look-ahead at the five parks
  forecastDays: 3,        // the rule may use a 3-day rain forecast
  performanceRatio: 0.8,
  resetDropPct: 3,        // a fall this large between days means a wash or rain
};

// Sky when the photos were taken -> DeepSolarEye's normalised irradiance.
export const SKY = { bright: 0.55, hazy: 0.35, overcast: 0.15 };

export const DAY_MS = 86400000;
export const dayKey = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
export const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate());
export const daysBetween = (a, b) => Math.round((startOfDay(b) - startOfDay(a)) / DAY_MS);

// ---- file checks ------------------------------------------------------------------
export function checkFile(file) {
  if (!LIMITS.types.includes(file.type)) return `${file.name}: only JPG, PNG or WebP photos can be read.`;
  if (file.size > LIMITS.maxBytes) return `${file.name}: larger than 20 MB.`;
  return null;
}

// Brightness and sharpness from a small greyscale copy of the photo.
export function imageQuality(rgba, w, h) {
  const g = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0; i < w * h; i++) {
    g[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
    sum += g[i];
  }
  const mean = sum / (w * h);
  let lap = 0, lap2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
      lap += v; lap2 += v * v; n++;
    }
  }
  const sharpness = lap2 / n - (lap / n) ** 2;
  const issues = [];
  if (mean < 25) issues.push("too dark");
  if (mean > 235) issues.push("washed out");
  return { brightness: mean, sharpness, blurry: sharpness < 30, rejected: issues.length > 0, issues };
}

// ---- dates: EXIF DateTimeOriginal from a JPEG, else the file's own date -------------
export function exifDate(buffer) {
  const v = new DataView(buffer);
  if (v.byteLength < 4 || v.getUint16(0) !== 0xffd8) return null;
  let off = 2;
  while (off + 4 < v.byteLength) {
    const marker = v.getUint16(off);
    const len = v.getUint16(off + 2);
    if (marker === 0xffe1 && v.getUint32(off + 4) === 0x45786966) return parseTiff(v, off + 10);
    if ((marker & 0xff00) !== 0xff00) break;
    off += 2 + len;
  }
  return null;
}

function parseTiff(v, t) {
  const le = v.getUint16(t) === 0x4949;
  const u16 = (o) => v.getUint16(t + o, le);
  const u32 = (o) => v.getUint32(t + o, le);
  const readIfd = (o) => {
    const tags = {};
    const n = u16(o);
    for (let i = 0; i < n; i++) {
      const e = o + 2 + i * 12;
      tags[u16(e)] = { type: u16(e + 2), count: u32(e + 4), value: e + 8 };
    }
    return tags;
  };
  const ascii = (tag) => {
    if (!tag || tag.type !== 2) return null;
    const at = tag.count > 4 ? u32(tag.value) : tag.value;
    let s = "";
    for (let i = 0; i < tag.count - 1; i++) s += String.fromCharCode(v.getUint8(t + at + i));
    return s;
  };
  try {
    const ifd0 = readIfd(u32(4));
    let s = null;
    if (ifd0[0x8769]) {
      const exif = readIfd(u32(ifd0[0x8769].value));   // pointer to the Exif sub-IFD
      s = ascii(exif[0x9003]) || ascii(exif[0x9004]);
    }
    s = s || ascii(ifd0[0x0132]);
    const m = s && s.match(/^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})/);
    if (!m) return null;
    const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
    return isNaN(d) ? null : d;
  } catch {
    return null;
  }
}

// ---- inputs for the power-loss network ------------------------------------------------
export function envVector(date, sky) {
  const frac = (date.getHours() + date.getMinutes() / 60) / 24;
  return [SKY[sky] ?? SKY.bright, Math.sin(2 * Math.PI * frac), Math.cos(2 * Math.PI * frac)];
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

export function severity(lossPct, bins = [5, 15, 30]) {
  return bins.filter((b) => lossPct >= b).length;
}

// ---- from photos to a soiling trend ----------------------------------------------------
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// One reading per day: the median of that day's photos.
export function dailyReadings(photos) {
  const by = new Map();
  for (const p of photos) {
    const k = dayKey(p.date);
    if (!by.has(k)) by.set(k, { date: startOfDay(p.date), values: [] });
    by.get(k).values.push(p.lossPct);
  }
  return [...by.values()]
    .map((d) => ({ date: d.date, lossPct: median(d.values), photos: d.values.length }))
    .sort((a, b) => a.date - b.date);
}

// Theil-Sen slope: the median of all pairwise slopes, robust to a bad photo.
export function theilSen(points) {
  const slopes = [];
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++)
      if (points[j].x !== points[i].x) slopes.push((points[j].y - points[i].y) / (points[j].x - points[i].x));
  if (!slopes.length) return null;
  const slope = median(slopes);
  const intercept = median(points.map((p) => p.y - slope * p.x));
  return { slope, intercept };
}

/**
 * Soiling trend from the daily readings.
 * Dust only builds up between cleanings, so the fit uses the readings after
 * the last sharp drop (a wash or heavy rain), or after a wash date the user gives.
 */
export function soilingTrend(days, { lastWash = null } = {}) {
  let start = 0;
  for (let i = 1; i < days.length; i++) if (days[i - 1].lossPct - days[i].lossPct > SOILING.resetDropPct) start = i;
  if (lastWash) {
    const i = days.findIndex((d) => d.date >= startOfDay(lastWash));
    if (i > start) start = i;
  }
  const seg = days.slice(start);
  const t0 = seg[0].date;
  const pts = seg.map((d) => ({ x: daysBetween(t0, d.date), y: d.lossPct }));
  if (lastWash && startOfDay(lastWash) <= t0) pts.unshift({ x: daysBetween(t0, startOfDay(lastWash)), y: SOILING.residualPct });

  let rate = SOILING.defaultRatePct, source = "default";
  const fit = pts.length >= 2 ? theilSen(pts) : null;
  if (fit && fit.slope > 0) {
    rate = Math.min(Math.max(fit.slope, SOILING.rateBounds[0]), SOILING.rateBounds[1]);
    source = fit.slope > SOILING.rateBounds[1] ? "capped" : "photos";
  }
  const last = seg[seg.length - 1];
  const fitted = fit && fit.slope > 0 ? fit.intercept + fit.slope * daysBetween(t0, last.date) : last.lossPct;
  // Trust the fitted line, but never stray more than 3 points from the newest reading.
  const current = Math.max(0, Math.min(Math.max(fitted, last.lossPct - 3), last.lossPct + 3));
  return { rate, rateSource: source, currentPct: current, asOf: last.date, segmentStart: t0, segmentDays: seg.length, fit, t0 };
}

// ---- weather ---------------------------------------------------------------------------------
export function openMeteoUrl(lat, lon) {
  const q = new URLSearchParams({
    latitude: lat.toFixed(3), longitude: lon.toFixed(3),
    daily: "precipitation_sum,shortwave_radiation_sum", forecast_days: "16", timezone: "auto",
  });
  return `https://api.open-meteo.com/v1/forecast?${q}`;
}

// Open-Meteo daily JSON -> [{date, rainMm, ghi}] with ghi in kWh/m2/day.
export function parseForecast(json) {
  const d = json.daily;
  return d.time.map((t, i) => ({
    date: new Date(`${t}T00:00:00`),
    rainMm: d.precipitation_sum[i] ?? 0,
    ghi: (d.shortwave_radiation_sum[i] ?? 18) / 3.6,
  }));
}

// ---- the schedule: Module 3's Solara rule, run forward from today ---------------------------
function rainClean(mm) {
  if (mm >= SOILING.rainFullMm) return SOILING.rainFullClean;
  if (mm >= SOILING.rainPartialMm) return SOILING.rainPartialClean;
  return 0;
}

/**
 * Plan the next `days` days.
 *   startLossPct  soiling loss today
 *   ratePct       dust added per dry day
 *   weather       daily [{date, rainMm, ghi}] (forecast, then climate filler)
 *   plant         {kwp, tariff, washCost}
 *   policy        "solara" | "every15" | "never"
 */
export function simulate({ startLossPct, ratePct, weather, plant, policy = "solara", days = 60, today }) {
  const { kwp, tariff, washCost } = plant;
  // Today counts as a possible wash day for Solara; the calendar starts its count today.
  let loss = startLossPct, sinceWash = policy === "solara" ? 1 : 0;
  const out = [];
  let lostKwh = 0, washes = 0, firstWash = null;
  const meanEnergy = weather.slice(0, 30).reduce((a, w) => a + w.ghi, 0) / Math.min(30, weather.length) * kwp * SOILING.performanceRatio;
  for (let t = 0; t < days; t++) {
    const w = weather[Math.min(t, weather.length - 1)];
    const energy = w.ghi * kwp * SOILING.performanceRatio;
    let wash = false;
    if (policy === "every15") wash = sinceWash >= 14;
    if (policy === "solara" && sinceWash >= 1) {
      const ahead = weather.slice(t, t + SOILING.forecastDays).map((x) => x.rainMm);
      const heavy = ahead.findIndex((mm) => mm >= SOILING.rainFullMm);
      const useful = heavy >= 0 ? Math.min(SOILING.horizonDays, heavy) : SOILING.horizonDays;
      const value = (loss - SOILING.residualPct) / 100 * meanEnergy * tariff * useful;
      wash = value >= washCost;
    }
    if (wash) {
      loss = SOILING.residualPct; washes++; sinceWash = 0;
      if (!firstWash) firstWash = new Date(today.getTime() + t * DAY_MS);
    } else sinceWash++;
    const lost = energy * loss / 100;
    lostKwh += lost;
    out.push({ date: new Date(today.getTime() + t * DAY_MS), lossPct: loss, wash, rainMm: w.rainMm, lostKwh: lost });
    const dry = w.rainMm < SOILING.rainPartialMm;
    loss = Math.min(loss + (dry ? ratePct : 0), SOILING.maxLossPct) * (1 - rainClean(w.rainMm));
  }
  return {
    days: out, washes, firstWash, lostKwh,
    lostInr: lostKwh * tariff, washInr: washes * washCost, totalInr: lostKwh * tariff + washes * washCost,
  };
}

// Forecast for the first days, then the forecast's average sunshine and no rain.
export function extendWeather(forecast, days, fallbackGhi = 5.2) {
  const ghi = forecast.length ? forecast.reduce((a, w) => a + w.ghi, 0) / forecast.length : fallbackGhi;
  const out = forecast.slice(0, days).map((w) => ({ ...w, source: "forecast" }));
  while (out.length < days) out.push({ rainMm: 0, ghi, source: "assumed" });
  return out;
}

/** Everything the results panel shows, from the per-photo readings. */
export function plan({ photos, plant, forecast, today = new Date(), lastWash = null, days = 60 }) {
  const daily = dailyReadings(photos);
  const trend = soilingTrend(daily, { lastWash });
  const gap = Math.max(0, daysBetween(trend.asOf, today));
  // Bring the newest reading up to today, assuming dry days since then.
  const nowPct = Math.min(trend.currentPct + gap * trend.rate, SOILING.maxLossPct);
  const weather = extendWeather(forecast || [], days);
  const base = { startLossPct: nowPct, ratePct: trend.rate, weather, plant, days, today: startOfDay(today) };
  const solara = simulate({ ...base, policy: "solara" });
  const every15 = simulate({ ...base, policy: "every15" });
  const never = simulate({ ...base, policy: "never" });
  const energyToday = weather[0].ghi * plant.kwp * SOILING.performanceRatio;
  return {
    daily, trend, gapDays: gap, stale: gap > LIMITS.staleDays, nowPct,
    lostTodayKwh: energyToday * nowPct / 100, lostTodayInr: energyToday * nowPct / 100 * plant.tariff,
    solara, every15, never, weather,
    nextWash: solara.firstWash,
    rainSoon: weather.slice(0, 16).find((w) => w.rainMm >= SOILING.rainFullMm) || null,
  };
}

/** Do the photos meet the minimums? Returns a list of problems (empty = ready). */
export function readiness(photos) {
  const ok = photos.filter((p) => !p.rejected);
  const problems = [];
  if (ok.length < LIMITS.minPhotos) problems.push(`Add at least ${LIMITS.minPhotos} usable photos (you have ${ok.length}).`);
  if (ok.length > LIMITS.maxPhotos) problems.push(`Use at most ${LIMITS.maxPhotos} photos.`);
  const days = new Set(ok.map((p) => dayKey(p.date))).size;
  if (days < LIMITS.minDays) problems.push(`Photos must come from at least ${LIMITS.minDays} different days (you have ${days}).`);
  return problems;
}
