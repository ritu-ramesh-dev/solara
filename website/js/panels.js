// "Your panels" page: photographs in, a cleaning schedule out.
// The two networks (Module 1 classifier, Module 2 power-loss CNN) run in the
// browser with ONNX Runtime Web, so photographs never leave the device.
import * as core from "./panels-core.js";
import { inr, loadJSON, observeReveals } from "./site.js";

const ORT_VERSION = "1.19.2";
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const $ = (id) => document.getElementById(id);

const PLACES = [
  ["Solar parks", [
    ["Bhadla Solar Park, Rajasthan", 27.54, 71.92], ["Charanka Solar Park, Gujarat", 23.9, 71.18],
    ["Rewa Ultra Mega Solar, Madhya Pradesh", 24.52, 81.53], ["Kurnool Ultra Mega Solar Park, Andhra Pradesh", 15.6, 78.3],
    ["Pavagada Solar Park, Karnataka", 14.25, 77.43]]],
  ["Cities", [
    ["Ahmedabad", 23.02, 72.57], ["Bengaluru", 12.97, 77.59], ["Chennai", 13.08, 80.27], ["Coimbatore", 11.02, 76.96],
    ["Hyderabad", 17.39, 78.49], ["Jaipur", 26.91, 75.79], ["Kolkata", 22.57, 88.36], ["Mumbai", 19.08, 72.88],
    ["New Delhi", 28.61, 77.21], ["Pune", 18.52, 73.86]]],
];
const PRESETS = {
  rooftop: { kwp: 5, tariff: 6, wash: 400 },
  commercial: { kwp: 100, tariff: 7, wash: 3000 },
  utility: { kwp: 1000, tariff: 2.6, wash: 9500 },
};

const state = { photos: [], demo: false, results: null };
let nextId = 1;

// ---- form setup -----------------------------------------------------------------------
const place = $("place");
for (const [group, items] of PLACES) {
  const og = document.createElement("optgroup");
  og.label = group;
  for (const [name, lat, lon] of items) og.append(new Option(name, `${lat},${lon}`));
  place.append(og);
}
place.append(new Option("Somewhere else: enter coordinates", "custom"));
place.value = "26.91,75.79";
place.addEventListener("change", () => { $("coords").hidden = place.value !== "custom"; });

function applyPreset() {
  const p = PRESETS[$("preset").value];
  $("kwp").value = p.kwp; $("tariff").value = p.tariff; $("wash").value = p.wash;
}
$("preset").addEventListener("change", applyPreset);
applyPreset();

$("locate").addEventListener("click", () => {
  if (!navigator.geolocation) return say("run-status", "This browser cannot share its location; type the coordinates instead.");
  navigator.geolocation.getCurrentPosition(
    (pos) => { $("lat").value = pos.coords.latitude.toFixed(2); $("lon").value = pos.coords.longitude.toFixed(2); },
    () => say("run-status", "Location was not shared; type the coordinates instead."),
    { timeout: 10000 },
  );
});

const say = (id, text) => { $(id).textContent = text; };

// ---- adding photographs -------------------------------------------------------------------
$("pick").addEventListener("click", () => $("files").click());
$("files").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addFiles([...e.dataTransfer.files]));
$("clear").addEventListener("click", () => { state.photos.forEach((p) => URL.revokeObjectURL(p.url)); state.photos = []; state.demo = false; refresh(); });

// Centre-crop to a square, as the training photographs were square.
async function squareCrop(bitmap, size) {
  const s = Math.min(bitmap.width, bitmap.height);
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, (bitmap.width - s) / 2, (bitmap.height - s) / 2, s, s, 0, 0, size, size);
  return c;
}

async function readPhoto(file, { date = null, irradiance = null } = {}) {
  const problem = core.checkFile(file);
  if (problem) return { error: problem };
  if (state.photos.some((p) => p.file.name === file.name && p.file.size === file.size)) return { error: `${file.name}: already added.` };
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    return { error: `${file.name}: this image could not be opened.` };
  }
  if (Math.min(bitmap.width, bitmap.height) < core.LIMITS.minSide) {
    return { error: `${file.name}: ${bitmap.width} × ${bitmap.height} pixels is too small (the short side needs at least ${core.LIMITS.minSide}).` };
  }
  const small = await squareCrop(bitmap, 64);
  const quality = core.imageQuality(small.getContext("2d").getImageData(0, 0, 64, 64).data, 64, 64);
  const input = await squareCrop(bitmap, 224);
  bitmap.close();
  if (!date && file.type === "image/jpeg") date = core.exifDate(await file.arrayBuffer());
  const dateSource = date ? "camera" : "file";
  if (!date) date = new Date(file.lastModified || Date.now());
  return {
    id: nextId++, file, url: URL.createObjectURL(file), input, date, dateSource, irradiance,
    quality, rejected: quality.rejected, reading: null,
  };
}

async function addFiles(files, opts = []) {
  if (!files.length) return;
  const room = core.LIMITS.maxPhotos - state.photos.length;
  const errors = [];
  if (files.length > room) errors.push(`Only ${core.LIMITS.maxPhotos} photographs can be used; ${files.length - Math.max(room, 0)} were left out.`);
  say("photo-status", `Reading ${Math.min(files.length, room)} photograph${files.length === 1 ? "" : "s"}…`);
  for (const [i, f] of files.slice(0, Math.max(room, 0)).entries()) {
    const r = await readPhoto(f, opts[i]);
    if (r.error) errors.push(r.error);
    else state.photos.push(r);
  }
  state.results = null;
  refresh(errors);
}

// Real DeepSolarEye photographs from three days, re-dated to the last three days.
$("demo").addEventListener("click", async () => {
  const d = await loadJSON("data/demo_panels.json");
  state.photos.forEach((p) => URL.revokeObjectURL(p.url));
  state.photos = [];
  const lastDay = new Date(d.photos[d.photos.length - 1].taken).getDate();
  const today = new Date();
  const files = [], opts = [];
  for (const p of d.photos) {
    const blob = await fetch(p.src).then((r) => r.blob());
    files.push(new File([blob], p.src.split("/").pop(), { type: "image/jpeg", lastModified: Date.now() }));
    const t = new Date(p.taken);
    const date = new Date(today.getFullYear(), today.getMonth(), today.getDate() - (lastDay - t.getDate()), t.getHours(), t.getMinutes());
    opts.push({ date, irradiance: p.irradiance });
  }
  state.demo = true;
  $("sky").value = "bright";
  await addFiles(files, opts);
  if (state.photos.length) say("photo-status", `Sample: ${state.photos.length} real DeepSolarEye photographs taken on 25, 26 and 27 June 2017, re-dated to the last three days so the plan uses today's forecast.`);
});

// ---- the thumbnail list ---------------------------------------------------------------------
const toLocalInput = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
const fmt = (d, opts) => d.toLocaleString("en-IN", opts);

function refresh(errors = []) {
  const list = $("thumbs");
  list.innerHTML = "";
  const sorted = [...state.photos].sort((a, b) => a.date - b.date);
  for (const p of sorted) {
    const li = document.createElement("li");
    li.className = "thumb" + (p.rejected ? " rejected" : "");
    const flags = [];
    if (p.rejected) flags.push(`Not used: ${p.quality.issues.join(", ")}`);
    else if (p.quality.blurry) flags.push("Looks blurred");
    if (p.dateSource === "file") flags.push("No camera date: check it");
    li.innerHTML = `
      <img src="${p.url}" alt="" width="120" height="120">
      <label><span class="visually-hidden">Date and time taken</span><input type="datetime-local" value="${toLocalInput(p.date)}"></label>
      ${p.reading ? `<span class="thumb-reading">${p.reading.lossPct.toFixed(1)}% lost · ${p.reading.className}</span>` : ""}
      ${flags.length ? `<span class="thumb-flag">${flags.join(". ")}</span>` : ""}
      <button type="button" class="thumb-remove" aria-label="Remove ${p.file.name}">Remove</button>`;
    li.querySelector("input").addEventListener("change", (e) => {
      const d = new Date(e.target.value);
      if (!isNaN(d)) { p.date = d; p.dateSource = "edited"; state.results = null; refresh(); }
    });
    li.querySelector(".thumb-remove").addEventListener("click", () => {
      URL.revokeObjectURL(p.url);
      state.photos = state.photos.filter((x) => x !== p);
      state.results = null;
      refresh();
    });
    list.append(li);
  }
  const problems = core.readiness(state.photos);
  const usable = state.photos.filter((p) => !p.rejected).length;
  const days = new Set(state.photos.filter((p) => !p.rejected).map((p) => core.dayKey(p.date))).size;
  if (!state.demo || errors.length) {
    say("photo-status", [...errors, state.photos.length ? `${usable} usable photograph${usable === 1 ? "" : "s"} from ${days} day${days === 1 ? "" : "s"}.` : ""].filter(Boolean).join(" "));
  }
  $("clear").hidden = !state.photos.length;
  $("run").disabled = problems.length > 0;
  say("run-status", problems.length ? problems.join(" ") : "Ready.");
}

// ---- running the models --------------------------------------------------------------------
let sessions = null;
function loadScript(src) {
  return new Promise((ok, fail) => {
    const s = document.createElement("script");
    s.src = src; s.onload = ok; s.onerror = () => fail(new Error("Could not load ONNX Runtime. Check the internet connection."));
    document.head.append(s);
  });
}

async function fetchWithProgress(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = +r.headers.get("content-length") || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onBytes(got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function loadModels(progress) {
  if (sessions) return sessions;
  if (!window.ort) await loadScript(`${ORT_URL}ort.min.js`);
  ort.env.wasm.wasmPaths = ORT_URL;
  ort.env.wasm.numThreads = 1;
  const meta = await loadJSON("models/meta.json");
  const files = ["soiling_classifier.onnx", "powerloss_cnn.onnx"];
  const bytes = [];
  for (const [i, f] of files.entries()) {
    bytes.push(await fetchWithProgress(`models/${f}`, (got, total) =>
      progress(0.05 + 0.4 * (i + (total ? got / total : 0)) / files.length, `Loading the networks (${(got / 1e6).toFixed(1)} MB of ${files[i]})`)));
  }
  const [clf, reg] = await Promise.all(bytes.map((b) => ort.InferenceSession.create(b, { executionProviders: ["wasm"] })));
  sessions = { clf, reg, meta };
  return sessions;
}

function toTensor(canvas, meta) {
  const { data } = canvas.getContext("2d").getImageData(0, 0, 224, 224);
  const n = 224 * 224, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) x[c * n + i] = (data[4 * i + c] / 255 - meta.mean[c]) / meta.std[c];
  }
  return new ort.Tensor("float32", x, [1, 3, 224, 224]);
}

async function readAll(progress) {
  const { clf, reg, meta } = await loadModels(progress);
  const sky = $("sky").value;
  const use = state.photos.filter((p) => !p.rejected);
  for (const [i, p] of use.entries()) {
    progress(0.45 + 0.4 * i / use.length, `Reading photograph ${i + 1} of ${use.length}`);
    const image = toTensor(p.input, meta);
    const env = core.envVector(p.date, sky);
    if (p.irradiance != null) env[0] = p.irradiance;          // sample photographs carry their measured irradiance
    const [c, r] = await Promise.all([
      clf.run({ image }),
      reg.run({ image, env: new ort.Tensor("float32", Float32Array.from(env), [1, 3]) }),
    ]);
    const probs = core.softmax([...c.logits.data]);
    const k = probs.indexOf(Math.max(...probs));
    const lossPct = r.loss.data[0] * 100;
    p.reading = { lossPct, className: meta.class_names[k], confidence: probs[k], probs, severity: core.severity(lossPct, meta.loss_bins_pct) };
    p.lossPct = lossPct;
  }
  return use;
}

async function forecastFor(lat, lon) {
  try {
    const r = await fetch(core.openMeteoUrl(lat, lon));
    if (!r.ok) throw new Error(r.status);
    return { days: core.parseForecast(await r.json()), ok: true };
  } catch {
    return { days: [], ok: false };
  }
}

function location() {
  if (place.value !== "custom") {
    const [lat, lon] = place.value.split(",").map(Number);
    return { lat, lon, name: place.options[place.selectedIndex].text };
  }
  const lat = parseFloat($("lat").value), lon = parseFloat($("lon").value);
  if (!(Math.abs(lat) <= 90 && Math.abs(lon) <= 180)) return null;
  return { lat, lon, name: `${lat.toFixed(2)}, ${lon.toFixed(2)}` };
}

function plantInputs() {
  const kwp = parseFloat($("kwp").value), tariff = parseFloat($("tariff").value), washCost = parseFloat($("wash").value);
  if (!(kwp > 0 && tariff > 0 && washCost > 0)) return null;
  return { kwp, tariff, washCost };
}

$("tool").addEventListener("submit", async (e) => {
  e.preventDefault();
  const loc = location(), plant = plantInputs();
  if (!loc) return say("run-status", "Enter a latitude between -90 and 90 and a longitude between -180 and 180.");
  if (!plant) return say("run-status", "Plant size, unit value and wash cost must all be more than zero.");
  const bar = $("progress-bar");
  $("progress").hidden = false;
  $("run").disabled = true;
  const progress = (f, text) => { bar.style.transform = `scaleX(${f})`; say("run-status", text); };
  try {
    const [photos, forecast] = await Promise.all([readAll(progress), forecastFor(loc.lat, loc.lon)]);
    progress(0.95, "Planning");
    const lastWash = $("lastwash").value ? new Date(`${$("lastwash").value}T12:00:00`) : null;
    const result = core.plan({ photos, plant, forecast: forecast.days, lastWash });
    state.results = { ...result, photos, plant, loc, forecastOk: forecast.ok, made: new Date() };
    refresh();
    render(state.results);
    progress(1, "Done.");
    $("report").hidden = false;
    $("report").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  } catch (err) {
    console.error(err);
    say("run-status", `Something went wrong: ${err.message}`);
  } finally {
    $("run").disabled = core.readiness(state.photos).length > 0;
    setTimeout(() => { $("progress").hidden = true; }, 600);
  }
});

// ---- the report ------------------------------------------------------------------------------
const longDate = (d) => fmt(d, { weekday: "long", day: "numeric", month: "long" });
const shortDate = (d) => fmt(d, { day: "numeric", month: "short" });

function render(r) {
  const today = core.startOfDay(new Date());
  const w = r.nextWash;
  const inDays = w ? core.daysBetween(today, w) : null;
  $("verdict").textContent = !w ? "No wash needed in the next 60 days"
    : inDays === 0 ? "Wash today" : inDays === 1 ? `Wash tomorrow, ${longDate(w)}` : `Wash on ${longDate(w)}`;

  const why = [];
  why.push(`Your panels are losing about ${r.nowPct.toFixed(1)}% of their power today, around ${inr(r.lostTodayInr)} a day, and dust is adding ${r.trend.rate.toFixed(2)}% a day${r.trend.rateSource === "default" ? " (the Indian average: your photographs did not show a clear build-up)" : ""}.`);
  if (r.rainSoon) why.push(`The forecast for ${r.loc.name} shows ${r.rainSoon.rainMm.toFixed(0)} mm of rain on ${shortDate(r.rainSoon.date)}, which washes most dust off for free.`);
  if (w && inDays > 0) why.push(`From then on, the power the dust takes is worth more than the ${inr(r.plant.washCost)} a wash costs.`);
  if (w && inDays === 0) why.push(`The power the dust is taking already costs more than a ${inr(r.plant.washCost)} wash.`);
  if (!w) why.push(`At ${inr(r.plant.washCost)} a wash, washing would cost more than the energy it wins back.`);
  if (r.stale) why.push(`Your newest photograph is ${r.gapDays} days old, so today's dust is projected from the trend; take a new photograph for a firmer answer.`);
  if (!r.forecastOk) why.push("The weather forecast could not be reached, so no rain is assumed.");
  $("verdict-why").textContent = why.join(" ");

  const saving = r.every15.totalInr - r.solara.totalInr;
  $("report-figures").innerHTML = [
    [`${r.nowPct.toFixed(1)}%`, "of power lost to dust today"],
    [inr(r.lostTodayInr), "lost to dust today"],
    [`${r.trend.rate.toFixed(2)}%`, "more dust each dry day"],
    [saving > 0 ? inr(saving) : inr(0), "saved over 60 days compared with washing every 15 days"],
  ].map(([n, l]) => `<p><span class="fig-num">${n}</span> <span class="fig-label">${l}</span></p>`).join("");

  drawPlan($("plan-chart"), r);
  $("plan-caption").textContent = `Dust on your panels: each dot is one photograph, the gold line is the plan from today with its washes marked underneath, the grey line is what happens without washing. Blue bars hanging from the top are forecast rain at ${r.loc.name}${r.forecastOk ? " (Open-Meteo, 16 days)" : ""}.`;

  $("readings-body").innerHTML = [...r.photos].sort((a, b) => a.date - b.date).map((p) => `
    <tr><th scope="row"><img src="${p.url}" alt="" width="48" height="48"></th>
    <td>${fmt(p.date, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</td>
    <td>${p.reading.className} <span class="muted">(${Math.round(p.reading.confidence * 100)}%)</span></td>
    <td>${p.reading.lossPct.toFixed(1)}%</td></tr>`).join("");

  const row = (name, s, best) => `<tr${best ? ' class="best"' : ""}><th scope="row">${name}</th><td>${s.washes}</td><td>${inr(s.lostInr)}</td><td>${inr(s.totalInr)}</td></tr>`;
  $("compare").innerHTML = `<thead><tr><th scope="col">Plan</th><th scope="col">Washes</th><th scope="col">Energy lost</th><th scope="col">Total</th></tr></thead><tbody>
    ${row("Solara", r.solara, true)}${row("Every 15 days", r.every15)}${row("Never wash", r.never)}</tbody>`;
}

const SVG_NS = "http://www.w3.org/2000/svg";
function drawPlan(svg, r) {
  const W = 760, H = 330, pad = { l: 52, r: 12, t: 14, b: 66 };
  const w = W - pad.l - pad.r, h = H - pad.t - pad.b;
  const today = core.startOfDay(new Date());
  const first = Math.min(...r.photos.map((p) => core.startOfDay(p.date).getTime()));
  const start = new Date(Math.min(first, today.getTime()));
  const span = core.daysBetween(start, today) + r.solara.days.length;
  const maxY = Math.max(6, ...r.photos.map((p) => p.lossPct), ...r.never.days.map((d) => d.lossPct)) * 1.1;
  const sx = (d) => pad.l + (core.daysBetween(start, d) / Math.max(span - 1, 1)) * w;
  const sy = (v) => pad.t + h - (Math.min(v, maxY) / maxY) * h;
  svg.replaceChildren();
  const el = (name, attrs, text) => {
    const e = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    svg.append(e);
    return e;
  };
  el("rect", { x: pad.l, y: pad.t, width: w, height: h, class: "plot-bg" });
  r.weather.slice(0, 16).forEach((wd, i) => {
    if (wd.rainMm < 1) return;
    const d = new Date(today.getTime() + i * core.DAY_MS);
    el("rect", { x: sx(d) - 2, y: pad.t, width: 4, height: Math.min(wd.rainMm, 40) / 40 * h * 0.4, class: "rain" });
  });
  const step = maxY > 30 ? 10 : maxY > 12 ? 5 : 2;
  for (let v = 0; v <= maxY; v += step) el("text", { x: pad.l - 8, y: sy(v) + 4, class: "tick", "text-anchor": "end" }, `${v}%`);
  for (let i = 0; i < span; i += Math.ceil(span / 8)) {
    const d = new Date(start.getTime() + i * core.DAY_MS);
    el("text", { x: sx(d), y: pad.t + h + 18, class: "tick", "text-anchor": "middle" }, shortDate(d));
  }
  el("text", { x: sx(today), y: pad.t + 14, class: "tick today-label", "text-anchor": "middle" }, "today");
  const line = (days, cls) => el("polyline", { points: days.map((d) => `${sx(d.date).toFixed(1)},${sy(d.lossPct).toFixed(1)}`).join(" "), class: `loss ${cls}` });
  line(r.never.days, "fixed");
  line(r.solara.days, "solara");
  r.solara.days.forEach((d) => { if (d.wash) el("rect", { x: sx(d.date) - 1.5, y: pad.t + h + 30, width: 3, height: 14, class: "wash solara" }); });
  el("text", { x: pad.l - 8, y: pad.t + h + 42, class: "tick", "text-anchor": "end" }, "washes");
  for (const p of r.photos) el("circle", { cx: sx(p.date), cy: sy(p.lossPct), r: 4, class: "reading" });
}

// ---- downloads --------------------------------------------------------------------------------
$("csv").addEventListener("click", () => {
  const r = state.results;
  if (!r) return;
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const lines = [
    ["Solara report", r.made.toISOString()],
    ["Location", r.loc.name, r.loc.lat, r.loc.lon],
    ["Plant kWp", r.plant.kwp], ["Unit value INR/kWh", r.plant.tariff], ["Wash cost INR", r.plant.washCost],
    ["Loss today %", r.nowPct.toFixed(2)], ["Dust per dry day %", r.trend.rate.toFixed(3)],
    ["Next wash", r.nextWash ? core.dayKey(r.nextWash) : "none in 60 days"],
    [],
    ["photo", "taken", "class", "confidence", "power_loss_pct"],
    ...r.photos.map((p) => [p.file.name, p.date.toISOString(), p.reading.className, p.reading.confidence.toFixed(3), p.reading.lossPct.toFixed(2)]),
    [],
    ["date", "plan_loss_pct", "wash", "forecast_rain_mm", "no_wash_loss_pct"],
    ...r.solara.days.map((d, i) => [core.dayKey(d.date), d.lossPct.toFixed(2), d.wash ? "wash" : "", d.rainMm.toFixed(1), r.never.days[i].lossPct.toFixed(2)]),
  ];
  const csv = lines.map((l) => l.map(q).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `solara-schedule-${core.dayKey(new Date())}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
$("print").addEventListener("click", () => window.print());

refresh();
observeReveals(document.querySelectorAll(".tool-step h2, .rules, .examples, .drop, .fields, .run"));
