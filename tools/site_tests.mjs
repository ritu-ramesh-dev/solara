// Five end-to-end tests of the Solara websites in a real headless Chrome.
//
//   node tools/site_tests.mjs <chrome.exe> <test-photo folder> <mid-review index.html>
//
// 1. Live site health: every page loads, has a title, no console errors,
//    "Your panels" in the nav, no guide or university name anywhere.
// 2. Upload limits: too-small, too-dark and too-few-days photographs are
//    refused through the real file input, and the run button stays locked.
// 3. Real-user upload: six phone-style photographs with camera dates are read,
//    analysed for Pune and turned into a schedule.
// 4. Sample photographs with the live forecast: readings close to the
//    measured loss, live Open-Meteo forecast used.
// 5. Mid-review site from disk, offline: the working model runs without any
//    network, team names present, removed names absent.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [, , CHROME, PHOTOS, MID] = process.argv;
const SITE = "https://ritu-ramesh-dev.github.io/solara/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];

// ---- a tiny DevTools-protocol client ------------------------------------------------
async function browser() {
  const port = 9340 + Math.floor(Math.random() * 400);
  const proc = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`, "--no-first-run",
    `--user-data-dir=${process.env.TEMP}\\solara-tests-${port}`, "about:blank"], { stdio: "ignore" });
  let list;
  for (let i = 0; i < 60 && !list; i++) { try { list = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch { await sleep(250); } }
  const ws = new WebSocket(list.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const waiting = new Map();
  const errors = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); }
    if (d.method === "Runtime.exceptionThrown") errors.push(d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text);
    if (d.method === "Log.entryAdded" && d.params.entry.level === "error") errors.push(`${d.params.entry.text} ${d.params.entry.url || ""}`);
  };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const js = async (expr) => {
    const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || "script error");
    return r.result?.result?.value;
  };
  await send("Runtime.enable"); await send("Log.enable"); await send("Network.enable"); await send("DOM.enable"); await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  const go = async (url) => { await send("Page.navigate", { url }); await sleep(2500); };
  const upload = async (selector, files) => {
    const doc = await send("DOM.getDocument", { depth: 1 });
    const q = await send("DOM.querySelector", { nodeId: doc.result.root.nodeId, selector });
    await send("DOM.setFileInputFiles", { nodeId: q.result.nodeId, files });
    await sleep(2500);
  };
  const waitFor = async (expr, ms = 120000) => { const t = Date.now(); while (Date.now() - t < ms) { if (await js(expr)) return true; await sleep(500); } return false; };
  return { send, js, go, upload, waitFor, errors, close: () => { ws.close(); proc.kill(); } };
}

function check(name, cond, detail) {
  if (!cond) throw new Error(`${name}${detail ? ` (${detail})` : ""}`);
}

async function test(n, title, fn) {
  const b = await browser();
  const t = Date.now();
  try {
    const notes = await fn(b);
    results.push({ n, title, ok: true, notes, s: (Date.now() - t) / 1000 });
  } catch (e) {
    results.push({ n, title, ok: false, notes: e.message, s: (Date.now() - t) / 1000 });
  } finally {
    b.close();
  }
}

const photos = JSON.parse(readFileSync(join(PHOTOS, "manifest.json"), "utf8"));
const P = (f) => join(PHOTOS, f);
const FORBIDDEN = /Gargi|Amrita|Vidyapeetham/i;

// ---- 1 ------------------------------------------------------------------------------------
await test(1, "Live site: every page loads cleanly", async (b) => {
  const pages = ["", "analyze.html", "detect.html", "predict.html", "schedule.html", "about.html"];
  for (const p of pages) {
    b.errors.length = 0;
    await b.go(SITE + p);
    const info = await b.js(`({ h1: document.querySelector('h1')?.textContent.trim(), nav: [...document.querySelectorAll('.bar nav a')].map(a=>a.textContent), text: document.body.innerText })`);
    check(`/${p} has a title`, info.h1);
    check(`/${p} links to Your panels`, info.nav.includes("Your panels"));
    check(`/${p} has no guide or university name`, !FORBIDDEN.test(info.text));
    check(`/${p} has no console errors`, b.errors.length === 0, b.errors.join("; "));
  }
  await b.go(SITE + "detect.html");
  const acc = await b.js("document.querySelector('[data-metric=accuracy]').textContent");
  check("Detect page shows the model accuracy", acc === "79.9%", acc);
  return `${pages.length} pages, titles present, no errors, no removed names; Detect shows ${acc}`;
});

// ---- 2 ------------------------------------------------------------------------------------
await test(2, "Upload limits are enforced", async (b) => {
  await b.go(SITE + "analyze.html");
  await b.upload("#files", [P("tiny.jpg"), P("dark.jpg")]);
  const s1 = await b.js("({ msg: document.getElementById('photo-status').textContent, kept: document.querySelectorAll('.thumb').length, flag: [...document.querySelectorAll('.thumb-flag')].map(f=>f.textContent).join(), run: document.getElementById('run').disabled })");
  check("tiny photo refused", /too small/.test(s1.msg), s1.msg);
  check("dark photo kept but not used", s1.kept === 1 && /too dark/.test(s1.flag), s1.flag);
  await b.upload("#files", [P("user_4.jpg"), P("user_5.jpg")]);            // two photos, one day
  const s2 = await b.js("({ run: document.getElementById('run').disabled, why: document.getElementById('run-status').textContent })");
  check("run locked with too few photos and days", s2.run && /at least 4/.test(s2.why) && /3 different days/.test(s2.why), s2.why);
  await b.upload("#files", [P("user_4.jpg")]);                              // same file again
  const dup = await b.js("document.getElementById('photo-status').textContent");
  check("duplicate refused", /already added/.test(dup), dup);
  return `"${s1.msg.trim()}"; dark frame "${s1.flag}"; locked with: "${s2.why}"; duplicate refused`;
});

// ---- 3 ------------------------------------------------------------------------------------
await test(3, "Real-user upload: phone photos to a schedule", async (b) => {
  await b.go(SITE + "analyze.html");
  await b.upload("#files", photos.files.map(P));
  const dates = await b.js("[...document.querySelectorAll('.thumb input')].map(i=>i.value)");
  check("6 photos accepted", dates.length === 6, dates.length);
  check("dates read from the camera (EXIF)", JSON.stringify(dates) === JSON.stringify(photos.dates), `${dates} vs ${photos.dates}`);
  await b.js("document.getElementById('place').value='18.52,73.86'; true");
  await b.js("document.getElementById('run').click(); true");
  check("report appears", await b.waitFor("!document.getElementById('report').hidden || /wrong/.test(document.getElementById('run-status').textContent)"));
  const r = await b.js(`({ status: document.getElementById('run-status').textContent, verdict: document.getElementById('verdict').textContent,
    rows: document.querySelectorAll('#readings-body tr').length,
    losses: [...document.querySelectorAll('#readings-body tr td:last-child')].map(t=>parseFloat(t.textContent)),
    chart: document.querySelectorAll('#plan-chart polyline').length, csv: !!document.getElementById('csv') })`);
  check("analysis finished", r.status === "Done.", r.status);
  check("verdict given", /Wash|No wash/.test(r.verdict), r.verdict);
  check("6 readings, all 0-100%", r.rows === 6 && r.losses.every((x) => x >= 0 && x <= 100), r.losses);
  check("chart drawn", r.chart === 2);
  return `EXIF dates read for all 6; verdict "${r.verdict}"; readings ${r.losses.join(", ")}%`;
});

// ---- 4 ------------------------------------------------------------------------------------
await test(4, "Sample photographs with the live forecast", async (b) => {
  await b.go(SITE + "analyze.html");
  await b.js("document.getElementById('place').value='14.25,77.43'; document.getElementById('demo').click(); true");
  await sleep(4000);
  await b.js("document.getElementById('run').click(); true");
  check("report appears", await b.waitFor("!document.getElementById('report').hidden || /wrong/.test(document.getElementById('run-status').textContent)"));
  const demo = await (await fetch(SITE + "data/demo_panels.json")).json();
  const r = await b.js(`({ verdict: document.getElementById('verdict').textContent, caption: document.getElementById('plan-caption').textContent,
    losses: [...document.querySelectorAll('#readings-body tr td:last-child')].map(t=>parseFloat(t.textContent)) })`);
  const measured = demo.photos.map((p) => p.measuredLossPct);
  const mae = r.losses.reduce((a, x, i) => a + Math.abs(x - measured[i]), 0) / measured.length;
  check("8 readings", r.losses.length === 8);
  check("readings close to the measured loss (MAE under 2 points)", mae < 2, mae.toFixed(2));
  check("live Open-Meteo forecast used", /Open-Meteo/.test(r.caption), r.caption);
  check("no console errors", b.errors.length === 0, b.errors.join("; "));
  return `verdict "${r.verdict}"; readings vs measured loss: MAE ${mae.toFixed(2)} points; live forecast used`;
});

// ---- 5 ------------------------------------------------------------------------------------
await test(5, "Mid-review site from disk, offline", async (b) => {
  await b.send("Network.emulateNetworkConditions", { offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  await b.go(pathToFileURL(MID).href);
  const text = await b.js("document.body.innerText");
  check("opened from disk", (await b.js("location.protocol")) === "file:");
  check("team names present", ["Dinesh Karthik Raja", "Jayapriya", "Jaiakash", "Ritu Ramesh"].every((n) => text.includes(n)));
  check("removed names absent", !FORBIDDEN.test(text));
  await b.js("document.getElementById('demo').click(); true");
  await sleep(3000);
  await b.js("document.getElementById('run').click(); true");
  check("plan appears", await b.waitFor("!document.getElementById('out').hidden || /wrong/.test(document.getElementById('runMsg').textContent)"));
  const r = await b.js(`({ verdict: document.getElementById('verdict').textContent, note: document.getElementById('weatherNote').textContent,
    rows: document.querySelectorAll('#readings tbody tr').length })`);
  const other = b.errors.filter((e) => !/open-meteo/.test(e));
  check("8 readings", r.rows === 8, r.rows);
  check("offline weather used", /^Offline/.test(r.note), r.note);
  check("no errors except the expected offline forecast request", other.length === 0, other.join("; "));
  return `verdict "${r.verdict}" with no network; ${r.note}`;
});

// ---- report --------------------------------------------------------------------------------
for (const r of results) console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.n}. ${r.title}  (${r.s.toFixed(1)} s)\n      ${r.notes}`);
console.log(`\n${results.filter((r) => r.ok).length} of ${results.length} passed`);
process.exit(results.every((r) => r.ok) ? 0 : 1);
