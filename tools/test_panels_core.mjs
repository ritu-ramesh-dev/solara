// Tests for website/js/panels-core.js.   Run:  node --test tools/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as core from "../website/js/panels-core.js";

const day = (n, h = 12) => new Date(2026, 9, 1 + n, h, 0, 0);
const photo = (n, lossPct, h) => ({ date: day(n, h), lossPct, rejected: false });
const plant = { kwp: 10, tariff: 6, washCost: 400 };
const dry = (n, ghi = 5.5) => Array.from({ length: n }, (_, i) => ({ date: day(i), rainMm: 0, ghi }));

test("EXIF DateTimeOriginal is read from a JPEG", () => {
  const b = readFileSync(new URL("./fixtures/exif_le.jpg", import.meta.url));
  const d = core.exifDate(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
  assert.equal(d.toString(), new Date(2025, 9, 21, 12, 34, 56).toString());
});

test("JPEG without EXIF and non-JPEG give no date", () => {
  const b = readFileSync(new URL("./fixtures/no_exif.jpg", import.meta.url));
  assert.equal(core.exifDate(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)), null);
  assert.equal(core.exifDate(new Uint8Array([0x89, 0x50, 0x4e, 0x47]).buffer), null);
});

test("file checks: type and size", () => {
  assert.equal(core.checkFile({ name: "a.jpg", type: "image/jpeg", size: 1000 }), null);
  assert.match(core.checkFile({ name: "a.gif", type: "image/gif", size: 1000 }), /JPG, PNG or WebP/);
  assert.match(core.checkFile({ name: "a.jpg", type: "image/jpeg", size: 25e6 }), /20 MB/);
});

test("readiness enforces the minimums", () => {
  assert.equal(core.readiness([photo(0, 1), photo(0, 1), photo(1, 2), photo(1, 2)]).length, 1); // only 2 days
  assert.equal(core.readiness([photo(0, 1), photo(1, 2), photo(2, 3)]).length, 1);               // only 3 photos
  assert.deepEqual(core.readiness([photo(0, 1), photo(1, 2), photo(2, 3), photo(2, 3)]), []);
  const rejected = [photo(0, 1), photo(1, 2), photo(2, 3), { ...photo(3, 4), rejected: true }];
  assert.equal(core.readiness(rejected).length, 1);                                               // rejected photo does not count
});

test("daily readings take the median of each day", () => {
  const d = core.dailyReadings([photo(0, 1, 9), photo(0, 9, 12), photo(0, 2, 15), photo(1, 4)]);
  assert.equal(d.length, 2);
  assert.equal(d[0].lossPct, 2);
  assert.equal(d[0].photos, 3);
});

test("trend: a steady 0.5 %/day build-up is recovered", () => {
  const days = core.dailyReadings([0, 1, 2, 3, 4, 5].map((n) => photo(n, 1 + 0.5 * n)));
  const t = core.soilingTrend(days);
  assert.ok(Math.abs(t.rate - 0.5) < 1e-9);
  assert.ok(Math.abs(t.currentPct - 3.5) < 1e-9);
  assert.equal(t.rateSource, "photos");
});

test("trend: a sharp drop (wash or rain) restarts the fit", () => {
  const days = core.dailyReadings([photo(0, 8), photo(1, 9), photo(2, 1), photo(3, 1.6), photo(4, 2.2)]);
  const t = core.soilingTrend(days);
  assert.equal(t.segmentDays, 3);
  assert.ok(Math.abs(t.rate - 0.6) < 1e-9);
});

test("trend: falling readings fall back to the Indian average rate", () => {
  const t = core.soilingTrend(core.dailyReadings([photo(0, 3), photo(1, 2.5), photo(2, 2)]));
  assert.equal(t.rate, core.SOILING.defaultRatePct);
  assert.equal(t.rateSource, "default");
});

test("schedule: heavy dust is washed, the 15-day calendar washes on day 14", () => {
  const w = dry(60);
  const s = core.simulate({ startLossPct: 12, ratePct: 0.4, weather: w, plant, policy: "solara", days: 60, today: day(0) });
  assert.ok(s.days[0].wash, "12% loss on a 10 kWp plant at Rs 6/kWh is worth a Rs 400 wash today");
  const f = core.simulate({ startLossPct: 0, ratePct: 0.4, weather: w, plant, policy: "every15", days: 30, today: day(0) });
  assert.equal(f.days.findIndex((d) => d.wash), 14);
});

test("schedule: rain in the forecast postpones the wash", () => {
  const w = dry(60);
  w[2].rainMm = 12;
  const s = core.simulate({ startLossPct: 12, ratePct: 0.4, weather: w, plant, policy: "solara", days: 60, today: day(0) });
  assert.equal(s.days[0].wash, false);
  assert.equal(s.days[1].wash, false);
  assert.ok(s.days[3].lossPct < 2, "the rain washed the panels");
});

test("schedule: the Solara rule costs no more than washing every 15 days", () => {
  const w = dry(60);
  const base = { startLossPct: 2, ratePct: 0.4, weather: w, plant, days: 60, today: day(0) };
  const sol = core.simulate({ ...base, policy: "solara" });
  const fix = core.simulate({ ...base, policy: "every15" });
  assert.ok(sol.totalInr <= fix.totalInr);
});

test("plan: stale photos are projected forward to today", () => {
  const photos = [0, 1, 2, 3].map((n) => photo(n, 1 + 0.5 * n));
  const p = core.plan({ photos, plant, forecast: dry(16), today: day(23) });
  assert.equal(p.gapDays, 20);
  assert.ok(p.stale);
  assert.ok(Math.abs(p.nowPct - (2.5 + 20 * 0.5)) < 1e-9);
});

test("Open-Meteo parsing converts MJ/m2 to kWh/m2", () => {
  const f = core.parseForecast({ daily: { time: ["2026-10-27"], precipitation_sum: [3.2], shortwave_radiation_sum: [18] } });
  assert.equal(f[0].rainMm, 3.2);
  assert.equal(f[0].ghi, 5);
});

test("image quality rejects dark frames and flags blur", () => {
  const w = 16, h = 16, flat = new Uint8ClampedArray(w * h * 4).fill(10);
  assert.ok(core.imageQuality(flat, w, h).rejected);
  const grey = new Uint8ClampedArray(w * h * 4).fill(128);
  const q = core.imageQuality(grey, w, h);
  assert.ok(!q.rejected && q.blurry);
});
