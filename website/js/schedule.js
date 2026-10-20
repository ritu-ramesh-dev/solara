// Module 3: cleaning schedule at five Indian solar parks.
const SVG_NS = "http://www.w3.org/2000/svg";
const inr = (v) => `₹${Math.round(v).toLocaleString("en-IN")}`;
const fmtDate = (s) => new Date(`${s}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" });
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function drawTimeline(svg, tl) {
  const W = 760, H = 320, pad = { l: 56, r: 10, t: 14, b: 70 };
  const w = W - pad.l - pad.r, h = H - pad.t - pad.b;
  const n = tl.rain_mm.length;
  const maxLoss = Math.max(8, ...tl.fixed_common.loss, ...tl.solara.loss, ...tl.optimum.loss) * 1.08;
  const sx = (i) => pad.l + (i / (n - 1)) * w;
  const sy = (v) => pad.t + h - (v / maxLoss) * h;
  const el = (name, attrs, text) => {
    const e = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    svg.appendChild(e);
    return e;
  };
  svg.replaceChildren();
  el("rect", { x: pad.l, y: pad.t, width: w, height: h, class: "plot-bg" });
  // Rain hangs from the top edge, so it never hides the loss lines.
  const rainMax = 60;
  tl.rain_mm.forEach((r, i) => {
    if (r >= 1) el("rect", { x: sx(i) - 1, y: pad.t, width: 2.2, height: (Math.min(r, rainMax) / rainMax) * h * 0.45, class: "rain" });
  });
  const step = maxLoss > 20 ? 10 : 5;
  for (let v = 0; v <= maxLoss; v += step) el("text", { x: pad.l - 8, y: sy(v) + 4, class: "tick", "text-anchor": "end" }, `${v}%`);
  MONTHS.forEach((m, k) => el("text", { x: sx(Math.round(k * 30.44)) + 4, y: pad.t + h + 18, class: "tick" }, m));
  const line = (vals, cls) => el("polyline", {
    points: vals.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" "), class: cls,
  });
  line(tl.fixed_common.loss, "loss fixed");
  line(tl.optimum.loss, "loss optimum");
  line(tl.solara.loss, "loss solara");
  const ticks = (days, y, cls) => days.forEach((d) => el("rect", { x: sx(d) - 1.2, y, width: 2.4, height: 12, class: cls }));
  ticks(tl.fixed_common.washes, pad.t + h + 28, "wash fixed");
  ticks(tl.solara.washes, pad.t + h + 46, "wash solara");
  el("text", { x: pad.l - 8, y: pad.t + h + 38, class: "tick", "text-anchor": "end" }, "15-day");
  el("text", { x: pad.l - 8, y: pad.t + h + 56, class: "tick", "text-anchor": "end" }, "Solara");
}

function render(d) {
  const picker = document.getElementById("park-picker");
  const keys = Object.keys(d.sites);
  const year = d.sites[keys[0]].timeline.start.slice(0, 4);
  const sol = d.summary.solara, fix = d.summary.fixed_common;
  const [y0, y1] = d.assumptions.test_years;
  document.getElementById("schedule-lede").textContent =
    `Module 3 decides each morning whether washing the panels is worth it, from Module 2's reading of the dust, the expected sunshine and a three-day rain forecast. On ${y0} to ${y1} weather at five Indian solar parks it costs on average ${Math.round(sol.saving_vs_fixed_pct)}% less than washing every 15 days, with ${Math.round(fix.cleanings - sol.cleanings)} fewer washes a year.`;
  document.getElementById("schedule-status").innerHTML =
    `On average ${Math.round(sol.saving_vs_fixed_pct)}% cheaper than washing every 15 days at five Indian parks. <a href="#schedule">See the parks</a>.`;

  function show(key) {
    const s = d.sites[key], p = s.policies, so = p.solara, fx = p.fixed_common;
    for (const b of picker.children) {
      const on = b.dataset.key === key;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    document.getElementById("park-name").textContent = `${s.name}, ${s.state}. ${s.rain_mm_per_year.toLocaleString("en-IN")} mm of rain a year.`;
    document.getElementById("park-saving").textContent = inr(so.saving_vs_fixed_inr);
    document.getElementById("park-washes").textContent = so.cleanings.toFixed(1);
    document.getElementById("park-washes-label").textContent = `washes a year, instead of ${fx.cleanings.toFixed(0)}`;
    document.getElementById("park-water").textContent = Math.round(fx.water_l - so.water_l).toLocaleString("en-IN");
    const rec = s.recommendation;
    document.getElementById("park-next").innerHTML = rec.next_wash
      ? `Next wash: <strong>${fmtDate(rec.next_wash)}</strong>, projected from the weather up to ${fmtDate(rec.as_of)}.`
      : `No wash needed in the four months after ${fmtDate(rec.as_of)}.`;
    document.getElementById("timeline-caption").textContent =
      `Soiling loss at ${s.name} through ${year}: washing every 15 days (grey), when Solara decides (gold) and the hindsight optimum (dashed). Solara lets a little more dust build up, as the optimum does, because ${fx.cleanings.toFixed(0)} washes a year cost more than the energy they save. Ticks under the chart mark washes; blue bars hanging from the top are rain.`;
    document.getElementById("policies-caption").textContent = `Every policy at ${s.name}, per MW, ${y0} to ${y1}`;
    const order = ["rain_only", "fixed_common", "fixed_tuned", "threshold", "ppo", "solara", "optimum"];
    document.getElementById("policies-body").innerHTML = order.filter((k) => p[k]).map((k) => {
      const v = p[k].saving_vs_fixed_pct;
      const vs = k === "fixed_common" ? "–" : `${v >= 0 ? "−" : "+"}${Math.abs(v).toFixed(0)}%`;
      return `<tr${k === "solara" ? ' class="best"' : ""}><th scope="row">${p[k].label}</th>
        <td>${p[k].cleanings.toFixed(1)}</td><td>${p[k].energy_lost_mwh.toFixed(1)}</td>
        <td>${inr(p[k].total_cost_inr)}</td><td>${vs}</td></tr>`;
    }).join("");
    drawTimeline(document.getElementById("timeline"), s.timeline);
  }

  picker.innerHTML = keys.map((k) =>
    `<button type="button" role="radio" data-key="${k}" aria-checked="false">${d.sites[k].name.split(" ")[0]}<span>${d.sites[k].state}</span></button>`).join("");
  picker.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) show(b.dataset.key);
  });
  picker.addEventListener("keydown", (e) => {
    const step = ["ArrowRight", "ArrowDown"].includes(e.key) ? 1 : ["ArrowLeft", "ArrowUp"].includes(e.key) ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const i = keys.indexOf(picker.querySelector('[aria-checked="true"]').dataset.key);
    const next = keys[(i + step + keys.length) % keys.length];
    show(next);
    picker.querySelector(`[data-key="${next}"]`).focus();
  });
  show(keys[0]);
}

fetch("data/schedule.json")
  .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
  .then(render)
  .catch(() => { /* results not published yet */ });

// Entrances, matching the rest of the page.
if ("IntersectionObserver" in window && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { threshold: 0.15 });
  document.querySelectorAll(".schedule-head > *, .park-picker, .timeline, .park-facts, .policies").forEach((n, i) => {
    n.classList.add("reveal");
    n.style.setProperty("--d", `${(i % 6) * 75}ms`);
    io.observe(n);
  });
}
