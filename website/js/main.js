const LOSS_TEXT = ["under 5%", "5 to 15%", "15 to 30%", "over 30%"];
const DUST_LEVEL = [0, 0.38, 0.68, 1];

let scene = null;

function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch { return false; }
}

if (hasWebGL()) {
  import("./scene.js")
    .then(({ startScene }) => {
      scene = startScene(document.getElementById("stage"), {
        onReady: () => document.body.classList.add("stage-ready"),
      });
      const checked = document.querySelector('input[name="soil"]:checked');
      if (checked) scene.setSoiling(DUST_LEVEL[+checked.value]);
    })
    .catch((err) => { console.error(err); document.body.classList.add("no-webgl"); });
} else {
  document.body.classList.add("no-webgl");
}

// ---- bar turns solid once the hero has scrolled away ----------------------------
const bar = document.querySelector(".bar");
const results = document.getElementById("why");
const updateBar = () => bar.classList.toggle("solid", scrollY > results.offsetTop - 80);
addEventListener("scroll", updateBar, { passive: true });
updateBar();

// ---- tabs ---------------------------------------------------------------------
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    const pane = document.getElementById(t.getAttribute("aria-controls"));
    pane.hidden = !on;
    if (on) { pane.classList.remove("pane-in"); void pane.offsetWidth; pane.classList.add("pane-in"); }
  }
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t));
  t.addEventListener("keydown", (e) => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    const next = tabs[(i + d + tabs.length) % tabs.length];
    next.focus();
    selectTab(next);
  });
});

// ---- soiling picker: dusts the 3D panel ------------------------------------------
for (const input of document.querySelectorAll('input[name="soil"]')) {
  input.addEventListener("change", () => {
    const k = +input.value;
    document.getElementById("soil-loss").textContent = LOSS_TEXT[k];
    scene && scene.setSoiling(DUST_LEVEL[k]);
  });
}

// ---- results, filled from the model's own evaluation -------------------------------
const pct = (x) => `${(x * 100).toFixed(1)}%`;

fetch("data/results.json")
  .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
  .then((d) => {
    const m = d.metrics;
    document.querySelector('[data-metric="accuracy"]').textContent = pct(m.accuracy);
    document.querySelector('[data-metric="macro_f1"]').textContent = m.macro_f1.toFixed(3);
    document.querySelector('[data-metric="test_images"]').textContent = m.test_images.toLocaleString("en-IN");
    document.getElementById("results-lede").textContent =
      `Scores on ${m.test_images.toLocaleString("en-IN")} photographs the network never saw during training. Guessing the most common class would score ${pct(m.majority_class_baseline_accuracy)}.`;

    const names = d.class_names;
    const table = document.getElementById("confusion");
    const head = `<thead><tr><th scope="col"><span class="visually-hidden">True class</span></th>${names.map((n) => `<th scope="col">${n}</th>`).join("")}</tr></thead>`;
    const body = d.confusion.map((row, i) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      return `<tr><th scope="row">${names[i]}</th>${row.map((v, j) => {
        const share = v / total;
        const cls = [i === j ? "diag" : "", share > 0.5 ? "hi" : ""].filter(Boolean).join(" ");
        return `<td style="--share:${share.toFixed(3)}"${cls ? ` class="${cls}"` : ""}>${Math.round(share * 100)}%</td>`;
      }).join("")}</tr>`;
    }).join("");
    table.innerHTML = head + `<tbody>${body}</tbody>`;

    for (const [name, n] of Object.entries(d.class_counts)) {
      const cell = document.querySelector(`[data-count="${name}"]`);
      if (cell) cell.textContent = n.toLocaleString("en-IN");
    }

    document.getElementById("samples").innerHTML = d.samples.map((s) => `
      <figure class="sample ${s.true === s.pred ? "ok" : "miss"}">
        <img src="${s.src}" alt="Panel photograph, ${s.loss.toFixed(0)}% power loss" loading="lazy" width="192" height="192">
        <figcaption><span>${s.loss.toFixed(0)}% lost</span><span>Model: ${s.pred}${s.true === s.pred ? "" : ` (true: ${s.true})`}</span></figcaption>
      </figure>`).join("");
    observeReveals(document.querySelectorAll(".sample"));
  })
  .catch(() => { /* results not published yet: the dashes stay */ });

// ---- entrances -------------------------------------------------------------------
const io = "IntersectionObserver" in window && !matchMedia("(prefers-reduced-motion: reduce)").matches
  ? new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }, { threshold: 0.15 })
  : null;

function observeReveals(nodes) {
  nodes.forEach((n, i) => {
    if (!io) return;
    n.classList.add("reveal");
    n.style.setProperty("--d", `${(i % 6) * 75}ms`);
    io.observe(n);
  });
}
observeReveals(document.querySelectorAll(
  ".why-wide, .why-copy > *, .why-side, .results h2, .results .section-lede, .figures > p, .matrix, .data-copy > *, .bins, .close-meta p",
));

// ---- Module 2: power-loss prediction ------------------------------------------------
const SVG_NS = "http://www.w3.org/2000/svg";

function drawScatter(svg, points) {
  const size = 440, pad = { l: 48, r: 12, t: 12, b: 44 };
  const w = size - pad.l - pad.r, h = size - pad.t - pad.b;
  const sx = (v) => pad.l + (v / 100) * w;
  const sy = (v) => pad.t + h - (v / 100) * h;
  const el = (name, attrs, text) => {
    const n = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    svg.appendChild(n);
    return n;
  };
  svg.replaceChildren();
  el("rect", { x: pad.l, y: pad.t, width: w, height: h, class: "plot-bg" });
  for (const t of [0, 25, 50, 75, 100]) {
    el("text", { x: sx(t), y: size - pad.b + 20, class: "tick", "text-anchor": "middle" }, `${t}%`);
    el("text", { x: pad.l - 8, y: sy(t) + 4, class: "tick", "text-anchor": "end" }, `${t}%`);
  }
  el("text", { x: pad.l + w / 2, y: size - 4, class: "axis", "text-anchor": "middle" }, "measured loss");
  el("text", { x: 12, y: pad.t + h / 2, class: "axis", "text-anchor": "middle", transform: `rotate(-90 12 ${pad.t + h / 2})` }, "predicted loss");
  el("line", { x1: sx(0), y1: sy(0), x2: sx(100), y2: sy(100), class: "diag" });
  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("class", "dots");
  for (const [a, b] of points) {
    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", sx(a).toFixed(1));
    c.setAttribute("cy", sy(b).toFixed(1));
    c.setAttribute("r", "2.3");
    g.appendChild(c);
  }
  svg.appendChild(g);
}

fetch("data/powerloss.json")
  .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
  .then((d) => {
    const best = d.models.find((m) => m.key === d.best_model);
    document.querySelector('[data-power="mae"]').textContent = best.mae.toFixed(1);
    document.querySelector('[data-power="r2"]').textContent = best.r2.toFixed(3);
    document.getElementById("power-lede").textContent =
      `Module 2 puts a number on the dust. The best model, the ${best.label.split(",")[0]}, predicts the share of power lost from the photograph, the irradiance and the time of day, within ${best.mae.toFixed(1)} percentage points on average across ${d.test_images.toLocaleString("en-IN")} test photographs.`;
    document.getElementById("models-body").innerHTML = d.models.map((m) => `
      <tr${m.key === d.best_model ? ' class="best"' : ""}><th scope="row">${m.label}</th>
      <td>${m.mae.toFixed(2)}</td><td>${m.rmse.toFixed(2)}</td><td>${m.r2.toFixed(3)}</td></tr>`).join("");
    drawScatter(document.getElementById("scatter"), d.scatter);
    document.getElementById("predict-status").innerHTML =
      `Average error ${best.mae.toFixed(1)} percentage points, R² ${best.r2.toFixed(3)}. <a href="#power">See every model</a>.`;
  })
  .catch(() => { /* not published yet */ });

observeReveals(document.querySelectorAll(".power-copy > *, .scatter"));
