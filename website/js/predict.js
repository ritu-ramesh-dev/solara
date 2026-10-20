// Predict page: Module 2 results and the predicted-vs-measured scatter.
import { loadJSON, observeReveals, pct } from "./site.js";

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

loadJSON("data/powerloss.json").then((d) => {
  const best = d.models.find((m) => m.key === d.best_model);
  const env = d.models.find((m) => m.key === "env_xgboost");
  const cnn = d.models.find((m) => m.key === "cnn");
  document.querySelector('[data-power="mae"]').textContent = best.mae.toFixed(1);
  document.querySelector('[data-power="r2"]').textContent = best.r2.toFixed(3);
  if (env) document.querySelector('[data-power="env"]').textContent = env.mae.toFixed(1);
  document.getElementById("power-lede").textContent =
    `Module 2 puts a number on the dust. The best model, the ${best.label.split(",")[0]}, predicts the share of power lost from the photograph, the irradiance and the time of day to within ${best.mae.toFixed(1)} percentage points on average, across ${d.test_images.toLocaleString("en-IN")} test photographs.`;
  document.getElementById("models-body").innerHTML = d.models.map((m) => `
    <tr${m.key === d.best_model ? ' class="best"' : ""}><th scope="row">${m.label}</th>
    <td>${m.mae.toFixed(2)}</td><td>${m.rmse.toFixed(2)}</td><td>${m.r2.toFixed(3)}</td></tr>`).join("");
  if (cnn) {
    document.getElementById("binned-note").textContent =
      `Cut at the Detect page's class boundaries, the CNN regressor's predictions classify soiling with ${pct(cnn.binned_accuracy)} accuracy, more than the classifier built for the job (${pct(d.module1_classifier.accuracy)}).`;
  }
  drawScatter(document.getElementById("scatter"), d.scatter);
  document.getElementById("strip").innerHTML = (d.samples || []).map((s) => `
    <figure class="strip-item">
      <img src="${s.src}" alt="Test panel photograph with ${s.measured.toFixed(0)}% measured power loss" loading="lazy" width="192" height="192">
      <figcaption><span class="strip-measured">${s.measured.toFixed(0)}% lost</span><span>read as ${s.predicted.toFixed(1)}%</span></figcaption>
    </figure>`).join("");
  observeReveals(document.querySelectorAll(".power-figures > p, .models, .note, .scatter, .strip-item"));
}).catch(() => { /* not published yet */ });
