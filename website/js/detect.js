// Detect page: Module 1 results, filled from the model's own evaluation.
import { loadJSON, observeReveals, pct } from "./site.js";

loadJSON("data/results.json").then((d) => {
  const m = d.metrics;
  document.querySelector('[data-metric="accuracy"]').textContent = pct(m.accuracy);
  document.querySelector('[data-metric="macro_f1"]').textContent = m.macro_f1.toFixed(3);
  document.querySelector('[data-metric="baseline"]').textContent = pct(m.majority_class_baseline_accuracy);
  document.querySelector('[data-metric="test_images"]').textContent = m.test_images.toLocaleString("en-IN");

  const names = d.class_names;
  const head = `<thead><tr><th scope="col"><span class="visually-hidden">True class</span></th>${names.map((n) => `<th scope="col">${n}</th>`).join("")}</tr></thead>`;
  const body = d.confusion.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return `<tr><th scope="row">${names[i]}</th>${row.map((v, j) => {
      const share = v / total;
      const cls = [i === j ? "diag" : "", share > 0.5 ? "hi" : ""].filter(Boolean).join(" ");
      return `<td style="--share:${share.toFixed(3)}"${cls ? ` class="${cls}"` : ""}>${Math.round(share * 100)}%</td>`;
    }).join("")}</tr>`;
  }).join("");
  document.getElementById("confusion").innerHTML = head + `<tbody>${body}</tbody>`;

  for (const [name, n] of Object.entries(d.class_counts)) {
    const cell = document.querySelector(`[data-count="${name}"]`);
    if (cell) cell.textContent = n.toLocaleString("en-IN");
  }

  document.getElementById("samples").innerHTML = d.samples.map((s) => `
    <figure class="sample ${s.true === s.pred ? "ok" : "miss"}">
      <img src="${s.src}" alt="Panel photograph, ${s.loss.toFixed(0)}% power loss" loading="lazy" width="192" height="192">
      <figcaption><span>${s.loss.toFixed(0)}% lost</span><span>Model: ${s.pred}${s.true === s.pred ? "" : ` (true: ${s.true})`}</span></figcaption>
    </figure>`).join("");
  observeReveals(document.querySelectorAll(".figures > p, .matrix, .sample, .data-copy > p, .bins"));
}).catch(() => { /* results not published yet: the dashes stay */ });
