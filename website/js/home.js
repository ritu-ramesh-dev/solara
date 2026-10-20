// Home page: the 3D panel, its dust, and the chapter figures.
import { loadJSON, pct } from "./site.js";

let scene = null;
const pending = { level: null };

// Soiling loss (% of power) -> how thick the dust layer looks on the glass.
const lossToDust = (lossPct) => Math.min(1, Math.pow(Math.max(lossPct, 0) / 34, 0.85));

function setDust(level) {
  if (scene) scene.setSoiling(level);
  else pending.level = level;
}

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
      if (pending.level != null) scene.setSoiling(pending.level);
    })
    .catch((err) => { console.error(err); document.body.classList.add("no-webgl"); });
} else {
  document.body.classList.add("no-webgl");
}

// ---- dust: Bhadla as simulated by Module 3, or a chosen class -------------------
const readout = document.getElementById("soil-readout");
const LABELS = { 0: "Clean glass: under 5% of power lost.", 10: "Light soiling: 5 to 15% of power lost.", 22: "Moderate soiling: 15 to 30% of power lost.", 40: "Heavy soiling: over 30% of power lost." };
let bhadla = null;
const fmtDate = (s) => new Date(`${s}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" });

function applyChoice(value) {
  if (value === "bhadla") {
    if (!bhadla) return;
    setDust(lossToDust(bhadla.loss));
    readout.innerHTML = `Bhadla Solar Park, ${fmtDate(bhadla.date)}: <strong>${bhadla.loss.toFixed(1)}%</strong> of power lost, as simulated by Module 3. <a href="schedule.html">Next wash</a>`;
  } else {
    setDust(lossToDust(+value));
    readout.textContent = LABELS[value];
  }
}
for (const input of document.querySelectorAll('input[name="soil"]')) {
  input.addEventListener("change", () => applyChoice(input.value));
}

loadJSON("data/schedule.json").then((d) => {
  const rec = d.sites.bhadla.recommendation;
  bhadla = { loss: rec.loss_on_that_day_pct, date: rec.next_wash };
  const checked = document.querySelector('input[name="soil"]:checked');
  applyChoice(checked ? checked.value : "bhadla");
  document.querySelector('[data-home="saving"]').textContent = `${Math.round(d.summary.solara.saving_vs_fixed_pct)}%`;
}).catch(() => applyChoice("10"));

loadJSON("data/results.json").then((d) => {
  document.querySelector('[data-home="accuracy"]').textContent = pct(d.metrics.accuracy);
}).catch(() => {});
loadJSON("data/powerloss.json").then((d) => {
  const best = d.models.find((m) => m.key === d.best_model);
  document.querySelector('[data-home="mae"]').textContent = best.mae.toFixed(1);
}).catch(() => {});
