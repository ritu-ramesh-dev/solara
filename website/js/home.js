// Home page: the opening film and the chapter figures.
import { loadJSON, pct } from "./site.js";

// ---- opening film: pausable, still for people who ask for reduced motion ----------
const video = document.getElementById("hero-video");
const toggle = document.getElementById("video-toggle");
const reduce = matchMedia("(prefers-reduced-motion: reduce)");

function setPlaying(play) {
  if (play) video.play().catch(() => {});
  else video.pause();
  toggle.textContent = play ? "Pause film" : "Play film";
  toggle.setAttribute("aria-pressed", String(!play));
}
if (reduce.matches) setPlaying(false);
toggle.addEventListener("click", () => setPlaying(video.paused));
video.addEventListener("playing", () => document.body.classList.add("film-ready"), { once: true });

// Pause the film while it is scrolled out of view, to spare battery.
if ("IntersectionObserver" in window) {
  new IntersectionObserver(([e]) => {
    if (toggle.getAttribute("aria-pressed") === "true") return;
    if (e.isIntersecting) video.play().catch(() => {});
    else video.pause();
  }).observe(video);
}

// Slight parallax: the film drifts slower than the page.
if (!reduce.matches) {
  const media = document.querySelector(".hero-media");
  addEventListener("scroll", () => {
    const y = Math.min(scrollY, innerHeight);
    media.style.transform = `translate3d(0, ${y * 0.35}px, 0)`;
  }, { passive: true });
}

// ---- chapter figures from the models' own results ------------------------------------
loadJSON("data/results.json").then((d) => {
  document.querySelector('[data-home="accuracy"]').textContent = pct(d.metrics.accuracy);
}).catch(() => {});
loadJSON("data/powerloss.json").then((d) => {
  const best = d.models.find((m) => m.key === d.best_model);
  document.querySelector('[data-home="mae"]').textContent = best.mae.toFixed(1);
}).catch(() => {});
loadJSON("data/schedule.json").then((d) => {
  document.querySelector('[data-home="saving"]').textContent = `${Math.round(d.summary.solara.saving_vs_fixed_pct)}%`;
}).catch(() => {});
