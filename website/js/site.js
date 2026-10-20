// Shared by every page: the bar turns solid once the page has scrolled, and
// sections rise into view as they enter the screen.
const bar = document.querySelector(".bar");
const home = document.body.classList.contains("home");
const updateBar = () => bar.classList.toggle("solid", scrollY > (home ? innerHeight * 0.7 : 24));
addEventListener("scroll", updateBar, { passive: true });
updateBar();

const io = "IntersectionObserver" in window && !matchMedia("(prefers-reduced-motion: reduce)").matches
  ? new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }, { threshold: 0.12 })
  : null;

export function observeReveals(nodes) {
  if (!io) return;
  [...nodes].forEach((n, i) => {
    n.classList.add("reveal");
    n.style.setProperty("--d", `${(i % 6) * 75}ms`);
    io.observe(n);
  });
}

export const loadJSON = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${url}: ${r.status}`))));
export const pct = (x) => `${(x * 100).toFixed(1)}%`;
export const inr = (v) => `₹${Math.round(v).toLocaleString("en-IN")}`;

observeReveals(document.querySelectorAll(
  "main > section h2, .why-wide, .why-copy > *, .why-side, .project-grid > *, .panel-spec, .credit-list li, .method-grid > div, .chapter-list li, .pager a, .close-meta p",
));
