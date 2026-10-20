"""Publish Module 3 results to the website (website/data/schedule.json).

Run after evaluate.py:  python -m module3_scheduling.export_site
"""
import json

from . import config

SITE_DIR = config.ROOT / "website"
KEEP = ("cleanings", "water_l", "energy_lost_mwh", "lost_revenue_inr", "cleaning_cost_inr",
        "total_cost_inr", "saving_vs_fixed_inr", "saving_vs_fixed_pct", "label")


def main():
    m = json.loads((config.OUT_DIR / "metrics.json").read_text())
    sites = {}
    for key, s in m["sites"].items():
        sites[key] = {
            "name": s["name"], "state": s["state"],
            "rain_mm_per_year": round(s["rain_mm_per_year"]),
            "dry_day_soiling_rate_pct": round(s["dry_day_soiling_rate_pct"], 2),
            "policies": {n: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items() if k in KEEP}
                         for n, p in s["policies"].items()},
            "timeline": s["timeline"],
            "recommendation": s["recommendation"],
        }
    data = {"assumptions": m["assumptions"], "summary": m["summary"], "sensitivity": m["sensitivity"], "sites": sites}
    (SITE_DIR / "data").mkdir(exist_ok=True)
    (SITE_DIR / "data" / "schedule.json").write_text(json.dumps(data))
    print(f"Wrote {SITE_DIR / 'data' / 'schedule.json'}")


if __name__ == "__main__":
    main()
