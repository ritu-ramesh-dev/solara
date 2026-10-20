"""Tune every policy on 2005-2019 and test it on 2020-2024 at five Indian solar parks.

Writes to module3_scheduling/outputs/:
  metrics.json          per site and policy: washes, water, energy and money per MWp per year
  cost_by_policy.png    yearly cost of each policy at each site
  timeline_<site>.png   one year of soiling loss under three schedules
  sensitivity.png       Solara's saving when a wash costs half or double

Usage:  python -m module3_scheduling.evaluate   (after weather.py; rl.py for PPO)
"""
import json
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, dp
from .policies import CostBenefit, FixedInterval, NeverClean, tuning_grid
from .sim import calibrate_rate_constant, load_site, run

SEEDS = range(5)
TUNE_SEEDS = range(2)
COMMON_PRACTICE_DAYS = 15
TIMELINE_YEAR = 2023


def mean_per_year(site, policy, seeds=SEEDS):
    rows = [run(site, policy, seed=s).per_year() for s in seeds]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def tune(site):
    best = {}
    for family, candidates in tuning_grid().items():
        scored = [(mean_per_year(site, p, TUNE_SEEDS)["total_cost_inr"], i, p) for i, p in enumerate(candidates)]
        best[family] = min(scored)[2]
    return best


def evaluate_site(key, k, ppo=None):
    site = load_site(key, k)
    train, test = site.years(config.TRAIN_YEARS), site.years(config.TEST_YEARS)
    tuned = tune(train)
    policies = {
        "rain_only": NeverClean(),
        "fixed_common": FixedInterval(COMMON_PRACTICE_DAYS),
        "fixed_tuned": tuned["fixed"],
        "threshold": tuned["threshold"],
        "solara": tuned["cost_benefit"],
    }
    if ppo is not None:
        policies["ppo"] = ppo
    out = {}
    for name, pol in policies.items():
        out[name] = {"label": pol.name, **mean_per_year(test, pol)}
    opt = dp.solve(test)
    out["optimum"] = {"label": "Hindsight optimum (knows the future)", **opt.per_year()}
    return site, test, tuned, policies, out, opt


def timeline(test, policies, opt, year=TIMELINE_YEAR):
    m = test.dates.year == year
    traces = {}
    for name in ("fixed_common", "solara"):
        r = run(test, policies[name], seed=0)
        traces[name] = {"loss": r.loss_trace[m].round(2).tolist(), "washes": np.nonzero(r.clean_days[m])[0].tolist()}
    traces["optimum"] = {"loss": opt.loss_trace[m].round(2).tolist(), "washes": np.nonzero(opt.clean_days[m])[0].tolist()}
    traces["rain_mm"] = test.rain_mm[m].round(1).tolist()
    traces["start"] = str(test.dates[m][0].date())
    return traces


def recommend(site, test, policy):
    """Next wash date for the Solara policy, as of the last day of weather data.

    Runs the policy through the test years, then projects the loss forward
    with this site's average soiling rate and energy for each calendar day,
    and finds the first day a wash would pay for itself.
    """
    r = run(test, policy, seed=0)
    last = test.dates[-1]
    loss = float(r.loss_trace[-1])
    doy_rate = {d: site.rate_pct[site.dates.dayofyear == d].mean() for d in range(1, 367)}
    doy_energy = {d: site.energy_kwh[site.dates.dayofyear == d].mean() for d in range(1, 367)}
    for ahead in range(1, 121):
        day = last + timedelta(days=ahead)
        loss += doy_rate[day.dayofyear]
        gain = (loss - config.RESIDUAL_AFTER_CLEANING_PCT) / 100 * doy_energy[day.dayofyear]
        if gain * config.TARIFF_INR_PER_KWH * policy.horizon >= config.CLEANING_COST_INR:
            return {"as_of": str(last.date()), "next_wash": str(day.date()), "days_ahead": ahead,
                    "loss_on_that_day_pct": round(loss, 2)}
    return {"as_of": str(last.date()), "next_wash": None}


def sensitivity(k, multipliers=(0.5, 1.0, 2.0)):
    """Saving of Solara over the 15-day schedule when one wash costs more or less."""
    base = config.CLEANING_COST_INR
    rows = []
    try:
        for mlt in multipliers:
            config.CLEANING_COST_INR = base * mlt
            sav = []
            for key in config.SITES:
                site = load_site(key, k)
                train, test = site.years(config.TRAIN_YEARS), site.years(config.TEST_YEARS)
                cb = min(((mean_per_year(train, p, TUNE_SEEDS)["total_cost_inr"], i, p)
                          for i, p in enumerate(tuning_grid()["cost_benefit"])))[2]
                f = mean_per_year(test, FixedInterval(COMMON_PRACTICE_DAYS))["total_cost_inr"]
                s = mean_per_year(test, cb)["total_cost_inr"]
                sav.append((f - s) / f)
            rows.append({"wash_cost_inr": base * mlt, "mean_saving_pct": float(np.mean(sav) * 100),
                         "by_site_pct": {key: round(v * 100, 1) for key, v in zip(config.SITES, sav)}})
            print(f"  wash costs Rs {base * mlt:,.0f}: Solara saves {np.mean(sav):.1%} vs 15-day schedule")
    finally:
        config.CLEANING_COST_INR = base
    return rows


def main():
    k = calibrate_rate_constant()
    try:
        from .rl import MODEL_PATH, load_policy
        ppo = load_policy() if MODEL_PATH.exists() else None
    except ImportError:
        ppo = None
    print(f"Soiling constant k = {k:.3f} %/day per unit AOD; one wash = Rs {config.CLEANING_COST_INR:,.0f} per MWp")

    sites = {}
    for key, meta in config.SITES.items():
        site, test, tuned, policies, res, opt = evaluate_site(key, k, ppo)
        fixed = res["fixed_common"]["total_cost_inr"]
        for r in res.values():
            r["saving_vs_fixed_inr"] = fixed - r["total_cost_inr"]
            r["saving_vs_fixed_pct"] = (fixed - r["total_cost_inr"]) / fixed * 100
        sites[key] = {
            **meta,
            "dry_day_soiling_rate_pct": float(test.rate_pct[test.rain_mm < config.RAIN_PARTIAL_MM].mean()),
            "rain_mm_per_year": float(test.rain_mm.sum() / (len(test.dates) / 365.25)),
            "energy_mwh_per_year": float(test.energy_kwh.sum() / 1000 / (len(test.dates) / 365.25)),
            "tuned": {f: p.name for f, p in tuned.items()},
            "policies": res,
            "timeline": timeline(test, policies, opt),
            "recommendation": recommend(site, test, policies["solara"]),
        }
        print(f"\n{meta['name']} ({meta['state']})")
        for name, r in res.items():
            print(f"  {r['label']:42s} washes {r['cleanings']:5.1f}/yr  lost {r['energy_lost_mwh']:6.1f} MWh  "
                  f"cost Rs {r['total_cost_inr']:9,.0f}/yr  vs 15-day {r['saving_vs_fixed_pct']:+6.1f}%")

    names = [n for n in next(iter(sites.values()))["policies"]]
    summary = {n: {m: float(np.mean([s["policies"][n][m] for s in sites.values()]))
                   for m in ("cleanings", "water_l", "energy_lost_mwh", "total_cost_inr",
                             "saving_vs_fixed_inr", "saving_vs_fixed_pct")} for n in names}
    opt_cost = summary["optimum"]["total_cost_inr"]
    for n in names:
        summary[n]["excess_over_optimum_pct"] = (summary[n]["total_cost_inr"] - opt_cost) / opt_cost * 100
    print("\nSensitivity to the cost of one wash:")
    sens = sensitivity(k)

    out = {
        "assumptions": {
            "tariff_inr_per_kwh": config.TARIFF_INR_PER_KWH, "wash_cost_inr_per_mwp": config.CLEANING_COST_INR,
            "water_l_per_mwp": config.WATER_L_PER_MW, "performance_ratio": config.PERFORMANCE_RATIO,
            "mean_dry_day_soiling_rate_pct": config.MEAN_SOILING_RATE_PCT, "soiling_constant_k": k,
            "train_years": [min(config.TRAIN_YEARS), max(config.TRAIN_YEARS)],
            "test_years": [min(config.TEST_YEARS), max(config.TEST_YEARS)],
            "sensor": f"Module 2 {config.MODULE2_MODEL}, daily midday average, zeroed after each wash",
            "seeds": len(SEEDS),
        },
        "sites": sites, "summary": summary, "sensitivity": sens,
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    plot_costs(sites, config.OUT_DIR / "cost_by_policy.png")
    plot_timeline(sites["bhadla"], config.OUT_DIR / "timeline_bhadla.png")
    plot_sensitivity(sens, config.OUT_DIR / "sensitivity.png")
    print("\nAverage over the five parks (per MWp per year):")
    for n, r in summary.items():
        print(f"  {n:13s} cost Rs {r['total_cost_inr']:9,.0f}  washes {r['cleanings']:5.1f}  "
              f"vs 15-day {r['saving_vs_fixed_pct']:+6.1f}%  above optimum {r['excess_over_optimum_pct']:6.1f}%")


def plot_costs(sites, path):
    keys = list(sites)
    names = list(next(iter(sites.values()))["policies"])
    x = np.arange(len(keys))
    w = 0.85 / len(names)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, n in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, [sites[k]["policies"][n]["total_cost_inr"] / 1e5 for k in keys],
               w, label=sites[keys[0]]["policies"][n]["label"])
    ax.set_xticks(x, [f"{sites[k]['name'].split(' ')[0]}\n{sites[k]['state']}" for k in keys])
    ax.set_ylabel("lost revenue + washing, Rs lakh per MWp per year")
    ax.set_title(f"Cost of each cleaning policy, test years {min(config.TEST_YEARS)}-{max(config.TEST_YEARS)}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeline(site, path):
    tl = site["timeline"]
    days = np.arange(len(tl["rain_mm"]))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax2 = ax.twinx()
    ax2.bar(days, tl["rain_mm"], color="tab:blue", alpha=0.25, width=1)
    ax2.set_ylabel("rain (mm/day)")
    for name, c in (("fixed_common", "tab:gray"), ("solara", "tab:orange"), ("optimum", "tab:green")):
        ax.plot(days, tl[name]["loss"], c=c, lw=1.4, label=site["policies"][name]["label"])
        ax.plot(tl[name]["washes"], [-0.6 - 0.6 * ["fixed_common", "solara", "optimum"].index(name)] * len(tl[name]["washes"]),
                "|", c=c, ms=8)
    ax.set_xlabel(f"day of {TIMELINE_YEAR}")
    ax.set_ylabel("soiling loss (%)")
    ax.set_title(f"{site['name']}: soiling loss through {TIMELINE_YEAR} (ticks below zero mark washes)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(sens, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["wash_cost_inr"] for r in sens], [r["mean_saving_pct"] for r in sens], "o-")
    ax.set_xlabel("cost of one wash, Rs per MWp")
    ax.set_ylabel("Solara saving vs 15-day schedule (%)")
    ax.set_title("Sensitivity to washing cost (mean of five parks)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
