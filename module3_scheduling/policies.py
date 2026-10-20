"""Cleaning policies. Each sees only what a plant operator would have each
morning: the smoothed Module 2 loss estimate, its learned soiling rate, the
date, recent energy yield and a short rain forecast."""
import numpy as np

from . import config


class Policy:
    name = "policy"

    def reset(self):
        self.since = 0

    def decide(self, obs):
        clean = self._decide(obs)
        self.since = 0 if clean else self.since + 1
        return clean


class NeverClean(Policy):
    name = "Rain only (never wash)"

    def _decide(self, obs):
        return False


class FixedInterval(Policy):
    """Wash every `days` days, whatever the panels look like (common practice)."""

    def __init__(self, days):
        self.days = days
        self.name = f"Fixed, every {days} days"

    def _decide(self, obs):
        return self.since + 1 >= self.days


class Threshold(Policy):
    """Wash when Module 2's estimated loss passes a set percentage."""

    def __init__(self, pct):
        self.pct = pct
        self.name = f"Threshold, {pct:g}% loss"

    def _decide(self, obs):
        return self.since >= 1 and obs["estimate"] >= self.pct


class CostBenefit(Policy):
    """Solara's rule: wash when the money a wash would win back over the next
    `horizon` days is worth more than the wash itself.

    Value of washing today = (estimated loss - residual after washing)
                             x expected daily energy x tariff x useful days
    Useful days end early if the forecast shows rain heavy enough to clean the
    panels for free, so the policy waits for the rain instead of paying.
    """

    def __init__(self, horizon):
        self.horizon = horizon
        self.name = f"Solara cost-benefit (horizon {horizon} d)"

    def _decide(self, obs):
        if self.since < 1:
            return False
        rain = obs["rain_forecast_mm"]
        heavy = np.nonzero(rain >= config.RAIN_FULL_MM)[0]
        useful = min(self.horizon, int(heavy[0]) if len(heavy) else self.horizon)
        # Dust keeps settling whether or not we wash, so washing today saves the
        # same gap (current loss - residual) on every day until the next wash.
        gain_pct = obs["estimate"] - config.RESIDUAL_AFTER_CLEANING_PCT
        value = gain_pct / 100 * obs["mean_energy_kwh"] * config.TARIFF_INR_PER_KWH * useful
        return value >= config.CLEANING_COST_INR


def tuning_grid():
    """Parameter grids searched on the training years for each site."""
    return {
        "fixed": [FixedInterval(d) for d in range(3, 61)],
        "threshold": [Threshold(p) for p in np.arange(1.0, 20.01, 0.5)],
        "cost_benefit": [CostBenefit(h) for h in range(3, 91)],
    }
