"""Soiling, rain, energy and cost simulation for one Indian site.

Each day:
  1. morning: the camera + Module 2 give a noisy estimate of the soiling loss;
     the policy decides whether to wash the panels today
  2. if washed: loss drops to the residual, the cleaning cost is paid
  3. the day's energy is produced; the share lost to soiling is lost revenue
  4. dust settles (rate proportional to that day's aerosol optical depth)
  5. rain washes part or nearly all of the dust off
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .weather import load


# ---- soiling-rate calibration -----------------------------------------------------
def calibrate_rate_constant():
    """k such that k * AOD averages MEAN_SOILING_RATE_PCT on dry training days."""
    aods = []
    for key in config.SITES:
        w = load(key)
        w = w[w.index.year.isin(config.TRAIN_YEARS)]
        aods.append(w.loc[w.rain_mm < config.RAIN_PARTIAL_MM, "aod"])
    return config.MEAN_SOILING_RATE_PCT / pd.concat(aods).mean()


@dataclass
class Site:
    key: str
    dates: pd.DatetimeIndex
    energy_kwh: np.ndarray     # daily energy of a clean 1 MWp plant
    rate_pct: np.ndarray       # dust added per day (percentage points of loss)
    rain_mm: np.ndarray

    def years(self, years):
        m = np.isin(self.dates.year, list(years))
        return Site(self.key, self.dates[m], self.energy_kwh[m], self.rate_pct[m], self.rain_mm[m])


def load_site(key, k=None):
    k = calibrate_rate_constant() if k is None else k
    w = load(key)
    rate = np.clip(k * w["aod"].to_numpy(), *config.SOILING_RATE_BOUNDS_PCT)
    energy = w["ghi"].to_numpy() * config.PLANT_MWP * 1000 * config.PERFORMANCE_RATIO
    return Site(key, w.index, energy, rate, w["rain_mm"].to_numpy())


def rain_clean_fraction(rain_mm):
    if rain_mm >= config.RAIN_FULL_MM:
        return config.RAIN_FULL_CLEAN
    if rain_mm >= config.RAIN_PARTIAL_MM:
        return config.RAIN_PARTIAL_CLEAN
    return 0.0


def step_loss(loss, rate, rain_mm):
    """Overnight change: dust settles, then rain washes some of it away."""
    loss = min(loss + (rate if rain_mm < config.RAIN_PARTIAL_MM else 0.0), config.MAX_SOILING_LOSS_PCT)
    return loss * (1 - rain_clean_fraction(rain_mm))


# ---- what the camera + Module 2 report -----------------------------------------------
class Module2Sensor:
    """The plant's soiling sensor: a camera photographs the panels through the
    day and Module 2 estimates the loss in each photo; the daily reading is
    the average over the midday photos (09:00-15:00).

    Its error is taken from Module 2's real test-set behaviour. Averaging
    removes most per-photo noise, but part of the error is shared by all
    photos of a day. For each loss band we measure, over the test days, the
    mean (bias) and spread of the daily-average error, then interpolate
    between bands. Typical result for the CNN: about +2.4 points of bias and
    1.3 points of day-to-day spread on nearly clean panels.
    """
    def __init__(self, rng, model=config.MODULE2_MODEL):
        self.loss, self.bias, self.sd = _sensor_error_model(model)
        self.rng = rng

    def read(self, true_loss):
        bias = np.interp(true_loss, self.loss, self.bias)
        sd = np.interp(true_loss, self.loss, self.sd)
        return float(np.clip(true_loss + bias + self.rng.normal(0, sd), 0, 100))


@lru_cache(maxsize=None)
def _sensor_error_model(model, band_edges=(0, 5, 10, 15, 30, 101)):
    """Daily-average error of a Module 2 model per loss band: (loss, bias, sd)."""
    idx = pd.read_csv(config.ROOT / "data" / "deepsolareye_index.csv", usecols=["path", "timestamp"])
    p = pd.read_csv(config.MODULE2_PREDICTIONS).merge(idx, on="path")
    ts = pd.to_datetime(p["timestamp"])
    p = p[ts.dt.hour.between(9, 15)].assign(day=ts.dt.date)
    p["err"] = p[model] - p["power_loss_pct"]
    p["band"] = pd.cut(p["power_loss_pct"], list(band_edges), include_lowest=True)
    daily = p.groupby(["day", "band"], observed=True).agg(
        err=("err", "mean"), loss=("power_loss_pct", "mean"), n=("err", "size"))
    daily = daily[daily["n"] >= 20]
    stats = daily.groupby(level="band", observed=True).agg(
        loss=("loss", "mean"), bias=("err", "mean"), sd=("err", "std"))
    return tuple(stats[c].to_numpy() for c in ("loss", "bias", "sd"))


class LossEstimate:
    """What the operator believes the loss is.

    * Zeroing: the morning after a wash the panels are known to be clean, so
      the reading then (minus the residual dust) is the sensor's offset, which
      is subtracted from later readings.
    * Smoothing: an exponential average, which also predicts today's value
      from yesterday's plus the learned daily soiling rate.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = config.RESIDUAL_AFTER_CLEANING_PCT
        self.rate = config.MEAN_SOILING_RATE_PCT
        self.offset = 0.0
        self.zero_next = True

    def update(self, reading, rained):
        if self.zero_next:
            self.offset = reading - config.RESIDUAL_AFTER_CLEANING_PCT
            self.zero_next = False
        r = max(reading - self.offset, 0.0)
        if rained:
            self.value = r
        else:
            predicted = self.value + self.rate
            new = (1 - self.alpha) * predicted + self.alpha * r
            # learn the site's dry-day soiling rate slowly from the trend
            self.rate = float(np.clip(0.95 * self.rate + 0.05 * (new - self.value), 0.05, 1.5))
            self.value = new
        return self.value

    def cleaned(self):
        self.value = config.RESIDUAL_AFTER_CLEANING_PCT
        self.zero_next = True


# ---- running a policy ------------------------------------------------------------------
@dataclass
class Result:
    cleanings: int
    energy_lost_kwh: float
    lost_revenue_inr: float
    cleaning_cost_inr: float
    years: float
    loss_trace: np.ndarray
    clean_days: np.ndarray

    @property
    def total_cost_inr(self):
        return self.lost_revenue_inr + self.cleaning_cost_inr

    def per_year(self):
        y = self.years
        return {
            "cleanings": self.cleanings / y,
            "water_l": self.cleanings * config.WATER_L_PER_MW / y,
            "energy_lost_mwh": self.energy_lost_kwh / 1000 / y,
            "lost_revenue_inr": self.lost_revenue_inr / y,
            "cleaning_cost_inr": self.cleaning_cost_inr / y,
            "total_cost_inr": self.total_cost_inr / y,
            "mean_loss_pct": float(self.loss_trace.mean()),
        }


def run(site, policy, seed=0, start_loss=0.0):
    """Simulate `policy` on `site`. The policy sees only the sensor estimate,
    the date, the expected energy and a short rain forecast."""
    rng = np.random.default_rng(seed)
    sensor, est = Module2Sensor(rng), LossEstimate()
    policy.reset()
    n = len(site.dates)
    loss, lost_kwh, cleanings = start_loss, 0.0, 0
    trace, cleaned = np.zeros(n), np.zeros(n, bool)
    rained_yesterday = False
    for t in range(n):
        reading = sensor.read(loss)
        estimate = est.update(reading, rained_yesterday)
        f = config.RAIN_FORECAST_DAYS
        obs = {
            "t": t, "date": site.dates[t], "estimate": estimate, "rate": est.rate,
            "energy_kwh": site.energy_kwh[t], "mean_energy_kwh": float(site.energy_kwh[max(0, t - 30):t + 1].mean()),
            "rain_forecast_mm": site.rain_mm[t:t + f].copy(),
        }
        if policy.decide(obs):
            loss = config.RESIDUAL_AFTER_CLEANING_PCT
            est.cleaned()
            cleanings += 1
            cleaned[t] = True
        trace[t] = loss
        lost_kwh += site.energy_kwh[t] * loss / 100
        rained_yesterday = site.rain_mm[t] >= config.RAIN_PARTIAL_MM
        loss = step_loss(loss, site.rate_pct[t], site.rain_mm[t])
    return Result(cleanings, lost_kwh, lost_kwh * config.TARIFF_INR_PER_KWH,
                  cleanings * config.CLEANING_COST_INR, n / 365.25, trace, cleaned)
