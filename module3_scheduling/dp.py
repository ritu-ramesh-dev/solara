"""Hindsight-optimal cleaning schedule by dynamic programming.

Knows the true soiling loss and all future weather, so no real policy can beat
it. It is the yardstick: how close does each policy get to the best possible?

State: soiling loss on a 0.05-point grid. Each day choose wash / wait to
minimise lost revenue + washing cost over the whole period (Bellman recursion,
solved backwards in time).
"""
import numpy as np

from . import config
from .sim import Result, rain_clean_fraction

GRID = np.arange(0, config.MAX_SOILING_LOSS_PCT + 1e-9, 0.05)


def solve(site, start_loss=0.0):
    n, C, p = len(site.dates), config.CLEANING_COST_INR, config.TARIFF_INR_PER_KWH
    res = config.RESIDUAL_AFTER_CLEANING_PCT
    V = np.zeros_like(GRID)                     # cost-to-go after the last day
    wash = np.zeros((n, len(GRID)), bool)
    for t in range(n - 1, -1, -1):
        dry = site.rain_mm[t] < config.RAIN_PARTIAL_MM
        keep = 1 - rain_clean_fraction(site.rain_mm[t])

        def next_loss(L):
            return np.minimum(L + (site.rate_pct[t] if dry else 0), config.MAX_SOILING_LOSS_PCT) * keep

        day = site.energy_kwh[t] / 100 * p
        cost_wait = day * GRID + np.interp(next_loss(GRID), GRID, V)
        cost_wash = C + day * res + np.interp(next_loss(np.array([res])), GRID, V)[0]
        wash[t] = cost_wash < cost_wait
        V = np.minimum(cost_wait, cost_wash)

    # forward pass with the true dynamics
    loss, lost, cleanings = start_loss, 0.0, 0
    trace, cleaned = np.zeros(n), np.zeros(n, bool)
    for t in range(n):
        if wash[t, int(round(min(loss, GRID[-1]) / 0.05))]:
            loss, cleanings, cleaned[t] = res, cleanings + 1, True
        trace[t] = loss
        lost += site.energy_kwh[t] * loss / 100
        dry = site.rain_mm[t] < config.RAIN_PARTIAL_MM
        loss = min(loss + (site.rate_pct[t] if dry else 0), config.MAX_SOILING_LOSS_PCT)
        loss *= 1 - rain_clean_fraction(site.rain_mm[t])
    return Result(cleanings, lost, lost * p, cleanings * C, n / 365.25, trace, cleaned)
