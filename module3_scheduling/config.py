"""Settings for Module 3 (intelligent cleaning scheduling), based in India.

Every number here is an assumption you can change; the README lists the
sources behind each one. Costs are per MWp of installed capacity.
"""
from module1_detection import config as m1

ROOT = m1.ROOT
WEATHER_DIR = m1.DATA_DIR / "weather"          # NASA POWER downloads, not committed
OUT_DIR = ROOT / "module3_scheduling" / "outputs"
MODULE2_PREDICTIONS = ROOT / "module2_powerloss" / "outputs" / "test_predictions.csv"
MODULE2_MODEL = "cnn"             # Module 2 model inside the sensor: most accurate on clean panels

# Indian utility-scale solar parks (NASA POWER grid cells are 0.5 degrees).
SITES = {
    "bhadla":   {"name": "Bhadla Solar Park",            "state": "Rajasthan",      "lat": 27.54, "lon": 71.92},
    "charanka": {"name": "Charanka Solar Park",          "state": "Gujarat",        "lat": 23.90, "lon": 71.18},
    "rewa":     {"name": "Rewa Ultra Mega Solar",        "state": "Madhya Pradesh", "lat": 24.52, "lon": 81.53},
    "kurnool":  {"name": "Kurnool Ultra Mega Solar Park", "state": "Andhra Pradesh", "lat": 15.60, "lon": 78.30},
    "pavagada": {"name": "Pavagada Solar Park",          "state": "Karnataka",      "lat": 14.25, "lon": 77.43},
}
WEATHER_START, WEATHER_END = 2005, 2024
TRAIN_YEARS = range(2005, 2020)    # tune every policy here
TEST_YEARS = range(2020, 2025)     # report results here

# ---- plant ------------------------------------------------------------------------
PLANT_MWP = 1.0                    # results are per MWp
PERFORMANCE_RATIO = 0.80           # losses other than soiling (inverter, wiring, heat)

# ---- money (INR) ---------------------------------------------------------------------
TARIFF_INR_PER_KWH = 2.60          # value of energy lost: recent SECI auction level
CLEANING_LABOUR_INR_PER_MW = 8000  # one manual wash of 1 MWp (about 2,000 modules)
WATER_L_PER_MW = 15000             # litres per wash per MWp (reported 7,000-24,000)
WATER_INR_PER_L = 0.10             # tanker water in arid districts, about Rs 100 per kL
CLEANING_COST_INR = CLEANING_LABOUR_INR_PER_MW + WATER_L_PER_MW * WATER_INR_PER_L

# ---- soiling physics -------------------------------------------------------------
# Daily soiling rate on dry days is proportional to the aerosol optical depth
# (AOD, NASA POWER AOD_55) at the site, scaled so the average dry-day rate
# across the five sites is 0.40 %/day, the middle of Indian field measurements
# (Delhi 0.3-0.6, Gandhinagar 0.37 +/- 0.09, north India up to 0.47 %/day).
MEAN_SOILING_RATE_PCT = 0.40
SOILING_RATE_BOUNDS_PCT = (0.05, 1.50)
MAX_SOILING_LOSS_PCT = 60.0        # dust layers saturate
# Rain cleans panels: heavy rain washes almost everything off, light rain a part.
RAIN_FULL_MM = 5.0                 # >= this: 90 % of the dust removed
RAIN_PARTIAL_MM = 1.0              # between: 40 % removed; below: no effect
RAIN_FULL_CLEAN, RAIN_PARTIAL_CLEAN = 0.90, 0.40
RESIDUAL_AFTER_CLEANING_PCT = 0.3  # manual washing is not perfect

# ---- what the scheduler can see -------------------------------------------------
# Each morning a camera photographs the panels and Module 2 estimates the loss.
# Its estimate carries Module 2's real test-set error for that level of soiling.
RAIN_FORECAST_DAYS = 3             # the cost-benefit policy may use a 3-day forecast

# ---- reinforcement learning -------------------------------------------------------
PPO_TIMESTEPS = 400_000
SEED = m1.SEED
