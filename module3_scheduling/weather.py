"""Download daily NASA POWER weather for each Indian site.

Parameters: ALLSKY_SFC_SW_DWN (GHI, kWh/m2/day), PRECTOTCORR (rain, mm/day),
T2M (C), RH2M (%), WS2M (m/s), AOD_55 (aerosol optical depth at 550 nm).

Usage:  python -m module3_scheduling.weather
"""
import json
import urllib.parse
import urllib.request

import pandas as pd

from . import config

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "T2M", "RH2M", "WS2M", "AOD_55"]
RENAME = {"ALLSKY_SFC_SW_DWN": "ghi", "PRECTOTCORR": "rain_mm", "T2M": "temp_c",
          "RH2M": "rh", "WS2M": "wind", "AOD_55": "aod"}


def download(key, site):
    q = urllib.parse.urlencode({
        "parameters": ",".join(PARAMS), "community": "RE", "format": "JSON",
        "latitude": site["lat"], "longitude": site["lon"],
        "start": f"{config.WEATHER_START}0101", "end": f"{config.WEATHER_END}1231",
    })
    with urllib.request.urlopen(f"{API}?{q}", timeout=180) as r:
        raw = json.load(r)
    p = raw["properties"]["parameter"]
    df = pd.DataFrame({RENAME[k]: pd.Series(p[k]) for k in PARAMS})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.mask(df <= -999)          # NASA POWER fill value
    return df


def load(key):
    return pd.read_csv(config.WEATHER_DIR / f"{key}.csv", index_col="date", parse_dates=True)


def main():
    config.WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    for key, site in config.SITES.items():
        df = download(key, site)
        missing = df.isna().mean()
        df = df.interpolate(limit_direction="both")
        df.to_csv(config.WEATHER_DIR / f"{key}.csv")
        print(f"{site['name']:32s} {len(df)} days | GHI {df.ghi.mean():.2f} kWh/m2/d | "
              f"rain {df.rain_mm.sum() / (len(df) / 365.25):.0f} mm/yr | AOD {df.aod.mean():.2f} | "
              f"missing filled: {', '.join(f'{c} {v:.1%}' for c, v in missing.items() if v > 0) or 'none'}")


if __name__ == "__main__":
    main()
