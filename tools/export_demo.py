"""Pick the sample photographs for the "Your panels" demo button.

Real DeepSolarEye photographs from three consecutive days (25-27 June 2017)
on which dust built up naturally, three midday photos per day close to that
day's typical reading. The page re-dates them to the last three days so the
schedule runs against the real current forecast, and says so.

Writes website/assets/demo-panels/*.jpg and website/data/demo_panels.json.

Usage:  python -m tools.export_demo
"""
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAYS = ["2017-06-25", "2017-06-26", "2017-06-27"]
HOURS = [10, 12, 14]


def main():
    df = pd.read_csv(ROOT / "data" / "deepsolareye_index.csv")
    df = df[df.usable & (df.irradiance > 0.3)]
    df["ts"] = pd.to_datetime(df.timestamp)
    out_dir = ROOT / "website" / "assets" / "demo-panels"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = []
    for d in DAYS:
        day = df[df.date == d]
        typical = day[day.ts.dt.hour.between(10, 14)].power_loss_pct.median()
        for h in HOURS:
            hours = day.ts.dt.hour
            near = day[(hours - h).abs() == (hours - h).abs().min()]   # nearest hour with photos
            r = near.iloc[(near.power_loss_pct - typical).abs().argsort().iloc[0]]
            name = f"{d}_{r.ts.hour:02d}.jpg"
            if any(x["src"].endswith(name) for x in rows):
                continue                                   # that hour's photo is already in
            shutil.copy(ROOT / r.path, out_dir / name)
            rows.append({"src": f"assets/demo-panels/{name}", "taken": r.ts.isoformat(),
                         "irradiance": round(float(r.irradiance), 3), "measuredLossPct": round(float(r.power_loss_pct), 2)})
    (ROOT / "website" / "data" / "demo_panels.json").write_text(json.dumps({
        "source": "DeepSolarEye dataset (Mehta et al., WACV 2018), CC BY-NC-SA",
        "photos": rows,
    }, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
