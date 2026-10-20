"""Download (optional), unzip and index the DeepSolarEye dataset.

DeepSolarEye stores its labels in the filenames, e.g.

    solar_Fri_Jun_16_10__0__11_2017_L_0.906153208302_I_0.0058574424375.jpg
          day mon dd hh  mm  ss yyyy   power loss       irradiance

This script turns every filename into a row of data/deepsolareye_index.csv with
columns: path, timestamp, date, power_loss_pct, irradiance, severity, split.

Usage:
    python -m module1_detection.prepare_data            # zip already in data/raw
    python -m module1_detection.prepare_data --download # fetch it from Google Drive first
"""
import argparse
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

from . import config

NAME_RE = re.compile(
    r"solar_(?P<dow>[A-Za-z]{3})_(?P<mon>[A-Za-z]{3})_(?P<day>\d{1,2})_"
    r"(?P<hh>\d{1,2})__(?P<mm>\d{1,2})__(?P<ss>\d{1,2})_(?P<year>\d{4})"
    r"_L_(?P<loss>[-\d.eE]+)_I_(?P<irr>[-\d.eE]+)\.(?:jpg|jpeg|png)$"
)


def download():
    import gdown

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download(id=config.DRIVE_FILE_ID, output=str(config.RAW_DIR) + "/", quiet=False)


def extract():
    if config.IMAGE_DIR.exists() and any(config.IMAGE_DIR.rglob("*.jpg")):
        return
    zips = sorted(config.RAW_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zip in {config.RAW_DIR}. Run with --download or place the zip there.")
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for z in zips:
        print(f"Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(config.IMAGE_DIR)


def parse_name(name):
    m = NAME_RE.search(name)
    if not m:
        return None
    g = m.groupdict()
    ts = datetime.strptime(
        f"{g['year']} {g['mon']} {int(g['day']):02d} {int(g['hh']):02d}:{int(g['mm']):02d}:{int(g['ss']):02d}",
        "%Y %b %d %H:%M:%S",
    )
    return ts, float(g["loss"]), float(g["irr"])


def loss_to_severity(loss_pct):
    return int(np.digitize(loss_pct, config.LOSS_BINS))


def split_random(df, seed=config.SEED):
    """The usual stratified random split by image: TEST_FRACTION of the usable
    images go to test, and VAL_FRACTION of the remainder to validation, with
    each class kept in the same proportion in every split."""
    from sklearn.model_selection import train_test_split

    split = pd.Series("train", index=df.index)
    usable = df[df["usable"]]
    rest, test = train_test_split(usable.index, test_size=config.TEST_FRACTION,
                                  stratify=usable["severity"], random_state=seed)
    train, val = train_test_split(rest, test_size=config.VAL_FRACTION,
                                  stratify=usable.loc[rest, "severity"], random_state=seed)
    split[val] = "val"
    split[test] = "test"
    return split.tolist()


def split_by_day(df, seed=config.SEED, trials=200_000):
    """Assign whole days to train/val/test so neighbouring frames never leak.

    Each day of DeepSolarEye has its own soiling pattern (some days are almost
    all heavy, others almost all clean), so a plain random choice of days gives
    very unbalanced splits. Instead, many random day assignments are scored and
    the one whose val and test sets best match the overall class mix and the
    target sizes is kept. The search is seeded, so the split is reproducible.
    """
    usable = df[df["usable"]]
    counts = pd.crosstab(usable["date"], usable["severity"]).reindex(
        columns=range(len(config.CLASS_NAMES)), fill_value=0)
    days = counts.index.to_numpy()
    c = counts.to_numpy(dtype=float)
    total = c.sum(0)
    overall = total / total.sum()
    _, f_val, f_test = config.DAY_SPLIT_FRACTIONS

    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(trials):
        lab = rng.choice(3, size=len(days), p=config.DAY_SPLIT_FRACTIONS)  # 0 train, 1 val, 2 test
        score = 0.0
        for k, frac in ((1, f_val), (2, f_test)):
            part = c[lab == k].sum(0)
            n = part.sum()
            if n == 0 or (part / total).min() < frac * 0.35:  # every class must be present
                score = np.inf
                break
            score += np.abs(part / n - overall).sum() + 2 * abs(n / total.sum() - frac)
        if score < best_score:
            best, best_score = lab.copy(), score
    names = np.array(["train", "val", "test"])
    which = dict(zip(days, names[best]))
    return df["date"].map(which).tolist()


def build_index():
    rows, skipped = [], 0
    for p in config.IMAGE_DIR.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parsed = parse_name(p.name)
        if parsed is None:
            skipped += 1
            continue
        ts, loss, irr = parsed
        rows.append({
            "path": str(p.relative_to(config.ROOT)).replace("\\", "/"),
            "timestamp": ts,
            "date": ts.date().isoformat(),
            "power_loss_raw": loss,
            "irradiance": irr,
        })
    if not rows:
        raise SystemExit("No images matched the DeepSolarEye filename pattern.")
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # The loss label is stored as a fraction (0-1); convert to a percentage.
    scale = 100.0 if df["power_loss_raw"].max() <= 1.5 else 1.0
    df["power_loss_pct"] = (df["power_loss_raw"] * scale).clip(0, 100)
    df["severity"] = df["power_loss_pct"].apply(loss_to_severity)
    df["severity_name"] = df["severity"].map(dict(enumerate(config.CLASS_NAMES)))
    df["usable"] = df["irradiance"] >= config.MIN_IRRADIANCE
    df["split"] = split_by_day(df) if config.SPLIT_MODE == "day" else split_random(df)

    df.to_csv(config.INDEX_CSV, index=False)
    print(f"Indexed {len(df)} images ({skipped} skipped) -> {config.INDEX_CSV}")
    print(f"Usable (irradiance >= {config.MIN_IRRADIANCE}): {df['usable'].sum()}")
    print(df[df.usable].groupby(["split", "severity_name"]).size().unstack(fill_value=0)[config.CLASS_NAMES])
    print(f"Split mode: {config.SPLIT_MODE}")
    if config.SPLIT_MODE == "day":
        for s in ("train", "val", "test"):
            print(f"{s:5s} days:", ", ".join(sorted(df.loc[df.split == s, "date"].unique())))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
    extract()
    build_index()
