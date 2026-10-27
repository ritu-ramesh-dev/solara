"""Make phone-style test photographs for tools/site_tests.mjs.

Writes to the folder given (default: a temp folder):
  user_*.jpg   six 1280 x 960 photos (non-square, like a phone) made from
               real DeepSolarEye panels, with camera EXIF dates on the last
               three days, two per day
  tiny.jpg     120 x 120: below the 192 px minimum
  dark.jpg     an almost black frame: refused as too dark
and prints the EXIF dates as JSON.

Usage:  python -m tools.make_test_photos <folder>
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["2017-06-25_10", "2017-06-25_12", "2017-06-26_10", "2017-06-26_12", "2017-06-27_10", "2017-06-27_12"]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="solara-photos-"))
    out.mkdir(parents=True, exist_ok=True)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = []
    for i, name in enumerate(SOURCES):
        panel = Image.open(ROOT / "website/assets/demo-panels" / f"{name}.jpg").convert("RGB").resize((960, 960), Image.LANCZOS)
        canvas = Image.new("RGB", (1280, 960), (70, 74, 78))          # phone frame: panel in the middle
        canvas.paste(panel, (160, 0))
        taken = today - timedelta(days=2 - i // 2) + timedelta(hours=11 + (i % 2), minutes=7 * i)
        exif = Image.Exif()
        exif.get_ifd(0x8769)[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
        canvas.save(out / f"user_{i}.jpg", quality=90, exif=exif)
        dates.append(taken.strftime("%Y-%m-%dT%H:%M"))
    Image.new("RGB", (120, 120), (30, 40, 70)).save(out / "tiny.jpg")
    Image.new("RGB", (800, 600), (6, 6, 8)).save(out / "dark.jpg")
    manifest = {"folder": str(out), "files": [f"user_{i}.jpg" for i in range(len(SOURCES))], "dates": dates}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
