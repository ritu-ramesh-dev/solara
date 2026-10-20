"""Shared settings for Module 1 (soiling detection & severity classification)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"          # where the DeepSolarEye .jpg files end up
INDEX_CSV = DATA_DIR / "deepsolareye_index.csv"
OUT_DIR = ROOT / "module1_detection" / "outputs"
CHECKPOINT = OUT_DIR / "mobilenetv2_soiling.pt"

# DeepSolarEye Google Drive file id (https://deep-solar-eye.github.io/)
DRIVE_FILE_ID = "1qB5dPWZMi2-12sLHDykHb9i6GibbJ46l"

# Severity classes are made by binning the power-loss label (a percentage).
# Upper bounds of each bin; the last class takes everything above.
CLASS_NAMES = ["clean", "light", "moderate", "heavy"]
LOSS_BINS = [5.0, 15.0, 30.0]

# Frames shot in very dim light have unreliable power-loss labels.
MIN_IRRADIANCE = 0.05

# How images are divided into train / val / test.
#   "random": the usual stratified random split by image (default).
#             TEST_FRACTION of all images go to test; VAL_FRACTION of the rest
#             is held out for validation (choosing the best epoch).
#   "day":    whole days go to one split, so near-identical frames taken
#             minutes apart never sit on both sides (a stricter test).
SPLIT_MODE = "random"
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
DAY_SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
HEAD_EPOCHS = 3        # phase 1: train only the new classifier head
FINETUNE_EPOCHS = 12   # phase 2: unfreeze the backbone
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
