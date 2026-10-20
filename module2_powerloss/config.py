"""Settings for Module 2 (power-loss prediction).

The dataset index, image paths and train/val/test split are shared with
Module 1 (module1_detection/config.py), so both modules are scored on the
same held-out images.
"""
from module1_detection import config as m1

ROOT = m1.ROOT
INDEX_CSV = m1.INDEX_CSV
OUT_DIR = ROOT / "module2_powerloss" / "outputs"
CHECKPOINT = OUT_DIR / "powerloss_cnn.pt"
EMBEDDINGS = m1.DATA_DIR / "module1_embeddings.npz"   # cached, not committed
MODULE1_CHECKPOINT = m1.CHECKPOINT

SEED = m1.SEED
IMG_SIZE = m1.IMG_SIZE
BATCH_SIZE = 64
NUM_WORKERS = 4

# Environmental inputs available for every DeepSolarEye image.
#   irradiance       normalised solar irradiance at capture time
#   tod_sin, tod_cos time of day on a circle (sun angle, shadows, glare)
ENV_FEATURES = ["irradiance", "tod_sin", "tod_cos"]

# CNN regressor training: head + environment branch first, then everything.
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 12
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4

# Tree models on Module 1 embeddings: PCA size of the image embedding.
PCA_COMPONENTS = 64
