# Solara

**AI-based solar panel soiling detection, power-loss prediction and intelligent cleaning scheduling.**

Dust, pollen and bird droppings on photovoltaic panels cut their output, sometimes by more than 40% in dusty regions. Most sites still clean on a fixed calendar, which wastes water when panels are clean and loses energy when they are dirty. Solara replaces the calendar with three connected models:

| Module | What it does | Status |
|---|---|---|
| 1. Detect | CNN that classifies a panel image by soiling severity | Done |
| 2. Predict | Regression of the exact power loss (%) caused by the soiling | Planned |
| 3. Schedule | Cost-benefit / learned cleaning schedule vs. fixed-interval baseline | Planned |

The project website (`website/`) is a single page with a real-time 3D solar panel lit by a sun that follows your cursor.

---

## Module 1: soiling detection

### Data: DeepSolarEye

[DeepSolarEye](https://deep-solar-eye.github.io/) (Mehta et al., WACV 2018) has 45,754 RGB images (192×192) of one solar panel photographed over several weeks while dust settled and was cleaned away. Each filename encodes the timestamp, the **power loss** relative to a clean reference panel, and the **irradiance**:

```
solar_Fri_Jun_16_10__0__11_2017_L_0.906153208302_I_0.0058574424375.jpg
```

The dataset has no soiling-*type* labels, so Solara classifies soiling **severity** by binning the measured power loss:

| Class | Power loss |
|---|---|
| clean | < 5% |
| light | 5 – 15% |
| moderate | 15 – 30% |
| heavy | > 30% |

Frames taken in very dim light (normalised irradiance < 0.05) have unreliable loss labels and are excluded.

**Split.** By default the usable images are split at random, stratified by class: 20% for testing, and of the remaining 80%, 10% for validation (used to pick the best epoch), giving 72 / 8 / 20.

A stricter option is in `config.py`: `SPLIT_MODE = "day"` keeps whole days together, so near-identical frames taken minutes apart never appear in both train and test. Expect lower but more conservative scores with it.

### Model

- **Backbone:** MobileNetV2 pre-trained on ImageNet (light enough for edge devices such as a Jetson Nano or Raspberry Pi).
- **Head:** Dropout → Linear(1280→256) → ReLU → Dropout → Linear(256→4).
- **Training:** 3 epochs with the backbone frozen (head only, lr 1e-3), then 12 epochs fine-tuning the whole network (lr 1e-4, cosine decay), AdamW, label smoothing 0.05, mixed precision.
- **Class imbalance:** weighted random sampling so each severity is seen equally often.
- **Augmentation:** rotation, horizontal flip, brightness/contrast/saturation jitter, Gaussian noise (robustness to lighting and camera angle).
- **Model selection:** the checkpoint with the best validation macro-F1.

### Results

<!-- RESULTS:START -->
Test set: 8,722 held-out images (stratified random 20% split).

| Metric | Value |
|---|---|
| Accuracy | **79.9%** |
| Macro precision | 0.742 |
| Macro recall | 0.750 |
| Macro F1 | **0.733** |
| Majority-class baseline accuracy | 35.7% |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| clean | 0.77 | 0.91 | 0.83 | 2,822 |
| light | 0.81 | 0.59 | 0.69 | 2,078 |
| moderate | 0.39 | 0.62 | 0.48 | 705 |
| heavy | 0.99 | 0.88 | 0.93 | 3,117 |

![Confusion matrix](module1_detection/outputs/confusion_matrix.png)
![Training curves](module1_detection/outputs/training_curves.png)

**Reading the results.** Clean and heavy soiling are recognised reliably (F1 0.83 and 0.93). The middle classes are harder: their power-loss ranges are narrow (5 to 15% and 15 to 30%), so images near a boundary look alike, and *moderate* is the rarest class (8% of images). Most errors fall in a neighbouring class rather than a distant one. Validation F1 levels off after about 10 epochs with no sign of overfitting.

Possible next steps: predict the loss percentage directly (Module 2) and bin it, higher input resolution, or a larger backbone (EfficientNet, ResNet-50).
<!-- RESULTS:END -->

### Run it yourself

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -m module1_detection.prepare_data --download   # download, unzip, index, split
python -m module1_detection.train                     # fine-tune MobileNetV2 (GPU recommended)
python -m module1_detection.evaluate                  # metrics + figures in module1_detection/outputs/
python -m module1_detection.export_site                # publish results to the website
python -m module1_detection.predict path/to/panel.jpg # classify your own image (trained weights are included)
```

Settings (class thresholds, irradiance cut-off, split mode, epochs, learning rates) live in [`module1_detection/config.py`](module1_detection/config.py).

---

## Website

```bash
python -m http.server 8000 --directory website
# open http://localhost:8000
```

Built with three.js. The solar panel is generated procedurally: 60 monocrystalline cells with busbars and fingers, white backsheet, anodised aluminium frame and anti-reflective glass. The **Detect** tab lets you dust the panel at each severity level.

## Repository layout

```
module1_detection/   data preparation, model, training, evaluation, prediction
  outputs/           metrics.json, confusion matrix, training curves, sample predictions
website/             project website (three.js)
data/                dataset (downloaded, not committed)
```

## References

1. S. Mehta, A. P. Azad, S. A. Chemmengath, V. Raykar, S. Kalyanaraman, "DeepSolarEye: Power Loss Prediction and Weakly Supervised Soiling Localization via Fully Convolutional Networks for Solar Panels," *WACV*, 2018.
2. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, 2018.

## Author

Ritu Ramesh, B.Tech Artificial Intelligence and Data Science, Amrita Vishwa Vidyapeetham, with Dr. Gargi.

Code: MIT licence. The DeepSolarEye dataset belongs to its authors and is used under CC BY-NC-SA. It is not redistributed here.
