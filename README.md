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

**Leak-free split.** Consecutive frames, minutes apart, are almost identical. A random image-level split would put near-copies in both train and test and inflate accuracy. Solara splits by **whole days** (70 / 15 / 15), so the model is tested on days it never saw.

### Model

- **Backbone:** MobileNetV2 pre-trained on ImageNet (light enough for edge devices such as a Jetson Nano or Raspberry Pi).
- **Head:** Dropout → Linear(1280→256) → ReLU → Dropout → Linear(256→4).
- **Training:** 3 epochs with the backbone frozen (head only, lr 1e-3), then 12 epochs fine-tuning the whole network (lr 1e-4, cosine decay), AdamW, label smoothing 0.05, mixed precision.
- **Class imbalance:** weighted random sampling so each severity is seen equally often.
- **Augmentation:** rotation, horizontal flip, brightness/contrast/saturation jitter, Gaussian noise (robustness to lighting and camera angle).
- **Model selection:** the checkpoint with the best validation macro-F1.

### Results

<!-- RESULTS:START -->
Results are added here after training.
<!-- RESULTS:END -->

### Run it yourself

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -m module1_detection.prepare_data --download   # download, unzip, index, split by day
python -m module1_detection.train                     # fine-tune MobileNetV2 (GPU recommended)
python -m module1_detection.evaluate                  # metrics + figures in module1_detection/outputs/
python -m module1_detection.predict path/to/panel.jpg # classify your own image
```

Settings (class thresholds, irradiance cut-off, epochs, learning rates) live in [`module1_detection/config.py`](module1_detection/config.py).

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
