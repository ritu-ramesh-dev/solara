# Solara

**AI-based solar panel soiling detection, power-loss prediction and intelligent cleaning scheduling.**

Dust, pollen and bird droppings on photovoltaic panels cut their output, sometimes by more than 40% in dusty regions. Most sites still clean on a fixed calendar, which wastes water when panels are clean and loses energy when they are dirty. Solara replaces the calendar with three connected models:

| Module | What it does | Status |
|---|---|---|
| 1. Detect | CNN that classifies a panel image by soiling severity | Done |
| 2. Predict | Regression of the exact power loss (%) caused by the soiling | Done |
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

## Module 2: power-loss prediction

Module 2 predicts the **percentage of power lost** to soiling, a continuous number, instead of a class. Inputs are the panel photograph plus the environmental data DeepSolarEye records for every image: **irradiance** and **time of day** (encoded as sine and cosine so 23:59 sits next to 00:00). It uses the same images and the same train/val/test split as Module 1, so the scores are directly comparable.

### Models

| Model | Inputs | How |
|---|---|---|
| Mean baseline | none | always predicts the training mean |
| XGBoost, environment only | irradiance, time | shows what can be predicted *without* looking at the panel |
| Random Forest | image + env | 1280-d Module 1 embedding, reduced to 64 PCA components, plus environment |
| XGBoost | image + env | same features; early stopping on validation MAE |
| CNN regressor | image + env | MobileNetV2 initialised from the Module 1 classifier, image features fused with a small environment branch, sigmoid output; Smooth L1 loss; frozen-backbone phase then full fine-tune |

### Results

Test set: 8,722 images. Errors are in **percentage points** of power loss.

| Model | MAE | RMSE | R² | Binned accuracy | Binned macro F1 |
|---|---|---|---|---|---|
| Mean baseline | 25.25 | 29.00 | -0.000 | 8.1% | 0.037 |
| XGBoost, environment only | 13.49 | 19.85 | 0.532 | 51.0% | 0.457 |
| Random Forest, image + env | 3.79 | 7.64 | 0.931 | 75.7% | 0.695 |
| XGBoost, image + env | 3.85 | 7.54 | 0.932 | 76.9% | 0.703 |
| CNN regressor, image + env | 3.87 | 8.29 | 0.918 | 80.8% | 0.743 |

*Binned* means the predicted loss is cut at Module 1's thresholds (5 / 15 / 30%) and scored as a four-class classifier. For comparison, the Module 1 classifier reaches 79.9% accuracy and 0.733 macro F1.

MAE by soiling severity (percentage points):

| Model | clean | light | moderate | heavy |
|---|---|---|---|---|
| XGBoost, environment only | 11.9 | 9.7 | 9.8 | 18.3 |
| Random Forest, image + env | 3.5 | 2.9 | 5.7 | 4.2 |
| XGBoost, image + env | 3.2 | 3.1 | 6.1 | 4.4 |
| CNN regressor, image + env | 2.2 | 3.1 | 7.4 | 5.0 |

![Model comparison](module2_powerloss/outputs/model_comparison.png)
![Predicted vs measured](module2_powerloss/outputs/predicted_vs_actual.png)

**Reading the results.**
- The photograph carries the signal. The environment-only model's error (13.5) is about 3.5 times higher than any image model's.
- The three image models are close: about 3.8 to 3.9 points MAE and R² of about 0.92 to 0.93. Random Forest has the lowest MAE, XGBoost the lowest RMSE (fewer large misses) and the CNN the lowest error on clean panels.
- Predicting the number and then binning it classifies soiling **better than the Module 1 classifier**: the CNN regressor reaches 80.8% accuracy and 0.743 macro F1. The regression target keeps the ordering between classes that a plain classifier throws away.
- *Moderate* soiling (15–30%) remains the hardest range for every model.

### Run it

```bash
python -m module2_powerloss.train_cnn      # CNN regressor (starts from the Module 1 weights)
python -m module2_powerloss.features       # cache Module 1 embeddings for the tree models
python -m module2_powerloss.train_trees    # baselines, Random Forest, XGBoost
python -m module2_powerloss.evaluate       # compare all models, write figures
python -m module2_powerloss.export_site    # publish to the website
python -m module2_powerloss.predict panel.jpg --irradiance 0.45 --time 12:30
```

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
module2_powerloss/   power-loss regression: CNN, Random Forest, XGBoost, baselines
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
