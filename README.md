# 🫀 ECG Arrhythmia Classification via Deep Learning

> **DSA4050 — Deep Learning for Computer Vision** | Group 7
>
> George Rading (667965) · Stephen W. Austine (667917)

A full end-to-end pipeline that frames ECG analysis as a **computer vision problem**: raw cardiac signals are segmented beat-by-beat, converted into STFT spectrogram images, and fed to a fine-tuned **EfficientNetB0** for 4-class arrhythmia classification.

| Metric | Score |
|---|---|
| Test Accuracy | **97.47%** |
| Macro F1 | **0.9548** |
| Val Loss (best) | **0.0933** |
| Dataset | MIT-BIH Arrhythmia Database |

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Approach Overview](#approach-overview)
- [Classification Target](#classification-target)
- [Dataset](#dataset)
- [Pipeline Stages](#pipeline-stages)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Problem Statement

Cardiovascular diseases are among the leading causes of death globally. Electrocardiograms (ECGs) are the frontline diagnostic tool — but manual interpretation by cardiologists is time-consuming and expertise-dependent. This project builds an automated classification system that detects four clinically significant cardiac conditions from raw ECG signals, without any domain-specific feature engineering. Instead of hand-crafting ECG features, we let a deep convolutional network learn directly from time-frequency image representations of individual heartbeats.

---

## Approach Overview

```
Raw ECG Signal (MIT-BIH)
        │
        ▼
Beat Segmentation — 360-sample windows centered on R-peaks
        │
        ▼
STFT Spectrogram Conversion — 1D signal → 2D time-frequency image
        │
        ▼
EfficientNetB0 — Two-phase transfer learning
   Phase 1: Frozen backbone → train head only (5 epochs)
   Phase 2: Full fine-tuning, lr=1e-4 (up to 15 epochs)
        │
        ▼
Evaluation — Accuracy, F1, Confusion Matrix, Grad-CAM
```

The key design insight is that cardiac morphology differences between arrhythmia types manifest as visually distinct patterns in time-frequency space. A pretrained CNN trained on natural images can transfer spatial feature detection to spectrogram images with minimal labeled data requirements.

---

## Classification Target

| Class | MIT-BIH Symbols | Clinical Meaning |
|---|---|---|
| `Normal` | N | Healthy sinus rhythm |
| `Arrhythmia` | V, E | Premature ventricular contractions, ventricular escape beats |
| `AFib` | A, S, e | Atrial fibrillation, supraventricular ectopic beats, atrial escape |
| `MI` | L, R | Myocardial infarction proxy — bundle branch block patterns |

All other annotation symbols (pacemaker beats, fusion beats, rhythm markers) are discarded to avoid label noise — their morphology does not consistently correspond to a single condition.

> ⚠️ **MI Label Note:** The MIT-BIH Arrhythmia Database contains no explicit myocardial infarction labels. Left and right bundle branch block beats (L, R) are used as an accepted research proxy for MI-associated conduction abnormalities. This is acknowledged as a limitation.

---

## Dataset

**MIT-BIH Arrhythmia Database** — the gold standard benchmark for ECG research.

| Property | Value |
|---|---|
| Records | 48 half-hour ECG recordings |
| Sampling frequency | 360 Hz |
| Signal leads | 2 (Lead I and II — we use Lead I only) |
| Signal length | ~650,000 samples per record (~30 min) |
| Annotations | Expert-labeled beat-by-beat using AAMI symbols |
| Total annotated beats | ~110,000 across all records |
| Download size | ~100 MB |
| Source | PhysioNet via `wfdb.dl_database('mitdb', ...)` |

### Class Distribution (after processing)

| Class | Images | Proportion |
|---|---|---|
| Normal | ~75,000 | ~74.9% |
| Arrhythmia | ~7,000 | ~7.0% |
| MI | ~8,000 | ~8.0% |
| AFib | ~2,600 | ~2.6% |
| **Total** | **~100,000** | — |

The severe class imbalance (Normal : AFib ≈ 29:1) is clinically realistic but must be addressed explicitly in training — see [Design Decisions](#design-decisions).

---

## Pipeline Stages

### Stage 1 — Environment Setup
Creates the full local directory structure:
```
ecg_project/
    mitdb/              ← raw MIT-BIH .dat/.hea/.atr files
    ecg_spectrograms/
        Normal/         ← spectrogram PNGs, one per beat
        Arrhythmia/
        AFib/
        MI/
    checkpoints/        ← model weights
    outputs/            ← figures, reports
```

### Stage 2 — Data Acquisition
Downloads all 48 MIT-BIH records (~100 MB) via the `wfdb` library. Idempotent — skips records already on disk. Produces 144 files (48 × `.dat` + `.hea` + `.atr`).

### Stage 3 — Beat Segmentation
Each annotated R-peak in every record is extracted as a **360-sample window** (180 samples before + 180 samples after the R-peak), corresponding to exactly 1 second at 360 Hz. Beats too close to the recording boundary are discarded. Only Lead I is used.

```python
WINDOW_BEFORE = 180  # samples before R-peak
WINDOW_AFTER  = 180  # samples after R-peak
```

### Stage 4 — Signal-to-Image Conversion
Each 1D beat segment is converted to a 2D spectrogram via **Short-Time Fourier Transform (STFT)**:

```python
f, t, Zxx = scipy.signal.stft(beat, fs=360, nperseg=64, noverlap=32)
```

Post-processing:
1. Convert to power in dB scale: `20 × log10(|Zxx| + ε)`
2. Normalize to [0, 1] per image
3. Save as a 33×13 PNG (upsampled to 224×224 during training)

The 50% overlap (`noverlap=32`) provides smooth time-frequency coverage. The dB scale emphasizes low-amplitude frequency components that would otherwise be swamped by the QRS peak energy.

### Stage 5 — Batch Processing
All 48 records are processed in one pass. File naming: `{record_id}_{beat_index:04d}.png`. Already-generated files are skipped, making the process safely interruptible and resumable. A corruption scan follows to remove any partial writes from interrupted runs.

### Stage 6 — Dataset Preparation
- **Normal class capped at 20,000 samples** to make CPU training feasible while preserving all minority class samples
- **Stratified 70/15/15 train/val/test split** (preserves class proportions in every subset)
- **Training augmentations:** RandomHorizontalFlip, RandomRotation(±10°), ColorJitter, RandomAffine
- **Val/Test:** Resize + Normalize only (deterministic)
- All images normalized to ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- `num_workers=0` for macOS multiprocessing compatibility

| Split | Images | Batches (bs=32) |
|---|---|---|
| Train | ~31,587 | ~988 |
| Val | ~6,769 | ~212 |
| Test | ~6,769 | ~212 |

### Stage 7 — Class Weighting
Inverse-frequency weights are computed from training labels only and passed to `CrossEntropyLoss`:

```
weight_i = total_train / (num_classes × count_i)
```

AFib receives the highest weight (~9.76) because it has the fewest samples. Every AFib misclassification contributes ~10× more to the loss than an MI misclassification — appropriate for a clinical tool where minority class detection is critical.

---

## Model Architecture

**EfficientNetB0** pretrained on ImageNet, with a custom 4-class classification head:

```
EfficientNetB0 Backbone (feature extractor)
    └── [ImageNet pretrained weights]
    └── Outputs: (batch, 1280) feature vector

Custom Head:
    Dropout(p=0.4)
    Linear(1280 → 4)
```

| Parameter | Value |
|---|---|
| Base model | EfficientNetB0 |
| Total parameters | ~4.0M |
| Phase 1 trainable | ~5,124 (head only) |
| Phase 2 trainable | ~4,012,672 (full network) |
| Input size | 224 × 224 × 3 |
| Output classes | 4 |
| Dropout | 0.4 (higher than default 0.2 — regularizes domain shift) |

---

## Training Strategy

### Phase 1 — Frozen Backbone (5 epochs)
The backbone is fully frozen. Only the `Linear(1280→4)` head is trained at `lr=1e-3`. This protects pretrained ImageNet features from high-gradient updates from a randomly initialized head.

- Optimizer: Adam
- Scheduler: ReduceLROnPlateau (factor=0.3, patience=3)
- Early stopping: patience=3 on val loss
- Checkpoint: saved to `ecg_project/checkpoints/ecg_best_model.pth` on val loss improvement

### Phase 2 — Full Fine-Tuning (up to 15 epochs)
Best Phase 1 checkpoint is reloaded. All layers are unfrozen and trained end-to-end at `lr=1e-4` — 10× smaller to protect pretrained features during adaptation.

- Optimizer: Adam
- Scheduler: ReduceLROnPlateau (factor=0.3, patience=3)
- Early stopping: patience=3 on val loss
- Same checkpoint path (overwritten only on improvement)

### Phase 2 Training Log

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Status |
|-------|-----------|-----------|----------|---------|--------|
| 01/15 | 0.6259 | 75.39% | 0.2618 | 90.83% | ✅ Saved |
| 02/15 | 0.3033 | 89.28% | 0.1823 | 93.69% | ✅ Saved |
| 03/15 | 0.2270 | 92.22% | 0.1730 | 95.54% | ✅ Saved |
| 04/15 | 0.1814 | 93.49% | 0.1425 | 95.51% | ✅ Saved |
| 05/15 | 0.1501 | 94.80% | 0.1146 | 96.01% | ✅ Saved |
| 06/15 | 0.1312 | 95.48% | 0.1090 | 96.88% | ✅ Saved |
| 07/15 | 0.1193 | 95.75% | 0.1031 | 96.71% | ✅ Saved |
| 08/15 | 0.0960 | 96.50% | 0.1110 | 96.07% | ⏳ Patience 1/3 |
| 09/15 | 0.0940 | 96.55% | 0.1143 | 97.47% | ⏳ Patience 2/3 |
| 10/15 | 0.0801 | 97.10% | 0.0985 | 97.61% | ✅ Saved |
| 11/15 | 0.0728 | 97.35% | **0.0933** | **97.75%** | ✅ Saved |

---

## Results

### Test Set Performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.9848 | 0.9707 | 0.9777 | ~3,000 |
| Arrhythmia | 0.9812 | 0.9621 | 0.9716 | ~1,050 |
| AFib | 0.8299 | 0.9377 | 0.8805 | ~385 |
| MI | 0.9934 | 0.9852 | 0.9893 | ~1,200 |
| **Macro avg** | **0.9473** | **0.9639** | **0.9548** | — |
| **Weighted avg** | **0.9760** | **0.9747** | **0.9751** | — |

**Test Accuracy: 97.47%**

### Key Observations

- **MI (F1: 0.9893)** — strongest class, remarkable given the absence of explicit MI labels in MIT-BIH. Bundle branch block morphology is highly distinctive in spectrogram space, and the model learned it with near-perfect precision and recall.
- **Arrhythmia (F1: 0.9716)** — premature ventricular contractions have a wide, bizarre QRS complex that is visually distinctive in STFT images, explaining the high bilateral performance.
- **Normal (F1: 0.9777)** — precision (0.9848) exceeds recall (0.9707), meaning the model is slightly more likely to flag a Normal beat as something else than to call something abnormal as Normal. This is the safer failure direction clinically.
- **AFib (F1: 0.8805)** — the weakest class, with precision (0.8299) lower than recall (0.9377). Supraventricular beats overlap visually with Normal beats in spectrogram space, and AFib has the smallest support. Crucially, **recall is prioritised**: missing an AFib case is more dangerous than a false positive that prompts further review.
- **Macro avg vs Weighted avg gap (0.9548 vs 0.9751)** — the weighted average is inflated by large Normal and MI classes. Macro F1 is the honest metric for imbalanced datasets and is the number to lead with.

### Explainability — Grad-CAM

Grad-CAM attention maps are generated for a correctly classified sample of each class by hooking into `model.features[-1]` (the deepest convolutional block). Activations concentrated on the mid-frequency, beat-center region confirm the model is attending to QRS complex morphology — the clinically relevant signal — rather than image artifacts.

---

## Project Structure

```
ECG_Local_VS_Code.ipynb          ← main notebook (all stages)

ecg_project/
├── mitdb/                       ← raw MIT-BIH files (.dat, .hea, .atr)
│   └── 100.dat, 100.hea, ...    ← 144 files total (48 records × 3)
│
├── ecg_spectrograms/            ← generated spectrogram PNGs
│   ├── Normal/
│   ├── Arrhythmia/
│   ├── AFib/
│   └── MI/
│
├── checkpoints/
│   └── ecg_best_model.pth       ← best model weights (Phase 1 + 2)
│
└── outputs/
    ├── record100_raw.png         ← raw ECG waveform sample
    ├── beat_window.png           ← single extracted beat window
    ├── spectrogram_example.png   ← 1D beat → 2D spectrogram illustration
    ├── class_distribution.png    ← bar chart + pie chart of class counts
    ├── learning_curves.png       ← train/val loss and accuracy across both phases
    ├── classification_report.txt ← per-class F1, precision, recall (test set)
    ├── confusion_matrix.png      ← raw counts + row-normalized confusion matrices
    ├── confidence_distribution.png ← per-class predicted probability histograms
    └── gradcam.png               ← Grad-CAM attention overlays for all 4 classes
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- macOS, Linux, or Windows

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ecg-classification.git
cd ecg-classification
```

### 2. Install dependencies

```bash
pip install wfdb torch torchvision scipy matplotlib scikit-learn tqdm opencv-python-headless
```

Or with a requirements file:

```bash
pip install -r requirements.txt
```

### 3. Verify your environment

```python
import torch
print(torch.__version__)
print("GPU available:", torch.cuda.is_available())
# Note: the notebook is configured for CPU (DEVICE = torch.device('cpu'))
# GPU support can be added by changing the DEVICE variable
```

> **macOS note:** `num_workers=0` is set in all DataLoaders due to Python multiprocessing constraints on macOS. If running on Linux with a GPU, increasing `num_workers` (e.g. to 4) will significantly speed up data loading.

---

## Running the Pipeline

Open `ECG_Local_VS_Code.ipynb` in Jupyter or VS Code and run cells sequentially from top to bottom. Each stage is clearly labeled. Key things to know:

| Stage | Cell(s) | Est. Time (CPU) | Notes |
|---|---|---|---|
| Environment setup | 1–2 | Instant | Run once |
| MIT-BIH download | 3 | 2–5 min | Run once, ~100 MB |
| Spectrogram generation | 8 | 15–25 min | Run once; idempotent |
| Corruption scan | 9 | 1–2 min | Safe to re-run |
| Phase 1 training | 15 | ~1 hour | 5 epochs |
| Phase 2 training | 16 | 2–3 hours | Up to 15 epochs |
| Evaluation | 18–22 | ~10 min | — |

### Resuming after interruption

Training checkpoints are saved to disk after every epoch that improves validation loss. To resume after a shutdown:

1. Wait for the current epoch to finish and the `✅ Checkpoint saved` line to appear
2. Close the notebook / shut down
3. Reopen and re-run from the checkpoint reload cell — the model loads from `ecg_project/checkpoints/ecg_best_model.pth` automatically

Spectrogram generation is also safe to interrupt — already-generated files are skipped on the next run.

---

## Design Decisions

### Why convert ECG to spectrograms instead of using raw 1D signals?

1D ECG signals can be classified directly with 1D CNNs or RNNs. However, STFT spectrograms unlock the full pretrained ImageNet backbone: EfficientNetB0's spatial convolutional filters, trained on millions of natural images, transfer to time-frequency patterns in cardiac signals. This avoids training a temporal model from scratch and dramatically reduces the data requirement.

### Why EfficientNetB0?

EfficientNetB0 balances accuracy and parameter efficiency. At ~4M parameters it is compact enough to train on CPU without excessive memory pressure, while still providing strong ImageNet feature representations. Larger models (ResNet-50, EfficientNetB4) would offer higher capacity but are impractical for local CPU training on this dataset.

### Why two-phase training instead of end-to-end from the start?

A randomly initialized head produces high-gradient updates in early training. Applying these to the backbone immediately risks catastrophic forgetting — corrupting ImageNet features before the head has learned anything meaningful. Phase 1 (frozen backbone) stabilizes the head weights first, giving Phase 2 (full fine-tuning) a clean starting point with lower learning rate.

### Why cap Normal at 20,000?

The raw dataset has ~75,000 Normal beats — 74.9% of all samples. Without capping, each epoch takes 15–20 minutes on CPU and the model has diminishing returns from seeing redundant Normal examples. Capping at 20,000 reduces total dataset size by 55% while preserving 100% of all minority class samples, making training feasible in 8–15 minutes per epoch without sacrificing classification performance on rare classes.

### Why class-weighted loss instead of oversampling?

Oversampling minority classes (e.g. SMOTE) generates synthetic spectrogram images that may not reflect real cardiac morphology. Inverse-frequency loss weighting is a simpler, safer approach: it directly scales gradient magnitude per class without creating artificial training examples.

### Why ReduceLROnPlateau instead of a fixed schedule?

The plateau at Phase 2 epochs 8–9 illustrates exactly why: the model temporarily stopped improving but recovered strongly at epoch 10 after the scheduler reduced the learning rate. A fixed cosine schedule would have applied the same LR regardless of the model's current state. `ReduceLROnPlateau` adapts to the actual training dynamics.

### Why checkpoint on validation loss rather than accuracy?

Validation accuracy is discrete and noisy — small fluctuations in the number of correctly classified samples can move it without a real improvement in model quality. Validation loss is a continuous signal that is more sensitive to genuine improvement and more reliable as a stopping criterion.

---

## Limitations

1. **No explicit MI labels.** MIT-BIH contains no myocardial infarction records. Bundle branch block beats (L, R) are used as a proxy. While this is an accepted research approach, it means the model detects BBB patterns, not MI directly. A dedicated MI dataset (e.g. PTB-XL) would be needed for a true MI classifier.

2. **Single lead.** Only Lead I is used. Clinical ECG interpretation uses 12 leads simultaneously. Multi-lead input would provide substantially richer information, particularly for MI detection which is strongly lead-dependent.

3. **CPU-only training.** The notebook is configured for CPU. Phase 2 training takes 2–3 hours on a modern laptop. GPU support requires only changing `DEVICE = torch.device('cuda')` and potentially increasing `num_workers`.

4. **Normal class capping.** Capping Normal at 20,000 reduces training data volume significantly. The model may be slightly less calibrated for edge-case Normal beats that appear rarely.

5. **Single dataset generalization.** The model is trained and evaluated entirely on MIT-BIH. ECG morphology varies by patient, recording equipment, and electrode placement. External validation on a different dataset (e.g. PhysioNet 2017 Challenge data) would be needed to assess clinical generalizability.

6. **STFT resolution trade-off.** The 33×13 STFT output (determined by `nperseg=64, noverlap=32`) is upsampled to 224×224 for EfficientNetB0. This introduces interpolation artifacts. A longer beat window or higher STFT resolution would produce a denser time-frequency representation.

---

## Future Improvements

### Signal Processing

- [ ] **Multi-lead input** — concatenate Lead I and Lead II as separate image channels, giving the model 2×more cardiac information per beat
- [ ] **Longer beat windows** — extend from 360 to 540 samples (1.5s) to capture P-wave and T-wave tails more fully, particularly relevant for AFib detection
- [ ] **Mel spectrogram** — replace linear STFT with mel-scale frequency bins, which better align with the perceptually relevant frequency bands in ECG signals
- [ ] **Wavelet transform** — replace STFT with continuous wavelet transform (CWT) for better time-frequency localization, especially for short-duration transients like PVCs

### Model & Training

- [ ] **GPU support** — change `DEVICE = torch.device('cuda')` and set `num_workers=4+` for a 10–20× speedup in training
- [ ] **Gradual unfreezing** — in Phase 2, unfreeze EfficientNetB0 layer groups progressively from top to bottom rather than all at once, for gentler domain adaptation
- [ ] **Larger backbone** — try EfficientNetB2 or B4 once GPU is available; larger models may push AFib F1 above 0.92
- [ ] **Cosine annealing with warmup** — replace ReduceLROnPlateau with a warmed cosine schedule to reduce the initial Phase 2 loss spike (0.6259 at epoch 1)
- [ ] **Mixed precision training** (`torch.cuda.amp`) — halves GPU memory usage with no accuracy cost

### Data & Augmentation

- [ ] **External MI dataset** — incorporate PTB-XL or PTB Diagnostic ECG Database for genuine MI labels rather than BBB proxies
- [ ] **SpecAugment** — apply frequency and time masking directly to spectrograms during training for stronger regularization
- [ ] **Mixup / CutMix** — blend spectrogram pairs during training to smooth decision boundaries, particularly between AFib and Normal
- [ ] **Cross-dataset validation** — evaluate on PhysioNet 2017 Challenge or CPSC 2018 to test generalization beyond MIT-BIH

### Evaluation & Explainability

- [ ] **AUROC per class** — add area-under-ROC curves to surface performance at different operating thresholds, critical for clinical decision support where sensitivity/specificity trade-offs matter
- [ ] **Calibration curves** — verify that the model's confidence scores match actual accuracy (a model that says 90% confident should be correct ~90% of the time)
- [ ] **Layer-wise Grad-CAM** — compare attention maps from early vs. late backbone layers to understand which frequency-time regions matter at different levels of abstraction
- [ ] **Error analysis** — manually inspect and clinically annotate the most confidently wrong predictions to identify systematic failure modes

### Deployment

- [ ] **ONNX export** — make the model portable for inference outside PyTorch
- [ ] **Real-time beat classifier** — wrap the segmentation + STFT + model pipeline into a streaming inference function that classifies beats as they arrive from a live ECG feed
- [ ] **FastAPI endpoint** — serve the model as a REST API accepting raw ECG signal arrays and returning class probabilities
- [ ] **Experiment tracking** — integrate Weights & Biases (`wandb`) or MLflow to log all runs, hyperparameters, and checkpoints systematically

---

## References

- Moody, G.B. and Mark, R.G. (2001). [The impact of the MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). *IEEE Engineering in Medicine and Biology*, 20(3): 45-50.
- Tan, M. and Le, Q.V. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). *ICML 2019*.
- Selvaraju, R.R. et al. (2017). [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391). *ICCV 2017*.
- PhysioNet: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
