# Vehicle Motion Conformer: Benchmark History & Experiment Observations

This document records the complete chronological benchmark history, evaluation metrics breakdown, accuracy thresholds, and architectural observations across all training experiments conducted on the Vehicle Motion Conformer project.

---

## 1. Master Benchmark Comparison (1,000-Clip Blind Test)

All models evaluated on the held-out independent blind test dataset (`/Users/rohith/Downloads/vs13_path2d_1000_varied`, 1,000 unseen trajectories).

| # | Run Name | Dataset | Epochs | Key Changes / Features | Val Loss | ADE (m) ↓ | Median ADE (m) ↓ | Online ADE (m) ↓ | FDE (m) ↓ | X MAE (m) ↓ | Y MAE (m) ↓ |
|---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Original Baseline** | 1,000 | 30 | Linear FFT projection, Beam search | 2.8428 | 5.047 | 4.810 | 5.102 | 11.009 | — | — |
| 2 | **`vs13_1000_option_a_cnn`** | 1,000 | 30 | + Frequency CNN frontend | 2.1404 | 4.158 | 3.910 | 4.194 | 10.070 | — | — |
| 3 | **`smoke_1000`** | 1,000 | 50 | + Soft labels, kinematic loss, SpecAugment | 2.2886† | 4.365 | 3.775 | 4.567 | 9.947 | 2.679 | 2.840 |
| 4 | **`vs13_7500_varied_v1`** | 7,500 | 50 | Baseline 7.5k scaled dataset (Linear frontend) | 1.9416 | 3.043 | 2.710 | 3.093 | 7.748 | — | — |
| 5 | **`vs13_7500_varied_cnn`** | 7,500 | 44 | + Frequency CNN frontend | **1.8173** | 2.965 | 2.612 | 3.013 | 7.333 | 1.385 | 2.301 |
| 6 | **`vs13_7500_varied_improved`** | 7,500 | 26 | + Soft labels (σ=1.0), Kinematic loss, SpecAugment | 2.1622† | 2.806 | 2.422 | 2.821 | 7.623 | 1.525 | 1.988 |
| 7 | **`vs13_7500_varied_poshead` (Beam)** | 7,500 | 50 | + Auxiliary PosHead, Endpoint loss ramp ($1\times \rightarrow 3\times$) | 2.1473† | 2.660 | 2.306 | 2.707 | 7.087 | 1.319 | 1.979 |
| 8 | **`vs13_7500_varied_poshead` (Expectation RTS)** 🏆 | 7,500 | 50 | Same checkpoint + Expectation RTS smoother ($q=6, r=1.5$) | 2.1473† | **2.549** | **2.251** | **2.707** | **6.139** | **1.312** | **1.875** |
| 9 | **`vs13_7500_varied_physics`** | 7,500 | 59 | + Energy channel, Polar head ($r, v_r$), Path integration | 2.1881† | 2.621 | 2.351 | 2.767 | 6.545 | 1.397 | 1.902 |

> † Note: Validation loss values marked with (†) are not directly comparable to earlier CE-only runs because Gaussian soft targets, kinematic penalties, and auxiliary regression loss terms alter the baseline loss entropy and magnitude.

---

## 2. Accuracy Thresholds Progression

Percentage of test clips achieving Average Displacement Error (ADE) below given distance envelopes:

| Run Name | ADE < 1 m | ADE < 2 m | ADE < 5 m | ADE < 10 m | ADE < 20 m |
|:---|:---:|:---:|:---:|:---:|:---:|
| `vs13_7500_varied_cnn` (Old Baseline) | 2.4% | 30.8% | 89.9% | 99.2% | 99.8% |
| `vs13_7500_varied_improved` | 2.5% | 34.2% | 91.5% | 99.5% | 99.9% |
| `vs13_7500_varied_poshead` (Beam) | 4.4% | 40.2% | 93.2% | 99.7% | 100.0% |
| **`vs13_7500_varied_poshead` (Expectation RTS)** 🏆 | **4.1%** | **41.2%** | **94.8%** | **100.0%** | **100.0%** |
| `vs13_7500_varied_physics` | 3.8% | 37.0% | 93.9% | 99.9% | 100.0% |

---

## 3. Distribution Tail & Error Statistics

Percentile and extreme error metrics on the 1,000 blind test clips:

| Run Name | ADE Mean | ADE Median | ADE p90 | ADE p95 | ADE Max | FDE Mean | FDE Median | FDE p90 | FDE p95 | FDE Max |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `vs13_7500_varied_poshead` (Beam) | 2.660 m | 2.306 m | 4.526 m | 5.478 m | 13.074 m | 7.087 m | 6.061 m | 13.892 m | 16.854 m | 26.446 m |
| **`poshead` (Expectation RTS)** 🏆 | **2.549 m** | **2.251 m** | **4.258 m** | **5.034 m** | **8.727 m** | **6.139 m** | **5.111 m** | 12.521 m | **14.815 m** | **25.702 m** |
| `vs13_7500_varied_physics` | 2.621 m | 2.351 m | 4.283 m | 5.273 m | 10.013 m | 6.545 m | 5.466 m | **12.365 m** | 15.561 m | 34.206 m |

---

## 4. Milestone-by-Milestone Analysis & Key Observations

### Milestone 1: Frequency CNN Frontend (`vs13_7500_varied_cnn`)
* **Hypothesis:** Convolving across frequency bins captures shift-invariant acoustic harmonic structures (engine firing order, tire tread peaks) better than naive linear projection.
* **Observation:** ADE dropped from 3.043 m to 2.965 m; FDE improved from 7.748 m to 7.333 m. Proved that spectral convolutions provide an effective inductive bias for Doppler features.

### Milestone 2: Soft Ordinal Targets & Kinematic Loss (`vs13_7500_varied_improved`)
* **Hypothesis:** Discrete one-hot classification penalizes a prediction in bin 41 just as harshly as bin 100 when ground truth is bin 42. Gaussian soft labels reward proximity.
* **Observation:** Significant drop in longitudinal coordinate error: $Y$ MAE fell from 2.301 m down to 1.988 m (-13.6%). Overall ADE dropped to 2.806 m. However, FDE remained stubborn at 7.623 m due to late-clip acoustic signal fading.

### Milestone 3: Auxiliary Position Head & Endpoint Loss Ramp (`vs13_7500_varied_poshead`)
* **Hypothesis:** Time-weighted endpoint loss ($1\times \rightarrow 3\times$) forces the Conformer to care about trajectory termination, while a continuous position regression head anchors feature representations.
* **Observation:** Achieved the first sub-2.7 m ADE (2.660 m with beam search) and reversed FDE regression (7.623 m $\rightarrow$ 7.087 m). The fraction of clips under 2 m ADE jumped by nearly 10 percentage points (30.8% $\rightarrow$ 40.2%).

### Milestone 4: Expectation RTS Decoder (Zero Retraining Win)
* **Hypothesis:** Kinematic beam search suffered from discretization artifacts (snapping to 2-meter bins and over-penalizing sharp turns). Applying an RTS Kalman smoother directly to continuous head expectations ($\mu = \sum b_i P(b_i)$) should produce smooth paths without staircasing.
* **Observation:** All-time project best on both ADE (**2.549 m**) and FDE (**6.139 m**). 
  - FDE dropped by **-13.4%** without touching neural network weights.
  - Eliminated the worst-case outlier ceiling: maximum clip ADE fell from 13.07 m to 8.73 m.
  - 100% of test clips finished within the 10-meter ADE envelope.

### Milestone 5: Polar Heads, Energy Conditioning & Path Integration (`vs13_7500_varied_physics`)
* **Hypothesis:** Adding explicit acoustic energy channels, auxiliary polar heads (range $r$ and Doppler radial velocity $v_r$), and a cumulative velocity integration loss would eliminate lateral heading drift.
* **Observation:** Achieved solid results (ADE 2.621 m, FDE 6.545 m) and produced the best 90th percentile FDE (12.365 m). However, it slightly underperformed `poshead` (2.549 m) because simultaneously optimizing four competing auxiliary objectives (soft CE, polar Smooth-L1, geometric consistency, and path integration) diluted gradients from the primary categorical heads.

---

## 5. Physical Insights & Open Challenges

1. **Acoustic Asymmetry ($X$ vs $Y$):**
   * Doppler frequency shift is proportional to radial velocity $v_r = \vec{v} \cdot \frac{\vec{p}}{r}$. 
   * When vehicles travel primarily along $X$, $v_x$ dominates the Doppler signal. As a result, $X$-axis tracking is highly accurate across all models (MAE $\approx 1.3\text{ m}$).
   * Lateral motion ($Y$-axis) contributes weakly to Doppler frequency when distant ($|x| \gg |y|$), making $Y$-axis heading the dominant contributor to FDE (cross-track error at trajectory termination averages nearly $6\text{ m}$).

2. **The Amplitude / Loudness Dilemma:**
   * Range information ($r = \sqrt{x^2 + y^2}$) is largely carried by acoustic sound pressure level (decaying as $1/r$).
   * Standard deep learning audio backbones (Conformer, Transformers) rely heavily on LayerNorm, which normalizes away frame-level energy envelopes. Preserving global acoustic amplitude curves without destabilizing deep self-attention remains a critical design challenge.

3. **Current Champion:**
   * **Model Checkpoint:** `runs/vs13_7500_varied_poshead/model.pt`
   * **Decoder:** Expectation RTS (`config.py`: `DECODER_MODE = 'expectation_rts'`, `EXPECTATION_RTS_Q_ACCEL = 6.0`, `EXPECTATION_RTS_R_POS = 1.5`, `EXPECTATION_RTS_R_VEL = 4.0`)
   * **Score:** **ADE = 2.549 m, FDE = 6.139 m, Online ADE = 2.707 m**

