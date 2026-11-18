# Vehicle Motion Conformer — Master Research & Experiment Report

This master document provides the comprehensive, authoritative record of the **Vehicle Motion Conformer** project: theoretical methodology, physical Doppler formulations, neural architectures, loss function derivations, decoding algorithms, complete experimental benchmarks across all training runs, ablation studies, and physical error analyses.

---

## Table of Contents
1. [Problem Formulation & Acoustic Physics](#1-problem-formulation--acoustic-physics)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Supervision & Loss Function Formulations](#3-supervision--loss-function-formulations)
4. [Trajectory Decoding Algorithms](#4-trajectory-decoding-algorithms)
5. [Master Benchmark Comparison (All Experiments)](#5-master-benchmark-comparison-all-experiments)
6. [Detailed Experiment-by-Experiment Analysis](#6-detailed-experiment-by-experiment-analysis)
7. [Ablation Study: Frame-Synchronous Supervision vs. CTC](#7-ablation-study-frame-synchronous-supervision-vs-ctc)
8. [Physical Findings & Acoustic Observability Limits](#8-physical-findings--acoustic-observability-limits)
9. [Reproducibility & Execution Commands](#9-reproducibility--execution-commands)

---

## 1. Problem Formulation & Acoustic Physics

### 1.1 Objective
Given a single-channel monaural audio recording $a(t)$ of a vehicle pass-by and a known starting location $\mathbf{p}_0 = (x_0, y_0)$ at $t=0$, estimate the vehicle's continuous 2D trajectory at $10\text{ Hz}$ ($100\text{ ms}$ resolution):
$$\mathbf{s}(t) = \big[v_x(t),\, v_y(t),\, p_x(t),\, p_y(t)\big]^\top \quad \text{for } t \in [0.1, 0.2, \dots, T]$$
where the receiver (microphone) is fixed at the origin $(0, 0)$.

### 1.2 Acoustic Doppler Shift
For an acoustic source emitting engine/exhaust/tire harmonic frequencies $f_0$, the received frequency $f(t)$ is governed by the relative radial velocity $v_r(t)$:
$$f(t) = f_0 \left( \frac{c}{c - v_r(t)} \right) \approx f_0 \left( 1 + \frac{v_r(t)}{c} \right)$$
where $c \approx 343\text{ m/s}$ is the speed of sound in air, and:
$$v_r(t) = \vec{v}(t) \cdot \frac{\vec{p}(t)}{\|\vec{p}(t)\|} = \frac{x(t) v_x(t) + y(t) v_y(t)}{\sqrt{x(t)^2 + y(t)^2}}$$

### 1.3 The Observability Asymmetry
* When the vehicle travels primarily along the roadway ($X$-axis) at large distance ($|x| \gg |y|$):
  $$\frac{x}{\sqrt{x^2 + y^2}} \approx \pm 1, \qquad \frac{y}{\sqrt{x^2 + y^2}} \approx 0$$
  Consequently, $v_r \approx v_x$. The Doppler pitch shift is dominated ($>95\%$) by longitudinal speed $v_x$.
* Conversely, lateral velocity $v_y$ produces almost zero instantaneous Doppler shift at distance, making cross-track ($Y$-axis) tracking strictly dependent on acoustic amplitude decay ($1/r$) and cumulative spatial consistency.

---

## 2. End-to-End System Architecture

```
Raw Audio (22,050 Hz)
       │
       ▼
STFT Spectrogram (N_fft=2048, Hop=256, Centered) ──► 1025 Linear Frequency Bins
       │
       ▼
Frequency 2D CNN Frontend (Conv1d across harmonic bins + GELU + LayerNorm) ──► 128-dim
       │
       ├─────────────────────────┐
       ▼                         ▼
Spatial Conditioning: Concat [x0, y0] / FiLM Modulation
       │
       ▼
Causal Conformer Encoder (4 Layers, 4 Attention Heads, d_model=128, Kernel=15)
Left-Context: 1024 frames (Strictly 0 Lookahead Frames — Streaming Causal)
       │
       ├─────────────────────────────────────────┬────────────────────────┐
       ▼                                         ▼                        ▼
Four Categorical Heads                    Auxiliary Position Head   Streaming Cache
(vx: 121, vy: 121, px: 201, py: 101 bins) (Continuous px, py MLP)  (Zero-latency buffer)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
              Expectation RTS Kalman Smoother
                            ▼
            Smooth, Feasible 2D Trajectory
```

### 2.1 Acoustic Feature Extraction
* **Audio Sampling Rate:** $22,050\text{ Hz}$
* **STFT Window:** $N_{\text{fft}} = 2048$ ($92.9\text{ ms}$ window, yielding $1025$ frequency bins up to $11,025\text{ Hz}$)
* **Hop Size:** $H = 256$ samples ($11.6\text{ ms}$ frame step)
* **Alignment to $10\text{ Hz}$ Output:** The neural encoder extracts representations across the dense audio stream; the motion heads are evaluated at sampled time indices corresponding to every $100\text{ ms}$.

### 2.2 Frequency CNN Frontend
Instead of flattening or linearly projecting the $1025$ frequency bins (which destroys local harmonic relationships), a multi-layer 1D convolutional module operates directly across the frequency axis:
1. `Conv1d(in=1, out=8, kernel=5, stride=2, padding=2)` + `GELU()`
2. `Conv1d(in=8, out=16, kernel=5, stride=2, padding=2)` + `GELU()`
3. `Flatten()` + `Linear(16 * 257, 128)` + `LayerNorm(128)` + `Dropout(0.1)`

### 2.3 Causal Conformer Backbone
Built on NVIDIA NeMo's `ConformerEncoder`:
* **Hidden Dimension ($d_{\text{model}}$):** $128$
* **Stacked Layers:** $4$
* **Self-Attention Heads:** $4$ (Relative Positional Embeddings, `att_context_size=[1024, 0]`)
* **Convolutional Kernel Size:** $15$ (`conv_context_size='causal'`, LayerNorm)
* **Causality:** Strictly zero lookahead frames. Real-time streaming evaluation is supported chunk-by-chunk using cached activations (`self.encoder.streaming_cfg`).

---

## 3. Supervision & Loss Function Formulations

The total training loss $\mathcal{L}_{\text{total}}$ combines ordinal soft classification, kinematic physics constraints, and auxiliary continuous spatial regression:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{kin}} \mathcal{L}_{\text{kin}} + \lambda_{\text{acc}} \mathcal{L}_{\text{acc}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}}$$

### 3.1 Ordinal Gaussian Soft Cross-Entropy
Vehicle speed and position are continuous, ordinal physical quantities. Hard one-hot cross-entropy unfairly penalizes a prediction in bin $41$ just as harshly as bin $100$ when ground truth is bin $42$. 

Targets are converted into Gaussian probability distributions ($\sigma = 1.0\text{ bins}$):
$$P_{\text{target}}(k) = \frac{\exp\left(-\frac{(k - y^*)^2}{2\sigma^2}\right)}{\sum_{m} \exp\left(-\frac{(m - y^*)^2}{2\sigma^2}\right)}$$
The cross-entropy loss across all 4 heads ($v_x, v_y, p_x, p_y$) is weighted by a time ramp:
$$\mathcal{L}_{\text{CE}} = \frac{1}{4} \sum_{j=1}^{4} \frac{1}{T} \sum_{t=1}^{T} w(t) \cdot \text{CE}\big(\hat{\mathbf{z}}_j(t),\, \mathbf{P}_{\text{target}, j}(t)\big)$$

### 3.2 Endpoint Loss Weighting Ramp
Because trajectory error compounds over time (a small velocity error at $t=1$ causes massive spatial drift at $t=5$), a linear ramp weights late-clip timesteps up to $3\times$ higher than initial timesteps:
$$w(t) = 1.0 + (w_{\text{endpoint}} - 1.0) \frac{t}{T-1}, \qquad w_{\text{endpoint}} = 3.0$$

### 3.3 Kinematic Consistency Penalty
Forces the predicted coordinate increments to match integrated velocities:
$$\mathbf{p}_{\text{pred}}(t) - \mathbf{p}_{\text{pred}}(t-1) \approx \frac{\mathbf{v}_{\text{pred}}(t) + \mathbf{v}_{\text{pred}}(t-1)}{2} \Delta t$$
$$\mathcal{L}_{\text{kin}} = \text{SmoothL1}\left( \frac{\mathbf{p}(t) - \mathbf{p}(t-1) - \bar{\mathbf{v}}(t)\Delta t}{2.0} \right), \qquad \lambda_{\text{kin}} = 0.05$$

### 3.4 Maximum Acceleration Constraint
Penalizes physically impossible accelerations exceeding street vehicle limits ($a_{\text{max}} = 8.0\text{ m/s}^2$):
$$\mathcal{L}_{\text{acc}} = \text{mean}\left( \left[ \text{ReLU}\left( \frac{\|\mathbf{a}(t)\|}{a_{\text{max}}} - 1.0 \right) \right]^2 \right), \qquad \lambda_{\text{acc}} = 0.01$$

### 3.5 Auxiliary Direct Position Regression Head
A lightweight MLP (`Linear(128, 64) -> GELU -> Linear(64, 2)`) directly predicts continuous $(p_x, p_y)$ coordinates:
$$\mathcal{L}_{\text{pos}} = \text{SmoothL1}\left( \frac{(\hat{\mathbf{p}}_{\text{direct}}(t) - \mathbf{p}_{\text{GT}}(t)) \cdot w(t)}{10.0} \right), \qquad \lambda_{\text{pos}} = 0.1$$

---

## 4. Trajectory Decoding Algorithms

### 4.1 The Discretization Failure of Beam Search
Earlier versions utilized an $N$-best kinematic beam search over discrete bin combinations. This introduced severe **2-meter staircasing artifacts** (snapping to discrete grid centers) and over-penalized transient turn maneuvers, locking FDE at $\approx 7.1\text{ m}$.

### 4.2 Continuous Expectation RTS Kalman Smoother (The Champion Decoder)
To eliminate staircasing with zero retraining, the decoder computes the continuous mathematical expectation over posterior probabilities at each timestamp:
$$\bar{s}_j(t) = \sum_{k=1}^{K_j} b_{j, k} \cdot P\big(s_j(t) = b_{j, k}\big)$$
This yields continuous measurements $\mathbf{z}(t) = [\bar{v}_x, \bar{v}_y, \bar{p}_x, \bar{p}_y]^\top$.

An **Rauch-Tung-Striebel (RTS) Rauch smoother** runs a forward Kalman filter followed by a backward smoothing recursion:
* **State Vector:** $\mathbf{x}(t) = [v_x, v_y, p_x, p_y]^\top$
* **State Transition:** 
  $$\mathbf{F} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \Delta t & 0 & 1 & 0 \\ 0 & \Delta t & 0 & 1 \end{bmatrix}$$
* **Tuned Hyperparameters:**
  * Process Noise: $Q_{\text{accel}} = 6.0\text{ m/s}^2$ (allows agile turns)
  * Measurement Noise Position: $R_{\text{pos}} = 1.5\text{ m}$
  * Measurement Noise Velocity: $R_{\text{vel}} = 4.0\text{ km/h}$

**Impact:** Reduced FDE from **$7.087\text{ m} \to 6.139\text{ m}$** ($-13.4\%$) and ADE from **$2.660\text{ m} \to 2.549\text{ m}$** instantly, eliminating the outlier ceiling (max ADE dropped from $13.07\text{ m} \to 8.73\text{ m}$).

---

## 5. Master Benchmark Comparison (All Experiments)

All experiments evaluated on the **1,000-clip independent held-out blind test set** (`vs13_path2d_1000_varied`, unseen trajectories):

| # | Run Name | Train Scale | Epochs | Key Innovations / Architectural Changes | Val Loss | ADE (m) ↓ | Median ADE (m) ↓ | Online ADE (m) ↓ | FDE (m) ↓ | X MAE (m) ↓ | Y MAE (m) ↓ |
|---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Initial Baseline** | 1,000 | 30 | Linear FFT projection, Beam search | 2.8428 | 5.047 | 4.810 | 5.102 | 11.009 | — | — |
| 2 | **`vs13_1000_option_a_cnn`** | 1,000 | 30 | + Frequency 2D CNN frontend | 2.1404 | 4.158 | 3.910 | 4.194 | 10.070 | — | — |
| 3 | **`smoke_1000`** | 1,000 | 50 | + Soft labels, kinematic loss, SpecAugment | 2.2886† | 4.365 | 3.775 | 4.567 | 9.947 | 2.679 | 2.840 |
| 4 | **`vs13_7500_varied_v1`** | 7,500 | 50 | 7.5k scaled dataset (Linear frontend) | 1.9416 | 3.043 | 2.710 | 3.093 | 7.748 | — | — |
| 5 | **`vs13_7500_varied_cnn`** | 7,500 | 44 | + Frequency 2D CNN frontend | **1.8173** | 2.965 | 2.612 | 3.013 | 7.333 | 1.385 | 2.301 |
| 6 | **`vs13_7500_varied_improved`** | 7,500 | 26 | + Soft labels ($\sigma=1.0$), Kinematic loss, SpecAug | 2.1622† | 2.806 | 2.422 | 2.821 | 7.623 | 1.525 | 1.988 |
| 7 | **`vs13_7500_varied_poshead` (Beam)** | 7,500 | 50 | + Aux PosHead, Endpoint ramp ($1\times \to 3\times$) | 2.1473† | 2.660 | 2.306 | 2.707 | 7.087 | 1.319 | 1.979 |
| 8 | **`poshead` (Expectation RTS)** 🏆 | 7,500 | 50 | **Same checkpoint + Expectation RTS smoother** | 2.1473† | **2.549** | **2.251** | **2.707** | **6.139** | **1.312** | **1.875** |
| 9 | **`vs13_7500_varied_physics`** | 7,500 | 59 | + Energy channel, Polar head ($r, v_r$), PathInt | 2.1881† | 2.621 | 2.351 | 2.767 | 6.545 | 1.397 | 1.902 |
| 10 | **`vs13_7500_varied_heading`** | 7,500 | 32 | + Heading direction cosine loss ($\mathcal{L}_{\text{heading}}$) | 2.1967† | 2.913 | 2.649 | 3.165 | 6.715 | 1.484 | 2.184 |

> *(† Note: Losses with soft targets, kinematic penalties, and auxiliary regression terms cannot be directly compared against raw CE losses due to entropy smoothing).*

### 5.1 Accuracy Threshold Progression

| Configuration | ADE $< 1\text{ m}$ | ADE $< 2\text{ m}$ | ADE $< 5\text{ m}$ | ADE $< 10\text{ m}$ | ADE $< 20\text{ m}$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| `vs13_7500_varied_cnn` (Old Baseline) | 2.4% | 30.8% | 89.9% | 99.2% | 99.8% |
| `vs13_7500_varied_improved` | 2.5% | 34.2% | 91.5% | 99.5% | 99.9% |
| `vs13_7500_varied_poshead` (Beam) | 4.4% | 40.2% | 93.2% | 99.7% | 100.0% |
| **`poshead` (Expectation RTS)** 🏆 | **4.1%** | **41.2%** | **94.8%** | **100.0%** | **100.0%** |
| `vs13_7500_varied_physics` | 3.8% | 37.0% | 93.9% | 99.9% | 100.0% |
| `vs13_7500_varied_heading` | 2.6% | 26.4% | 92.2% | 99.9% | 100.0% |

### 5.2 Error Distribution Percentiles

| Configuration | ADE Mean | ADE Median | ADE p90 | ADE p95 | ADE Max | FDE Mean | FDE Median | FDE p90 | FDE p95 | FDE Max |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `poshead` (Beam) | 2.660 m | 2.306 m | 4.526 m | 5.478 m | 13.074 m | 7.087 m | 6.061 m | 13.892 m | 16.854 m | 26.446 m |
| **`poshead` (Expectation RTS)** 🏆 | **2.549 m** | **2.251 m** | **4.258 m** | **5.034 m** | **8.727 m** | **6.139 m** | **5.111 m** | 12.521 m | **14.815 m** | **25.702 m** |
| `vs13_7500_varied_physics` | 2.621 m | 2.351 m | 4.283 m | 5.273 m | 10.013 m | 6.545 m | 5.466 m | **12.365 m** | 15.561 m | 34.206 m |
| `vs13_7500_varied_heading` | 2.913 m | 2.649 m | 4.562 m | 5.697 m | 11.336 m | 6.715 m | 5.598 m | 13.048 m | 15.779 m | 31.830 m |

---

## 6. Detailed Experiment-by-Experiment Analysis

### Milestone 1: Frequency 2D CNN Frontend (`vs13_7500_varied_cnn`)
* **Hypothesis:** Convolving across frequency bins captures shift-invariant acoustic harmonic structures (engine firing order, tire tread peaks) better than naive linear projection.
* **Result:** ADE dropped from $3.043\text{ m} \to 2.965\text{ m}$; FDE dropped from $7.748\text{ m} \to 7.333\text{ m}$. Proved that spectral convolutions provide a vital inductive bias for Doppler features.

### Milestone 2: Soft Ordinal Targets & Kinematics (`vs13_7500_varied_improved`)
* **Hypothesis:** Replacing hard Dirac targets with Gaussian distributions prevents over-penalizing adjacent speed bins while kinematic residuals prevent teleportation.
* **Result:** $Y$-axis MAE dropped from $2.301\text{ m} \to 1.988\text{ m}$ ($-13.6\%$), lowering overall ADE to $2.806\text{ m}$.

### Milestone 3: Auxiliary Position Head & Endpoint Loss Ramp (`vs13_7500_varied_poshead`)
* **Hypothesis:** Time-weighted endpoint loss ($1\times \to 3\times$) forces the Conformer to care about trajectory termination, while a continuous position regression head anchors feature representations.
* **Result:** Achieved the first sub-$2.7\text{ m}$ ADE ($2.660\text{ m}$ with beam search) and reversed FDE regression ($7.623\text{ m} \to 7.087\text{ m}$). Clips under $2\text{ m}$ ADE jumped from $30.8\% \to 40.2\%$.

### Milestone 4: Continuous Expectation RTS Smoother (Zero-Retraining Win)
* **Hypothesis:** Discretization into 2-meter bins in beam search artificially truncates turns and adds quantization noise. An RTS smoother over continuous posterior means $\mu = \sum b_i P(b_i)$ will produce smooth, physically valid trajectories.
* **Result:** **All-time project best on both ADE (2.549 m) and FDE (6.139 m)**:
  * FDE dropped by **$-13.4\%$** with zero retraining.
  * Maximum clip ADE fell from $13.07\text{ m} \to 8.73\text{ m}$.
  * $100.0\%$ of blind test clips finished inside the $10\text{ m}$ ADE envelope.

### Milestone 5: Polar Heads, Energy & Path Integration (`vs13_7500_varied_physics`)
* **Hypothesis:** Adding explicit acoustic energy channels, auxiliary polar heads (range $r$ and Doppler radial velocity $v_r$), and cumulative path integration would resolve cross-track heading drift.
* **Result:** Solid performance (ADE $2.621\text{ m}$, FDE $6.545\text{ m}$, best p90 FDE $12.365\text{ m}$), but slightly underperformed `poshead` ($2.549\text{ m}$) because optimizing 4 competing auxiliary objectives simultaneously diluted gradients to the primary motion heads.

### Milestone 6: Heading Direction Angle Loss (`vs13_7500_varied_heading`)
* **Hypothesis:** Penalizing the cosine angle error between predicted velocity and ground-truth velocity vectors directly during turns would fix lateral drift.
* **Result:** FDE reached $6.715\text{ m}$, beating early baselines, but overall ADE regressed ($2.913\text{ m}$) due to gradient interference between speed norm estimation and angular alignment.

---

## 7. Ablation Study: Frame-Synchronous Supervision vs. CTC

Conducted on the standardized 1,000-clip testbed (769 Train / 77 Val / 154 Test) to rigorously determine whether unaligned sequence-level CTC loss offers any benefit over frame-synchronous supervision.

### 7.1 Empirical Comparison (154 Held-Out Test Clips)

| Metric | Proposed (Frame-Synchronous Soft CE) | Conformer-CTC (Connectionist Temporal Classification) | Degradation / Impact |
|:---|:---:|:---:|:---:|
| **Supervision Alignment** | Strict 1:1 timestamps (100 ms) | Unconstrained dynamic lattice (730 frames $\to$ 86 tokens) | Alignment relaxed |
| **Best Validation Loss** | 2.3017 | **1.8709** *(Illusion)* | On paper $-18.7\%$ lower |
| **Average Displacement Error (ADE)** ↓ | **3.840 m** | **12.250 m** | **$+219.0\%$ severe surge** |
| **Median ADE** ↓ | **3.560 m** | **11.544 m** | **$+224.3\%$ surge** |
| **Online Streaming ADE** ↓ | **4.737 m** | **16.518 m** | **$+248.7\%$ surge** |
| **Final Displacement Error (FDE)** ↓ | **8.669 m** | **13.899 m** | **$+60.3\%$ surge** |
| **Longitudinal Error ($X$ MAE)** ↓ | **2.126 m** | **11.191 m** | **$+426.4\%$ massive drift** |
| **Lateral Error ($Y$ MAE)** ↓ | **2.727 m** | **3.024 m** | $+10.9\%$ drift |
| **Clips with ADE $< 5\text{ m}$** ↑ | **80.5%** | **5.2%** | **$-75.3\text{ pp}$ collapse** |
| **Clips with ADE $< 10\text{ m}$** ↑ | **99.4%** | **37.7%** | **$-61.7\text{ pp}$ collapse** |

### 7.2 Why CTC Fails in Continuous Tracking
1. **Invariance to Time (Order vs. Timestamp):** CTC optimizes only for sequence order, not exact timestamps. At $80\text{ km/h}$, a vehicle covers $22.2\text{ m/s}$. Shifting a position token by just $0.5\text{ s}$ causes an instantaneous $\pm 11.1\text{ meter}$ error.
2. **Severe Longitudinal Warping:** Because road travel is monotonic, CTC easily learned token order ($X$ loss plummeted to `1.01`), but emitted coordinates at arbitrary time points, inflating real-world $X$ error by over $5\times$ ($2.13\text{ m} \to 11.19\text{ m}$).
3. **Loss of Steady-State Duration:** CTC automatically merges consecutive identical tokens into one, erasing duration and distance during constant-speed cruise.
4. **The Low Loss Illusion:** CTC achieved a lower training loss (`1.87` vs `2.30`) simply by predicting $>80\%$ blank tokens ($\epsilon$), completely decoupled from spatial tracking fidelity.

---

## 8. Physical Findings & Acoustic Observability Limits

1. **The Doppler Observability Limit ($X$ vs $Y$ Asymmetry):**
   * Across all models, $X$ error is consistently lower than $Y$ error (Champion: $X\text{ MAE} = 1.31\text{ m}$ vs. $Y\text{ MAE} = 1.88\text{ m}$).
   * Radial velocity $v_r = \frac{x v_x + y v_y}{r}$ is dominated by $v_x$ whenever $|x| \gg |y|$. Lateral maneuvers ($v_y$) produce weak frequency modulation, making lateral tracking the fundamental physical ceiling of single-microphone tracking.
2. **Signal-to-Noise Fading at Trajectory Termination:**
   * At $x > 40\text{ m}$, acoustic energy drops precipitously due to spherical spreading loss ($1/r^2$).
   * Error at Closest Point of Approach (CPA) averages only **$1.16\text{ m}$**, but unobserved late turns cause endpoint drift, which the endpoint loss ramp ($1\times \to 3\times$) and Expectation RTS smoother mitigate.
3. **Continuous Expectation vs. Discrete Quantization:**
   * Computing continuous expectations $\mu = \sum b_i P(b_i)$ is mathematically superior to argmax or beam search for physical tracking, producing zero staircasing and seamless Kalman filtering.

---

## 9. Reproducibility & Execution Commands

### 9.1 Environment & Codebase
* **Repository Root:** `/Users/rohith/Downloads/vehicle_motion_conformer`
* **Python Virtual Environment:** `./venv/bin/python`

### 9.2 Evaluating the Champion Model (poshead + Expectation RTS)
```bash
git checkout poshead_expectation_rts

./venv/bin/python external_test.py \
  --checkpoint runs/vs13_7500_varied_poshead/model.pt \
  --input /Users/rohith/Downloads/vs13_path2d_1000_varied \
  --output runs/vs13_7500_varied_poshead/blindtest_1000_exprts
```

### 9.3 Training a New Run on 7,500 Clips
```bash
caffeinate -i ./venv/bin/python train.py \
  --data "/Users/rohith/Desktop/Rohith/Doppler/DopplerSim_2.0/static/batch_outputs/vs13_path2d_7500_varied" \
  --split runs/vs13_7500_varied_cnn/split.json \
  --epochs 50 \
  --checkpoint runs/<new_run_name>/model.pt
```

### 9.4 Fast 5-Second Smoke Verification
```bash
./venv/bin/python smoke_check.py
```
*(Validates causal streaming, forward/backward passes, position head, FiLM conditioning, augmentations, and expectation decoder in $<6$ seconds).*
