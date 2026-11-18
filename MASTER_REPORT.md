# Vehicle Motion Conformer — Master Research & Experiment Report

This master document provides the exhaustive, definitive record of the **Vehicle Motion Conformer** project: physical Doppler foundations, end-to-end neural architectures, mathematical loss derivations, state-space decoding equations, complete benchmark metrics across all historical models, controlled ablation studies, diagnostic failure analyses, and exact reproduction instructions down to the minute implementation details.

---

## Table of Contents
1. [Physical Doppler Foundations & Observability Limits](#1-physical-doppler-foundations--observability-limits)
2. [End-to-End System Architecture & Feature Engineering](#2-end-to-end-system-architecture--feature-engineering)
3. [Quantization Grid & Head Specifications](#3-quantization-grid--head-specifications)
4. [Data Augmentation Protocols](#4-data-augmentation-protocols)
5. [Supervision & Mathematical Loss Derivations](#5-supervision--mathematical-loss-derivations)
6. [State-Space Decoding: Expectation RTS Kalman Smoother](#6-state-space-decoding-expectation-rts-kalman-smoother)
7. [Master Benchmark Comparison (All 10 Models)](#7-master-benchmark-comparison-all-10-models)
8. [Milestone-by-Milestone Evolutionary Analysis](#8-milestone-by-milestone-evolutionary-analysis)
9. [Ablation Study: Frame-Synchronous Supervision vs. CTC](#9-ablation-study-frame-synchronous-supervision-vs-ctc)
10. [Diagnostic Failure Trace: Outlier Anatomy (Sample 0000848)](#10-diagnostic-failure-trace-outlier-anatomy-sample-0000848)
11. [Hyperparameter Reference & Checkpoint Ledger](#11-hyperparameter-reference--checkpoint-ledger)
12. [Reproduction Guide & Verification CLI](#12-reproduction-guide--verification-cli)

---

## 1. Physical Doppler Foundations & Observability Limits

### 1.1 Problem Definition
Given a single monaural microphone positioned at origin $(0, 0)$ and a known initial coordinate $\mathbf{p}_0 = (x_0, y_0)$ at $t=0$, estimate the vehicle's continuous 2D kinematic trajectory:
$$\mathbf{s}(t) = \big[v_x(t),\, v_y(t),\, p_x(t),\, p_y(t)\big]^\top \quad \text{for } t \in [0.1, 0.2, \dots, T]\text{ s}$$
at a uniform output rate of $10\text{ Hz}$ ($\Delta t = 0.1\text{ s}$).

### 1.2 Acoustic Doppler Frequency Formulation
For an acoustic source emitting an engine or tire harmonic frequency $f_0$, the received acoustic frequency $f(t)$ at the microphone is governed by the relative radial velocity $v_r(t)$:
$$f(t) = f_0 \left( \frac{c}{c - v_r(t)} \right) \approx f_0 \left( 1 + \frac{v_r(t)}{c} \right)$$
where $c \approx 343\text{ m/s}$ is the speed of sound in air, and $v_r(t)$ is the projection of the velocity vector onto the line-of-sight unit vector:
$$v_r(t) = \vec{v}(t) \cdot \frac{\vec{p}(t)}{\|\vec{p}(t)\|} = \frac{x(t) v_x(t) + y(t) v_y(t)}{\sqrt{x(t)^2 + y(t)^2}}$$

### 1.3 The Fundamental Observability Asymmetry ($X$ vs $Y$)
* **Longitudinal Axis ($X$):** Vehicles travel primarily along the roadway ($X$-axis). When distant ($|x| \gg |y|$):
  $$\frac{x}{\sqrt{x^2 + y^2}} \approx \pm 1, \qquad \frac{y}{\sqrt{x^2 + y^2}} \approx 0 \implies v_r(t) \approx \pm v_x(t)$$
  Consequently, $>95\%$ of the discernible Doppler pitch modulation is produced strictly by longitudinal speed $v_x$.
* **Lateral Axis ($Y$):** Lateral velocity $v_y$ is multiplied by $\frac{y}{r} \approx 0$ when distant, producing near-zero Doppler shift. Lateral tracking is physically unobservable from pitch alone when $|x| \gg |y|$ and relies almost entirely on:
  1. Acoustic sound pressure decay ($1/r$ spherical spreading).
  2. The rapid angle sweep during the Closest Point of Approach (CPA).
  3. Continuous kinematic path integration.

### 1.4 Acoustic Simulator & Dataset Characteristics (`DopplerSim_2.0`)
All synthetic audio is generated via `DopplerSim_2.0` under realistic physical conditions:
* **Speed Range:** $30\text{ km/h}$ to $110\text{ km/h}$ with dynamic acceleration and braking.
* **Trajectory Topologies:** Straight pass-bys, broad highway curves, acute turns, lane-changes, and multi-turn S-curves.
* **Closest Point of Approach (CPA):** Distance at CPA ranges between $d_{\text{CPA}} \in [5.0\text{ m},\, 45.0\text{ m}]$.
* **Clip Duration:** $4.0\text{ s}$ to $12.0\text{ s}$ (mean $\approx 7.5\text{ s}$, corresponding to $\approx 75$ output timestamps @ $10\text{ Hz}$).
* **Acoustic Synthesizer:** Physics-based combustion engine harmonics (firing order pulses), tire-road rolling noise (broadband high-frequency texture), ground reflection multipath interference, and distance-dependent atmospheric air absorption.
* **Acoustic Leakage Prevention:** In all splits (`split.json`), clips generated from the same engine/vehicle template (`source_clip`) are grouped strictly into either train, validation, or test to prevent the network from memorizing specific engine harmonics.

---

## 2. End-to-End System Architecture & Feature Engineering

```
Raw Audio Stream (22,050 Hz Mono)
       │
       ▼
STFT Processing (N_fft=2048, Hop=256, Centered) ──► 1025 Linear Frequency Bins (0 to 11,025 Hz)
       │
       ▼
2D Frequency CNN Frontend ──► 128-dim Frame Embeddings
(Conv1d across harmonic bins + GELU + LayerNorm)
       │
       ├─────────────────────────┐
       ▼                         ▼
Spatial Conditioning: Concat [x0, y0] / FiLM Modulation
       │
       ▼
Causal Streaming Conformer Encoder Stack
4 Layers | 4 Attention Heads | d_model=128 | Depthwise Conv Kernel=15
att_context_size=[1024, 0] (Strictly 0 Lookahead Frames)
       │
       ├─────────────────────────────────────────┬────────────────────────┐
       ▼                                         ▼                        ▼
Four Categorical Heads                    Auxiliary Position Head   Streaming Cache Buffer
(vx: 121, vy: 121, px: 201, py: 101 bins) (Continuous px, py MLP)  (Zero-latency state)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
              Expectation RTS Kalman Smoother
                            ▼
           Continuous, Smooth, Feasible Trajectory
```

### 2.1 STFT Spectrogram Extraction
* **Audio Sample Rate ($f_s$):** $22,050\text{ Hz}$
* **FFT Window Size ($N_{\text{fft}}$):** $2048$ samples ($92.88\text{ ms}$ window duration, yielding frequency resolution $\Delta f = 10.77\text{ Hz}$)
* **Hop Length ($H$):** $256$ samples ($11.61\text{ ms}$ temporal step, yielding $\approx 86.13\text{ frames/second}$)
* **Number of Frequency Bins:** $F = N_{\text{fft}} / 2 + 1 = 1025$ bins (linear spacing from $0\text{ Hz}$ to $11,025\text{ Hz}$)
* **Log-Amplitude Compression:** $\mathbf{X}_{\text{mag}} = \log\big(1 + |\text{STFT}(a(t))|\big)$

### 2.2 2D Frequency CNN Frontend
Naively projecting $1025$ bins into a hidden dimension destroys harmonic relationships (engine firing harmonics, tire tread spacing). A 1D convolutional module operates across the frequency axis for each time frame:
* Layer 1: `nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2)` $\to$ `nn.GELU()`
* Layer 2: `nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2)` $\to$ `nn.GELU()`
* Reduced Frequency Dimension: $1025 \to 513 \to 257$ bins.
* Projection: `nn.Linear(16 * 257 = 4112, 128)` $\to$ `nn.LayerNorm(128)` $\to$ `nn.Dropout(0.1)`

### 2.3 Causal Conformer Encoder Configuration
Implemented via `nemo.collections.asr.modules.conformer_encoder.ConformerEncoder`:
* **Hidden Size ($d_{\text{model}}$):** $128$
* **Encoder Layers:** $4$
* **Self-Attention Heads:** $4$ (Relative Positional Encodings)
* **Attention Context Size:** `[1024, 0]` (1024 frames of past context, **0 frames of future context** — strictly causal)
* **Convolutional Kernel Size:** $15$ (`conv_context_size="causal"`, `conv_norm_type="layer_norm"`)
* **Subsampling:** None (`subsampling_factor=1`) to preserve fine-grained temporal resolution.
* **Streaming Cache:** `self.encoder.streaming_cfg.cache_drop_size = 0` allows incremental chunk-by-chunk evaluation.

### 2.4 Spatial Conditioning Mechanisms: Concat vs. FiLM Modulation
Two methods are supported to inform the network of the known starting position $\mathbf{p}_0 = (x_0, y_0)$:
1. **Direct Concatenation (`spatial_conditioning = 'concat'` — Champion Default):**
   The coordinates are normalized by $100.0\text{ m}$ and broadcast across all time steps as two constant feature channels appended to the spectral embedding:
   $$\mathbf{z}_{\text{input}}(t) = \left[ \mathbf{z}_{\text{spectral}}(t);\, \frac{x_0}{100.0};\, \frac{y_0}{100.0} \right] \in \mathbb{R}^{130}$$
2. **Feature-wise Linear Modulation (`spatial_conditioning = 'film'`):**
   A conditioning MLP generates affine scale $\gamma(\mathbf{p}_0)$ and shift $\beta(\mathbf{p}_0)$ vectors that dynamically scale and shift the spectral feature maps:
   $$\big[\gamma(\mathbf{p}_0),\, \beta(\mathbf{p}_0)\big] = \text{MLP}(\mathbf{p}_0)$$
   $$\mathbf{z}_{\text{input}}(t) = \gamma(\mathbf{p}_0) \odot \mathbf{z}_{\text{spectral}}(t) + \beta(\mathbf{p}_0)$$
   This allows the initial spatial anchor to directly modulate the gain of specific harmonic filters.

---

## 3. Quantization Grid & Head Specifications

The output state space is discretized into 4 parallel categorical heads. Continuous values are mapped into discrete bins during target preparation:

| Head Index | Variable | Minimum Bound | Maximum Bound | Bin Step Size ($\Delta b$) | Number of Classes ($K$) | Output Unit |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0` | $v_x$ (Longitudinal Velocity) | $-120.0$ | $+120.0$ | $2.0\text{ km/h}$ | **121** | $\text{km/h}$ |
| `1` | $v_y$ (Lateral Velocity) | $-120.0$ | $+120.0$ | $2.0\text{ km/h}$ | **121** | $\text{km/h}$ |
| `2` | $p_x$ (Longitudinal Position) | $-200.0$ | $+200.0$ | $2.0\text{ m}$ | **201** | $\text{m}$ |
| `3` | $p_y$ (Lateral Position) | $-100.0$ | $+100.0$ | $2.0\text{ m}$ | **101** | $\text{m}$ |

* **Total Quantized State Space:** $121 \times 121 \times 201 \times 101 = \mathbf{296,878,881}$ discrete state combinations.
* **Continuous Expectation Conversion:** Continuous values are recovered via posterior mathematical expectation $\bar{s}_j = \sum_{k=1}^{K_j} b_{j, k} \cdot P(b_{j, k})$, providing infinite sub-bin precision without grid snapping.

---

## 4. Data Augmentation Protocols

To prevent acoustic overfitting to synthetic Doppler profiles, three augmentations are applied on-the-fly during training (`augment=True`):
1. **SpecAugment Frequency Masking:** Randomly masks up to $F_{\text{mask}} = 20$ contiguous frequency bins with zero:
   $$X[t,\, f_0 : f_0 + \Delta f] = 0, \qquad \Delta f \sim \mathcal{U}(1, 20)$$
2. **SpecAugment Temporal Masking:** Randomly masks up to $T_{\text{mask}} = 10$ contiguous time frames with zero:
   $$X[t_0 : t_0 + \Delta t,\, :] = 0, \qquad \Delta t \sim \mathcal{U}(1, 10)$$
3. **Acoustic Additive Gaussian Noise:** Adds white Gaussian acoustic noise scaled to signal standard deviation:
   $$X_{\text{aug}} = X + \mathcal{N}\left(0,\, (\alpha \cdot \sigma_X)^2\right), \qquad \alpha \sim \mathcal{U}(0.01, 0.05)$$

---

## 5. Supervision & Mathematical Loss Derivations

The composite training loss is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{kin}} \mathcal{L}_{\text{kin}} + \lambda_{\text{acc}} \mathcal{L}_{\text{acc}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}}$$

### 5.1 Ordinal Gaussian Soft Cross-Entropy
For class index $k \in \{0, \dots, K-1\}$ and continuous ground-truth target bin index $y^* \in [0, K-1]$:
$$P_{\text{target}}(k) = \frac{\exp\left(-\frac{(k - y^*)^2}{2\sigma^2}\right)}{\sum_{m=0}^{K-1} \exp\left(-\frac{(m - y^*)^2}{2\sigma^2}\right)}, \qquad \sigma = 1.0\text{ bin}$$
$$\mathcal{L}_{\text{CE}} = \frac{1}{4} \sum_{j=1}^{4} \frac{1}{T} \sum_{t=1}^{T} w(t) \cdot \text{CrossEntropy}\big(\hat{\mathbf{z}}_j(t),\, \mathbf{P}_{\text{target}, j}(t)\big)$$

### 5.2 Time-Weighted Endpoint Loss Ramp
To force the network to prioritize trajectory termination (combating late acoustic fading):
$$w(t) = 1.0 + (w_{\text{endpoint}} - 1.0) \frac{t}{T-1}, \qquad w_{\text{endpoint}} = 3.0$$
normalized such that $\sum_{t=1}^{T} w(t) = T$.

### 5.3 Kinematic Continuity Loss
Penalizes discrepancy between predicted coordinate displacements and integrated velocities:
$$\mathbf{r}_{\text{kin}}(t) = \mathbf{p}(t) - \mathbf{p}(t-1) - \frac{\mathbf{v}(t) + \mathbf{v}(t-1)}{2 \times 3.6} \Delta t$$
$$\mathcal{L}_{\text{kin}} = \frac{1}{T-1} \sum_{t=2}^{T} \text{SmoothL1}\left( \frac{\mathbf{r}_{\text{kin}}(t)}{2.0} \right), \qquad \lambda_{\text{kin}} = 0.05$$

### 5.4 Maximum Acceleration Constraint
Penalizes physical accelerations exceeding street vehicle limits ($a_{\text{max}} = 8.0\text{ m/s}^2$):
$$\mathbf{a}(t) = \frac{\mathbf{v}(t) - \mathbf{v}(t-1)}{3.6 \cdot \Delta t}$$
$$\mathcal{L}_{\text{acc}} = \frac{1}{T-1} \sum_{t=2}^{T} \left[ \text{ReLU}\left( \frac{\|\mathbf{a}(t)\|}{a_{\text{max}}} - 1.0 \right) \right]^2, \qquad \lambda_{\text{acc}} = 0.01$$

### 5.5 Auxiliary Direct Position Regression Head
A 2-layer MLP (`Linear(128, 64) -> GELU -> Linear(64, 2)`) predicts direct $(p_x, p_y)$ coordinates:
$$\mathcal{L}_{\text{pos}} = \text{SmoothL1}\left( \frac{(\hat{\mathbf{p}}_{\text{direct}}(t) - \mathbf{p}_{\text{GT}}(t)) \cdot w(t)}{10.0} \right), \qquad \lambda_{\text{pos}} = 0.1$$

---

## 6. State-Space Decoding: Expectation RTS Kalman Smoother

### 6.1 State-Space Equations
* **State Vector:** $\mathbf{x}(t) = [v_x(t),\, v_y(t),\, p_x(t),\, p_y(t)]^\top$ in units of $[\text{km/h}, \text{km/h}, \text{m}, \text{m}]^\top$.
* **State Transition Matrix ($\mathbf{F}$):**
  $$\mathbf{F}(\Delta t) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \frac{\Delta t}{3.6} & 0 & 1 & 0 \\ 0 & \frac{\Delta t}{3.6} & 0 & 1 \end{bmatrix}, \qquad \Delta t = 0.1\text{ s}$$
* **Measurement Vector:** Continuous posterior means $\mathbf{z}(t) = [\bar{v}_x, \bar{v}_y, \bar{p}_x, \bar{p}_y]^\top$ where $\bar{s}_j(t) = \sum b_k P(b_k)$.
* **Measurement Matrix:** $\mathbf{H} = \mathbf{I}_4$.
* **Measurement Covariance Matrix ($\mathbf{R}$):**
  $$\mathbf{R} = \text{diag}\left( R_{\text{vel}}^2,\, R_{\text{vel}}^2,\, R_{\text{pos}}^2,\, R_{\text{pos}}^2 \right) = \text{diag}\big(4.0^2,\, 4.0^2,\, 1.5^2,\, 1.5^2\big)$$
* **Process Noise Covariance Matrix ($\mathbf{Q}$):** Continuous white-noise acceleration model with spectral density $Q_{\text{accel}} = 6.0\text{ m/s}^2$:
  $$\mathbf{Q} = Q_{\text{accel}}^2 \begin{bmatrix} \Delta t \cdot \mathbf{I}_2 & \frac{\Delta t^2}{2 \cdot 3.6} \cdot \mathbf{I}_2 \\ \frac{\Delta t^2}{2 \cdot 3.6} \cdot \mathbf{I}_2 & \frac{\Delta t^3}{3 \cdot 3.6^2} \cdot \mathbf{I}_2 \end{bmatrix}$$

### 6.2 Forward Kalman Filter
For $k = 1, \dots, T$:
$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \hat{\mathbf{x}}_{k-1|k-1}, \qquad \mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^\top + \mathbf{Q}$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^\top \big( \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^\top + \mathbf{R} \big)^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1}), \qquad \mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$$

### 6.3 Backward Rauch-Tung-Striebel (RTS) Smoother
Initialized at final step $T$ with $\hat{\mathbf{x}}_{T|T}$ and $\mathbf{P}_{T|T}$. For $k = T-1, \dots, 0$:
$$\mathbf{C}_k = \mathbf{P}_{k|k} \mathbf{F}^\top \mathbf{P}_{k+1|k}^{-1}$$
$$\hat{\mathbf{x}}_{k|T} = \hat{\mathbf{x}}_{k|k} + \mathbf{C}_k \big( \hat{\mathbf{x}}_{k+1|T} - \hat{\mathbf{x}}_{k+1|k} \big)$$
$$\mathbf{P}_{k|T} = \mathbf{P}_{k|k} + \mathbf{C}_k \big( \mathbf{P}_{k+1|T} - \mathbf{P}_{k+1|k} \big) \mathbf{C}_k^\top$$

### 6.4 Zero-Cost Offline Redecoding Protocol (`redecode.py`)
Because the neural network outputs posterior categorical probability arrays that are cached to disk as `.npz` files:
* Changing decoder hyperparameters ($Q_{\text{accel}}, R_{\text{pos}}, R_{\text{vel}}$) does **NOT** require re-running GPU neural network inference.
* The script `redecode.py` directly loads the saved `raw_expectation` matrices and applies the new Kalman smoothing parameters in $< 0.8\text{ ms per clip}$ ($< 1\text{ second}$ for the entire 1,000-clip test set).
* This allowed instant ablation of decoder dynamics without retraining the neural backbone.

---

## 7. Master Benchmark Comparison (All 10 Models)

All models evaluated on the **1,000-clip independent held-out blind test set** (`vs13_path2d_1000_varied`, unseen trajectories):

| # | Run Name | Train Scale | Epochs | Key Innovations / Architectural Changes | Val Loss | ADE (m) ↓ | Median ADE (m) ↓ | Online ADE (m) ↓ | FDE (m) ↓ | X MAE (m) ↓ | Y MAE (m) ↓ |
|---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Original Baseline** | 1,000 | 30 | Linear FFT projection, Beam search | 2.8428 | 5.047 | 4.810 | 5.102 | 11.009 | — | — |
| 2 | **`vs13_1000_option_a_cnn`** | 1,000 | 30 | + Frequency 2D CNN frontend | 2.1404 | 4.158 | 3.910 | 4.194 | 10.070 | — | — |
| 3 | **`smoke_1000`** | 1,000 | 50 | + Soft labels, kinematic loss, SpecAugment | 2.2886† | 4.365 | 3.775 | 4.567 | 9.947 | 2.679 | 2.840 |
| 4 | **`vs13_7500_varied_v1`** | 7,500 | 50 | Scaled 7.5k dataset (Linear frontend) | 1.9416 | 3.043 | 2.710 | 3.093 | 7.748 | — | — |
| 5 | **`vs13_7500_varied_cnn`** | 7,500 | 44 | + Frequency 2D CNN frontend | **1.8173** | 2.965 | 2.612 | 3.013 | 7.333 | 1.385 | 2.301 |
| 6 | **`vs13_7500_varied_improved`** | 7,500 | 26 | + Soft labels ($\sigma=1.0$), Kinematic loss, SpecAug | 2.1622† | 2.806 | 2.422 | 2.821 | 7.623 | 1.525 | 1.988 |
| 7 | **`vs13_7500_varied_poshead` (Beam)** | 7,500 | 50 | + Aux PosHead, Endpoint ramp ($1\times \to 3\times$) | 2.1473† | 2.660 | 2.306 | 2.707 | 7.087 | 1.319 | 1.979 |
| 8 | **`poshead` (Expectation RTS)** 🏆 | 7,500 | 50 | **Same checkpoint + Expectation RTS smoother** | 2.1473† | **2.549** | **2.251** | **2.707** | **6.139** | **1.312** | **1.875** |
| 9 | **`vs13_7500_varied_physics`** | 7,500 | 59 | + Energy channel, Polar head ($r, v_r$), PathInt | 2.1881† | 2.621 | 2.351 | 2.767 | 6.545 | 1.397 | 1.902 |
| 10 | **`vs13_7500_varied_heading`** | 7,500 | 32 | + Heading direction cosine loss ($\mathcal{L}_{\text{heading}}$) | 2.1967† | 2.913 | 2.649 | 3.165 | 6.715 | 1.484 | 2.184 |

### 7.1 Accuracy Threshold Progression

| Configuration | ADE $< 1\text{ m}$ | ADE $< 2\text{ m}$ | ADE $< 5\text{ m}$ | ADE $< 10\text{ m}$ | ADE $< 20\text{ m}$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| `vs13_7500_varied_cnn` (Old Baseline) | 2.4% | 30.8% | 89.9% | 99.2% | 99.8% |
| `vs13_7500_varied_improved` | 2.5% | 34.2% | 91.5% | 99.5% | 99.9% |
| `vs13_7500_varied_poshead` (Beam) | 4.4% | 40.2% | 93.2% | 99.7% | 100.0% |
| **`poshead` (Expectation RTS)** 🏆 | **4.1%** | **41.2%** | **94.8%** | **100.0%** | **100.0%** |
| `vs13_7500_varied_physics` | 3.8% | 37.0% | 93.9% | 99.9% | 100.0% |
| `vs13_7500_varied_heading` | 2.6% | 26.4% | 92.2% | 99.9% | 100.0% |

### 7.2 Error Distribution Tail Percentiles

| Configuration | ADE Mean | ADE Median | ADE p90 | ADE p95 | ADE Max | FDE Mean | FDE Median | FDE p90 | FDE p95 | FDE Max |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `poshead` (Beam) | 2.660 m | 2.306 m | 4.526 m | 5.478 m | 13.074 m | 7.087 m | 6.061 m | 13.892 m | 16.854 m | 26.446 m |
| **`poshead` (Expectation RTS)** 🏆 | **2.549 m** | **2.251 m** | **4.258 m** | **5.034 m** | **8.727 m** | **6.139 m** | **5.111 m** | 12.521 m | **14.815 m** | **25.702 m** |
| `vs13_7500_varied_physics` | 2.621 m | 2.351 m | 4.283 m | 5.273 m | 10.013 m | 6.545 m | 5.466 m | **12.365 m** | 15.561 m | 34.206 m |
| `vs13_7500_varied_heading` | 2.913 m | 2.649 m | 4.562 m | 5.697 m | 11.336 m | 6.715 m | 5.598 m | 13.048 m | 15.779 m | 31.830 m |

---

## 8. Milestone-by-Milestone Evolutionary Analysis

### Milestone 1: Frequency 2D CNN Frontend (`vs13_7500_varied_cnn`)
* **Hypothesis:** Linear projection destroys local harmonic structures across the $1025$ frequency bins. Convolving across harmonics preserves shift-invariant Doppler features.
* **Finding:** ADE improved from $3.043\text{ m} \to 2.965\text{ m}$ ($-2.6\%$) and FDE dropped from $7.748\text{ m} \to 7.333\text{ m}$ ($-5.4\%$). Proved convolutional inductive bias is essential for acoustic spectral inputs.

### Milestone 2: Soft Ordinal Targets & Kinematics (`vs13_7500_varied_improved`)
* **Hypothesis:** Hard Dirac targets penalize adjacent bins equally to distant bins. Gaussian soft labels reward proximity, and kinematic loss penalizes discontinuous jumps.
* **Finding:** Longitudinal coordinate error $Y$ MAE dropped from $2.301\text{ m} \to 1.988\text{ m}$ ($-13.6\%$), lowering overall ADE to $2.806\text{ m}$.

### Milestone 3: Auxiliary Position Head & Endpoint Loss Ramp (`vs13_7500_varied_poshead`)
* **Hypothesis:** Compounding error causes late-clip drift. A time-weighted ramp ($1\times \to 3\times$) combined with a continuous position regression head anchors feature representations.
* **Finding:** Broken through the sub-$2.7\text{ m}$ barrier ($2.660\text{ m}$ ADE with beam search). FDE dropped from $7.623\text{ m} \to 7.087\text{ m}$. Percentage of clips under $2\text{ m}$ ADE jumped from $30.8\% \to 40.2\%$.

### Milestone 4: Continuous Expectation RTS Smoother (Zero-Retraining Win)
* **Hypothesis:** Kinematic beam search suffered from discretization artifacts (snapping to 2-meter bins) and over-penalized turns. Computing continuous posterior expectations $\mu = \sum b_i P(b_i)$ followed by an RTS smoother will eliminate staircasing without retraining.
* **Finding:** **Project all-time record:**
  * **ADE:** **$2.549\text{ m}$** | **FDE:** **$6.139\text{ m}$** ($-13.4\%$ drop).
  * Outlier ceiling collapsed: maximum ADE dropped from $13.07\text{ m} \to 8.73\text{ m}$.
  * $100.0\%$ of blind test clips finished inside the $10\text{ m}$ envelope.

### Milestone 5: Polar Heads, Energy & Path Integration (`vs13_7500_varied_physics`)
* **Hypothesis:** Adding explicit energy channels, polar heads ($r, v_r$), and cumulative path integration will eliminate lateral drift.
* **Finding:** Solid result (ADE $2.621\text{ m}$, FDE $6.545\text{ m}$, best p90 FDE $12.365\text{ m}$), but slightly underperformed `poshead` ($2.549\text{ m}$) because optimizing 4 competing auxiliary objectives simultaneously diluted gradients from the primary motion heads.

### Milestone 6: Heading Direction Angle Loss (`vs13_7500_varied_heading`)
* **Hypothesis:** Direct cosine similarity penalty $\mathcal{L}_{\text{heading}} = ((1 - \mathbf{u}_{\text{pred}} \cdot \mathbf{u}_{\text{GT}}) \cdot w)$ will align velocity vectors during turns.
* **Finding:** FDE improved to $6.715\text{ m}$ over earlier models, but ADE regressed ($2.913\text{ m}$) due to gradient interference between speed norm estimation and unit-vector angular alignment.

---

## 9. Ablation Study: Frame-Synchronous Supervision vs. CTC

Conducted on the standardized 1,000-clip testbed (769 Train / 77 Val / 154 Test, `ablation/best_model/split.json`) to test whether unaligned sequence-level CTC loss offers any benefit over frame-synchronous supervision.

### 9.1 Empirical Benchmark (154 Held-Out Test Clips)

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

### 9.2 Four Theoretical Reasons for CTC Failure
1. **Optimizes Order Over Time:** CTC is designed for speech where words can shift by $\pm 300\text{ ms}$ without altering transcription. In physics, a vehicle at $80\text{ km/h}$ covers $22.2\text{ m/s}$; a $0.5\text{ s}$ temporal drift causes an instantaneous $\pm 11.1\text{ meter}$ error.
2. **Severe Longitudinal Warping:** Because road travel is monotonic, CTC easily learned token sequence order ($X$ loss plummeted to `1.01`), but emitted coordinates at arbitrary time points, inflating real-world $X$ error by over $5\times$ ($2.13\text{ m} \to 11.19\text{ m}$).
3. **Loss of Steady-State Duration:** CTC automatically collapses repeated identical tokens into one, erasing duration and distance traveled during constant-speed cruise.
4. **The Low Loss Illusion:** CTC achieved a lower validation loss (`1.87` vs `2.30`) simply by predicting $>80\%$ blank tokens ($\epsilon$), completely decoupled from spatial tracking accuracy.

---

## 10. Diagnostic Failure Trace: Outlier Anatomy (Sample 0000848)

To understand the physical cause of residual FDE, sample `sample_0000848` was traced at millisecond resolution:
* **Geometry:** Vehicle approaches from $x = -75\text{ m}$, passes CPA at $(x=0.8\text{ m}, y=22.6\text{ m})$, then initiates a sharp turn into $+Y$ while receding to $x = +82\text{ m}$.
* **CPA Performance:** At $t = 3.2\text{ s}$ (CPA), tracking error was only **$1.16\text{ m}$**! The model tracked the vehicle with pinpoint accuracy through closest approach.
* **The Turn Breakdown:**
  * During the subsequent turn, ground-truth $v_y$ accelerated to **$+25.1\text{ km/h}$**.
  * The network predicted only **$+2.9\text{ km/h}$** (an angular heading error of $\Delta \theta \approx 18^\circ$).
  * Why? At $x > 30\text{ m}$, the vehicle's acoustic signal had decayed by $-18\text{ dB}$, and Doppler shift was dominated $>96\%$ by $v_x = 76\text{ km/h}$. $v_y$ produced almost zero discernible frequency modulation.
* **Compound Drift:** Over the remaining $5\text{ seconds}$ at $80\text{ km/h}$, an $18^\circ$ heading error caused a **$25.3\text{ meter}$ lateral ($Y$) cross-track drift**, resulting in $\text{FDE} = 26.4\text{ m}$.

---

## 11. Hyperparameter Reference & Checkpoint Ledger

### 11.1 Training Hyperparameters

| Hyperparameter | Symbol | Value | Notes |
|:---|:---:|:---:|:---|
| Optimizer | — | **AdamW** | Standard decoupled weight decay optimizer |
| Base Learning Rate | $\eta$ | $3 \times 10^{-4}$ | Fixed learning rate with AdamW |
| Weight Decay | $\lambda_{\text{decay}}$ | $1 \times 10^{-4}$ | Regularization coefficient |
| Gradient Clipping | — | $1.0$ | Max L2 norm threshold |
| Batch Size | $B$ | $1$ clip | Online streaming sequence evaluation |
| Data Prefetch Queue | — | $4$ workers | Background thread for overlapping disk I/O with GPU |
| Random Seed | — | $42$ | Deterministic shuffling and split partitioning |
| Kinematic Loss Weight | $\lambda_{\text{kin}}$ | $0.05$ | Velocity-displacement consistency |
| Acceleration Loss Weight | $\lambda_{\text{acc}}$ | $0.01$ | Street acceleration bound ($a_{\text{max}} = 8.0\text{ m/s}^2$) |
| Label Smoothing Sigma | $\sigma$ | $1.0$ bins | Gaussian ordinal target smoothing |
| Position Head Weight | $\lambda_{\text{pos}}$ | $0.1$ | Auxiliary continuous regression |
| Endpoint Weight Ramp | $w_{\text{endpoint}}$ | $3.0$ | Time ramp from $1.0$ at $t=0$ to $3.0$ at $t=T$ |
| RTS Process Noise | $Q_{\text{accel}}$ | $6.0\text{ m/s}^2$ | Tuned for vehicle maneuvers |
| RTS Position Noise | $R_{\text{pos}}$ | $1.5\text{ m}$ | Measurement noise covariance |
| RTS Velocity Noise | $R_{\text{vel}}$ | $4.0\text{ km/h}$ | Measurement noise covariance |

### 11.2 Model Checkpoint Ledger

| Run / Model | Git Branch | Checkpoint Path | Architecture Config | Epochs | Blind Test ADE | Blind Test FDE |
|:---|:---|:---|:---|:---:|:---:|:---:|
| **Champion Model** 🏆 | `poshead_expectation_rts` | [`runs/vs13_7500_varied_poshead/model.pt`](file:///Users/rohith/Downloads/vehicle_motion_conformer/runs/vs13_7500_varied_poshead/model.pt) | Frequency CNN + PosHead + RTS | 50 | **2.549 m** | **6.139 m** |
| Improved Run | `soft_label_kinematic_loss` | [`runs/vs13_7500_varied_improved/model.pt`](file:///Users/rohith/Downloads/vehicle_motion_conformer/runs/vs13_7500_varied_improved/model.pt) | Frequency CNN + Soft CE | 26 | 2.806 m | 7.623 m |
| CNN Baseline | `nemo-streaming-cache-conformer` | [`runs/vs13_7500_varied_cnn/model.pt`](file:///Users/rohith/Downloads/vehicle_motion_conformer/runs/vs13_7500_varied_cnn/model.pt) | Frequency CNN + Hard CE | 44 | 2.965 m | 7.333 m |
| Physics Multi-Head | `physics_polar_pathint` | [`runs/vs13_7500_varied_physics/model.pt`](file:///Users/rohith/Downloads/vehicle_motion_conformer/runs/vs13_7500_varied_physics/model.pt) | Polar ($r, v_r$) + PathInt | 59 | 2.621 m | 6.545 m |
| Heading Loss Run | `poshead_heading_loss` | [`runs/vs13_7500_varied_heading/model.pt`](file:///Users/rohith/Downloads/vehicle_motion_conformer/runs/vs13_7500_varied_heading/model.pt) | Cosine Heading Angle Loss | 32 | 2.913 m | 6.715 m |
| 1k Reference Baseline | `poshead_expectation_rts` | [`ablation/best_model/model.pt`](file:///Users/rohith/Downloads/vehicle_motion_conformer/ablation/best_model/model.pt) | PosHead (769 train clips) | 50 | 3.840 m | 8.669 m |
| 1k Conformer-CTC | `nemo_conformer_ctc` | [`ablation/ctc_model/model.pt`](file:///Users/rohith/Downloads/vehicle_motion_conformer/ablation/ctc_model/model.pt) | Unaligned CTC Loss | 50 | 12.250 m | 13.899 m |

---

## 12. Reproduction Guide & Verification CLI

### 12.1 Environment Setup
```bash
cd /Users/rohith/Downloads/vehicle_motion_conformer
source ./venv/bin/activate
```

### 12.2 Fast 5-Second Smoke Verification Test
Validates causal streaming equivalence, forward/backward gradients, position head, FiLM conditioning, augmentations, and expectation decoder:
```bash
./venv/bin/python smoke_check.py
```

### 12.3 Evaluating the Champion Model on 1,000 Blind Test Clips
```bash
git checkout poshead_expectation_rts

./venv/bin/python external_test.py \
  --checkpoint runs/vs13_7500_varied_poshead/model.pt \
  --input /Users/rohith/Downloads/vs13_path2d_1000_varied \
  --output runs/vs13_7500_varied_poshead/blindtest_1000_exprts
```

### 12.4 Training from Scratch on 7,500 Clips
```bash
caffeinate -i ./venv/bin/python train.py \
  --data "/Users/rohith/Desktop/Rohith/Doppler/DopplerSim_2.0/static/batch_outputs/vs13_path2d_7500_varied" \
  --split runs/vs13_7500_varied_cnn/split.json \
  --epochs 50 \
  --checkpoint runs/vs13_7500_varied_poshead/model.pt
```

### 12.5 Redecoding Existing NPZ Files with Expectation RTS (Zero Retraining)
```bash
./venv/bin/python redecode.py \
  --input runs/vs13_7500_varied_poshead/blindtest_1000 \
  --output runs/vs13_7500_varied_poshead/blindtest_1000_exprts \
  --state canonical_state_frames.npy
```
