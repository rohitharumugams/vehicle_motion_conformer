# Vehicle Motion Conformer: Acoustic Trajectory Tracking

Real-time 2D vehicle motion estimation (velocities and positions) from single-microphone pass-by audio, powered by a custom **Frequency CNN frontend**, an **NVIDIA NeMo Streaming Cache Conformer**, and a **Kinematic RTS Kalman Decoder**.

---

## Table of Contents
1. [Overview & Core Concept](#overview--core-concept)
2. [Repurposing NVIDIA NeMo](#repurposing-nvidia-nemo)
3. [Architecture Deep Dive](#architecture-deep-dive)
   - [Acoustic Preprocessing & Spatial Conditioning](#1-acoustic-preprocessing--spatial-conditioning)
   - [Frequency CNN Frontend](#2-frequency-cnn-frontend)
   - [NVIDIA NeMo Conformer Encoder](#3-nvidia-nemo-conformer-encoder)
   - [Four Categorical Prediction Heads](#4-four-categorical-prediction-heads)
   - [Kinematic Beam Search & RTS Kalman Decoder](#5-kinematic-beam-search--rts-kalman-decoder)
4. [Latest Benchmark Metrics](#latest-benchmark-metrics)
5. [Repository Structure](#repository-structure)
6. [Installation & Setup](#installation--setup)
7. [Usage & Workflows](#usage--workflows)
   - [Training](#training)
   - [Single-Clip Streaming Inference](#single-clip-streaming-inference)
   - [Independent Blind Evaluation](#independent-blind-evaluation)
   - [Offline Re-Decoding & Parameter Tuning](#offline-re-decoding--parameter-tuning)
8. [Output Archive Format (`.npz`)](#output-archive-format-npz)

---

## Overview & Core Concept

Tracking a moving vehicle using only a passive acoustic sensor (a single microphone) is a challenging inverse problem. Pass-by audio carries distinct physical signatures:
* **Acoustic Doppler Effect:** The observed engine and tire frequencies compress as the vehicle approaches the sensor and expand as it recedes.
* **Tire/Road Noise & Harmonics:** High-frequency broadband friction and discrete engine firing harmonics vary with vehicle velocity and distance.

This repository implements a strictly **causal, real-time streaming pipeline** that reconstructs the vehicle's continuous 2D position trajectory ($p_x, p_y$) and velocity vector ($v_x, v_y$) at 10 Hz (every 100 ms) using only the audio stream and an initial starting location $(p_{x0}, p_{y0})$.

```
Streaming Audio (22.05 kHz) ──► Causal STFT (1,025 bins) ──┐
                                                           ▼
Initial Anchor (x0, y0) ────────────────────────► [Frequency CNN] (1D spectral convolutions)
                                                           │
                                                           ▼ (130 dims: 128 audio + 2 pos)
                                           [NVIDIA NeMo Streaming Conformer]
                                              (4 layers, [1024, 0] causal attention)
                                                           │
                                                           ▼ (10 Hz outputs)
                                               [4 Categorical Heads]
                                              (P(vx), P(vy), P(px), P(py))
                                                           │
                                                           ▼
                                           [Kinematic Beam Search & RTS Smoother]
                                              (Acceleration bounds <= 8 m/s²)
                                                           │
                                                           ▼
                                           Smoothed 2D Vehicle Trajectory
```

---

## Repurposing NVIDIA NeMo

The sequential core of this model is built upon **NVIDIA NeMo** (`nemo.collections.asr.modules.conformer_encoder.ConformerEncoder`), NVIDIA's open-source conversational AI toolkit. 

### Why NeMo ASR Conformer for Physical Vehicle Tracking?
NVIDIA designed its ASR Conformer for enterprise-grade, low-latency streaming speech recognition. Rather than reinventing streaming transformers from scratch, we repurposed NVIDIA's battle-tested streaming engine for continuous physical vehicle tracking by introducing key architectural adaptations:

| Standard NeMo Speech ASR | Remodeled Vehicle Motion Conformer |
| :--- | :--- |
| **Input Domain:** Mel-scale filterbanks (80–128 bins) optimized for human vocal range. | **1,025-bin Linear STFT Spectrograms** capturing high-frequency tire friction and Doppler shifts up to 11 kHz. |
| **Subsampling:** 4x or 8x strided convolutions over time, shrinking temporal resolution. | **Subsampling Disabled (`subsampling=None`, factor 1)** to maintain strict 1:1 synchronization with physical 10 Hz motion timestamps. |
| **Conditioning:** No spatial inputs; unconditioned speech acoustic frames. | **2D Spatial Conditioning:** Fixed starting coordinate $(p_{x0}, p_{y0})$ is normalized and concatenated to every frame. |
| **Output Task:** CTC / RNN-T loss predicting discrete text character tokens. | **Multi-Task Categorical Kinematics:** 4 parallel heads predicting probability distributions over physical bins ($v_x, v_y, p_x, p_y$). |
| **Downstream Decoder:** Language models (n-gram / neural LM beam search). | **Kinematic Physics Decoder:** Beam search constrained by vehicle acceleration ($a \le 8\text{ m/s}^2$) + RTS Kalman Smoother. |

---

## Architecture Deep Dive

### 1. Acoustic Preprocessing & Spatial Conditioning
* **Audio Format:** Monophonic 22,050 Hz sampling rate.
* **Short-Time Fourier Transform (STFT):** Window/FFT size of 2048, hop size of 256 samples, centered STFT. Each time step yields 1,025 linear frequency bins ($N_{\text{fft}} / 2 + 1$).
* **Features:** Log-magnitude compression: $\mathbf{x} = \log(1 + |X|)$.
* **Starting Position Conditioning:** A single microphone cannot deduce absolute Cartesian distance from sound alone due to acoustic symmetry. To resolve this, the known initial coordinate $p_0 = (p_{x0}, p_{y0})$ (scaled by 1/100) is tiled across time and concatenated with the spectral features, anchoring the coordinate system.

### 2. Frequency CNN Frontend
Located in `model.py` (`FrequencyCNN`), this module replaces standard linear projections with a custom 1D convolutional network acting along the **frequency axis**:

```python
class FrequencyCNN(nn.Module):
    # Convolves across the 1,025 frequency bins per time frame
    self.convs = nn.Sequential(
        nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),   # Layer 1: local spectral peaks
        nn.GELU(),
        nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),  # Layer 2: harmonic overtone spacing
        nn.GELU(),
    )
    self.projection = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(16 * reduced_freq, 128),
        nn.LayerNorm(128),
        nn.Dropout(0.1),
    )
```

* **Doppler Shift Invariance:** As a vehicle approaches and recedes, engine firing harmonics and tire peaks slide across frequency bins. The 1D CNN slides across the frequency spectrum, detecting the invariant acoustic harmonic pattern regardless of its shifted position.
* **Compression:** Reduces 1,025 raw FFT bins into 128 compact acoustic embedding features per 0.1s frame.
* **Combined Channel Vector:** 128 acoustic features + 2 spatial features = **130 input channels** fed into the Conformer.

### 3. NVIDIA NeMo Conformer Encoder
* **Module:** `nemo.collections.asr.modules.conformer_encoder.ConformerEncoder`
* **Configuration:**
  * **Layers (`n_layers`):** 4 Conformer blocks
  * **Hidden Dimension (`d_model`):** 128
  * **Attention Heads (`n_heads`):** 4
  * **Attention Mechanism:** Multi-Head Self-Attention with Relative Positional Embeddings (`rel_pos`).
  * **Strict Causal Context (`att_context_size`):** `[1024, 0]`. Exactly 1024 past frames (~11.9 seconds of acoustic history) and **0 right-context frames**, guaranteeing zero future lookahead.
  * **Causal Depthwise Convolution (`conv_kernel_size`):** Kernel size 15, `conv_context_size="causal"`, `conv_norm_type="layer_norm"`.
  * **Subsampling:** None (1:1 frame preservation).
  * **STFT-to-10 Hz Alignment:** With subsampling disabled, the Conformer operates at the native ~86.13 Hz STFT frame rate (22,050 Hz / 256 hop), and 10 Hz predictions are formed by extracting the hidden representation of the latest completed STFT window available at each 0.1s timestamp (`logits = [z[0, ix]]` during training, or chunk-final slicing `z[0, -1]` during streaming).

#### Anatomy of a Conformer Block Inside NVIDIA NeMo
Each of the 4 layers contains a Macaron-style sandwich structure:
1. **Feed-Forward Module 1 (Half-step):** Pre-norm LayerNorm, Linear(128 $\rightarrow$ 512), Swish/SiLU, Dropout, Linear(512 $\rightarrow$ 128), residual connection ($\times 0.5$).
2. **Multi-Head Self-Attention Module:** Pre-norm LayerNorm, RelPosition Multi-Head Attention (4 heads, causal mask `[1024, 0]`), Dropout, residual connection.
3. **Convolution Module:** Pre-norm LayerNorm, Pointwise Conv, Gated Linear Unit (GLU), Depthwise Causal Conv1D (kernel 15), BatchNorm/LayerNorm, Swish, Pointwise Conv, Dropout, residual connection.
4. **Feed-Forward Module 2 (Half-step):** Pre-norm LayerNorm, Linear(128 $\rightarrow$ 512), Swish, Dropout, Linear(512 $\rightarrow$ 128), residual connection ($\times 0.5$).
5. **Final Block LayerNorm.**

#### NVIDIA NeMo Streaming Cache Engine
During streaming execution (`predict_chunk`), the Conformer avoids reprocessing past audio. It maintains internal state buffers passed forward at each 0.1s interval:
* `cache_last_channel`: Caches the activations for the attention key/value history across the 1024 past frames.
* `cache_last_time`: Caches the sliding convolution receptive field (the 14 previous samples required for kernel-15 causal convolution).
* `cache_last_channel_len`: Tracks valid cache sequence lengths.

### 4. Four Categorical Prediction Heads
The shared 128-dim representations from the Conformer feed four parallel classification heads (`nn.Linear(128, len(b))`):
* **$v_x$ (Lateral Velocity):** $[-120, +120]$ km/h, step 2 km/h (121 bins)
* **$v_y$ (Longitudinal Velocity):** $[-120, +120]$ km/h, step 2 km/h (121 bins)
* **$p_x$ (Lateral Position):** $[-200, +200]$ meters, step 2 m (201 bins)
* **$p_y$ (Longitudinal Position):** $[-100, +100]$ meters, step 2 m (101 bins)

**Why Categorical Classification over Direct Regression?**
Acoustic pass-bys suffer from multi-modal uncertainty (e.g., symmetry ambiguity where approaching from the front-left can produce similar Doppler pitches as front-right). A standard regression model (MSE loss) collapses to the arithmetic mean of multiple modes, predicting physically impossible paths down the middle. Categorical heads preserve the full multi-modal probability distributions $P(s)$, allowing the downstream kinematic decoder to pick the globally consistent trajectory.

### 5. Kinematic Beam Search & RTS Kalman Decoder
Located in `decoder.py` (`MotionDecoder`), this module turns raw probabilities into continuous, physically valid vehicle paths.

1. **Continuous Expectation Proposals:** Instead of picking discrete bin centers (which would cause 2-meter staircasing artifacts), position and velocity candidates are proposed using continuous expectations:
   $$\mu = \sum_{i} b_i \cdot P(b_i)$$
   Blended convex proposals combine expected observations with trapezoidal velocity integration:
   $$p_{\text{int}} = p_{t-1} + \frac{v_{t-1} + v_t}{2} \cdot \Delta t$$
2. **Path Scoring:** Paths are scored using a combination of acoustic probability and physical feasibility:
   $$\text{Score}_t = \text{Score}_{t-1} + \sum_{j \in \{vx,vy,px,py\}} \ln P(s_j) - w_m \cdot \text{Huber}(p - p_{\text{int}}) - w_a \cdot \text{Huber}(\text{excess } a)$$
   Candidates exceeding maximum speed ($120$ km/h) or physical acceleration limits ($a > 8\text{ m/s}^2$) are pruned.
3. **Diverse Beam Pruning:** Retains the top $K=32$ candidate trajectory histories.
4. **Rauch-Tung-Striebel (RTS) Kalman Smoothing:** Once the highest-scoring beam path is finalized, a backward Kalman smoother passes over the trajectory using a double-integrator kinematic model, eliminating discrete steps and enforcing continuous vehicle dynamics.

---

## Latest Benchmark Metrics

### Independent 1,000-Clip Blind Test (`vs13_path2d_1000_varied`)
Evaluated using the latest checkpoint (`runs/vs13_7500_varied_improved/model.pt`):

| Evaluation Metric | Score | Unit | Description |
| :--- | :---: | :---: | :--- |
| **Final Path ADE (Primary)** | **2.806** | meters | Mean Average Displacement Error of smoothed path |
| **Median Clip ADE** | **2.422** | meters | Median ADE across all 1,000 test clips |
| **Online Streaming ADE** | **2.821** | meters | Real-time causal state estimate error at each step |
| **Final Displacement Error (FDE)** | **7.623** | meters | Positional error at trajectory termination |
| **X MAE ($p_x$)** | **1.525** | meters | Lateral position mean absolute error |
| **Y MAE ($p_y$)** | **1.988** | meters | Longitudinal position mean absolute error |

### All Runs — Blind Test Results

| Run | Dataset | Training Changes | Best Val Loss | ADE ↓ | Median ADE ↓ | Online ADE ↓ | FDE ↓ | X MAE ↓ | Y MAE ↓ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vs13_7500_varied_improved`** 🏆 | 7,500 clips | + Soft labels, kinematic loss, SpecAugment | 2.1622† | **2.806 m** | **2.422 m** | **2.821 m** | 7.623 m | 1.525 m | **1.988 m** |
| `vs13_7500_varied_cnn` | 7,500 clips | + Frequency CNN frontend | **1.8173** | 2.965 m | 2.612 m | 3.013 m | **7.333 m** | **1.385 m** | 2.301 m |
| `vs13_7500_varied_v1` | 7,500 clips | Baseline (linear projection) | 1.9416 | 3.043 m | — | 3.093 m | 7.748 m | — | — |
| `smoke_1000` | 1,000 clips | + Soft labels, kinematic loss, SpecAugment | 2.2886† | 4.365 m | 3.775 m | 4.567 m | 9.947 m | 2.679 m | 2.840 m |
| `vs13_1000_option_a_cnn` | 1,000 clips | + Frequency CNN frontend | 2.1404 | 4.158 m | — | 4.194 m | 10.070 m | — | — |
| `vs13_1000_option_a` | 1,000 clips | Baseline (linear projection) | 2.1588 | 5.047 m | — | 5.102 m | 11.009 m | — | — |
| Original baseline | 1,000 clips | — | 2.8428 | — | — | — | — | — | — |

> † Val loss not directly comparable to earlier runs — Gaussian soft-label CE + kinematic loss terms inflate the absolute loss value.

---

## Repository Structure

```
vehicle_motion_conformer/
├── config.py             # Hyperparameters, bin definitions, feature settings
├── model.py              # FrequencyCNN, ConformerEncoder wrapper, 4 heads
├── decoder.py            # MotionDecoder, kinematic beam search, RTS smoother
├── data.py               # Audio loading, STFT extraction, clip batching
├── train.py              # Training loop with validation grouping and checkpointing
├── predict.py            # Single-clip streaming prediction script
├── plot.py               # Trajectory and time-series visualization
├── external_test.py      # Independent dataset evaluation script
├── test_split.py         # Partitioned train/val/test evaluation
├── redecode.py           # Offline re-decoding of saved probability archives
├── smoke_check.py        # Diagnostic verification of causal equivalence & caches
├── runs/                 # Training runs, checkpoints, and evaluation metrics
└── requirements.txt      # Python dependencies
```

---

## Installation & Setup

### 1. Requirements
* macOS (Metal/MPS supported) or Linux (CUDA supported)
* Python 3.10+
* PyTorch 2.0+
* NVIDIA NeMo toolkit

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify Causal Equivalence & Cache Integrity
Run the built-in diagnostic test to verify that chunked streaming execution exactly matches full-sequence causal execution:
```bash
./venv/bin/python smoke_check.py
```

---

## Usage & Workflows

### Training
To train a model on a dataset grouped by source clips:
```bash
mkdir -p runs/my_experiment
cd runs/my_experiment

../../venv/bin/python ../../train.py \
  --data "/path/to/dataset_directory" \
  --epochs 30 \
  --checkpoint model.pt
```
`train.py` logs training progress to `history.json` and automatically saves the checkpoint achieving the lowest validation loss.

### Single-Clip Streaming Inference
To stream a single clip and generate output probability archives:
```bash
./venv/bin/python predict.py /path/to/sample_dir \
  --checkpoint runs/vs13_7500_varied_cnn/model.pt \
  --output prediction.npz \
  --save-revisions

# Visualize the resulting trajectory
./venv/bin/python plot.py prediction.npz \
  --sample /path/to/sample_dir \
  --output trajectory_timeseries.png \
  --xy-output trajectory_2d.png
```

### Independent Blind Evaluation
To evaluate an entire dataset of unseen test clips and compute comprehensive ADE/FDE metrics:
```bash
./venv/bin/python external_test.py \
  --checkpoint runs/vs13_7500_varied_cnn/model.pt \
  --input /path/to/blind_test_dataset \
  --output runs/vs13_7500_varied_cnn/my_evaluation
```
This produces:
* `metrics.json`: Detailed per-clip errors.
* `summary.json`: Mean/median ADE, FDE, and coordinate MAEs.
* `.npz` and `.png` trajectory plots for every processed sample.

### Offline Re-Decoding & Parameter Tuning
Because raw network probabilities are saved in `.npz` files, you can experiment with decoder weights ($w_{\text{motion}}$, $w_{\text{accel}}$, or RTS smoothing) instantly without re-running neural network inference:
```bash
./venv/bin/python redecode.py prediction.npz \
  --output redecoded_prediction.npz \
  --accel-weight 0.1 \
  --motion-weight 1.5 \
  --smooth
```

---

## Output Archive Format (`.npz`)

Predictions saved by `predict.py` or `external_test.py` contain:

| Key | Shape / Type | Description |
| :--- | :--- | :--- |
| `seconds` | `[T]` | Output timestamps (0.1s, 0.2s, ..., T) |
| `initial_position` | `[2]` | Ground-truth starting coordinate $(p_{x0}, p_{y0})$ |
| `vx`, `vy`, `px`, `py` | `[T, Bins]` | Categorical probability distributions from the 4 heads |
| `vx_bins`, `vy_bins`, ... | `[Bins]` | Physical values corresponding to each bin center |
| `raw_argmax` | `[T, 4]` | Unconstrained highest-probability bin values |
| `raw_expectation` | `[T, 4]` | Unconstrained expected values $\sum b_i P(b_i)$ |
| `online_estimates` | `[T, 4]` | Causal, real-time state emitted at each 0.1s streaming step |
| `path_seconds` | `[T + 1]` | Timestamps of the revised path (including $t=0$) |
| `revised_path` | `[T + 1, 4]` | Final kinematically smoothed trajectory $[v_x, v_y, p_x, p_y]$ |
| `final_path_ade_m` | `float` | Mean Average Displacement Error against ground truth |
| `online_ade_m` | `float` | Mean Online Displacement Error of streaming estimates |
| `fde_m` | `float` | Final Displacement Error at trajectory end |
