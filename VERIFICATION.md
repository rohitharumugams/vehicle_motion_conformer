# Verification

The codebase has been verified for:
1. **Uncertainty-Aware Adaptive RTS Kalman Smoothing (`decoder.py`):** Automatically tracks categorical observation variances across time and dynamically scales the measurement covariance $R_k$, preventing tail-end divergence (reducing FDE) during faint receding pass-bys.
2. **Gaussian Soft-Label Ordinal Targets (`loss.py`):** Continuous distance-aware target distributions over physical bins, providing smooth gradients that penalize large errors proportionally rather than treating all non-target bins equally.
3. **Kinematic & Acceleration Penalties (`loss.py` & `config.py`):** Joint optimization of $\|p_t - p_{t-1} - v \Delta t\|$ and acceleration constraints during training to resolve longitudinal ($p_y$) drift.
4. **Acoustic Data Augmentation (`data.py`):** SpecAugment (frequency masking and time masking) and additive acoustic noise to enhance model generalization across varied road environments.
5. **2D Spectro-Temporal CNN & FiLM Spatial Conditioning (`model.py`):** Causal 2D convolutional frontend extracting Doppler frequency slopes ($\frac{df}{dt}$), and feature-wise linear modulation (FiLM) conditioning on the known initial position $(p_{x0}, p_{y0})$.
6. **Backward Compatibility:** All existing checkpoints (`runs/vs13_7500_varied_cnn/model.pt`, etc.) continue to load and execute with 100% numerical and causal fidelity.
7. **Smoke Check:** Verified via `smoke_check.py` covering causal streaming equivalence, cache integrity, multi-head gradient propagation, 2D CNN, FiLM, data augmentations, and the adaptive smoother.
