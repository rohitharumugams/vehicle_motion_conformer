import json
from pathlib import Path
import numpy as np
from config import BINS, DATA, STATE, SEED, feature_config


def samples(root=DATA):
    return sorted(p.parent.parent for p in Path(root).rglob('spectrograms/stft.npy')
                  if '__MACOSX' not in p.parts)


def load_audio(path, settings=None):
    """Features available at each 10 Hz wall-clock timestamp; no future frames."""
    p = Path(path)
    cfg = feature_config() if settings is None else settings
    meta = json.loads((p / 'metadata/simulation_parameters.json').read_text())['phase1']
    for key, value in [('spec_sr_hz', cfg['spec_sr']), ('n_fft', cfg['n_fft']),
                       ('hop_length', cfg['hop'])]:
        if meta[key] != value:
            raise ValueError(f'{p}: {key}={meta[key]}, expected {value}')
    t = np.load(p / 'metadata/frame_times.npy').astype(np.float64)
    a = np.load(p / 'spectrograms/stft.npy')
    if a.shape != (cfg['n_fft'] // 2 + 1, len(t)):
        raise ValueError(f'{p}: unexpected STFT shape {a.shape}')
    if len(t) < 2 or np.any(np.diff(t) <= 0) or not np.isclose(t[0], 0):
        raise ValueError('Frame times must start at zero and increase')
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Expected finite nonnegative magnitude STFT, not complex/log features')
    half = cfg['n_fft'] / (2 * cfg['spec_sr'])
    keep = t >= half - 1e-8 if cfg['centered'] else np.ones(len(t), dtype=bool)
    available = t[keep] + (half if cfg['centered'] else cfg['n_fft'] / cfg['spec_sr'])
    seconds = np.arange(1, int(np.floor(t[-1] * cfg['output_hz'] + 1e-6)) + 1,
                        dtype=np.float64) / cfg['output_hz']
    ix = np.searchsorted(available, seconds + 1e-8, side='right') - 1
    if not len(seconds) or np.any(ix < 0):
        raise ValueError('Clip has no full acoustic window before its first output timestamp')
    x = np.log1p(a.T[keep][:ix[-1] + 1]).astype(np.float32)
    return x, ix.astype(np.int64), seconds


def initial_position(path, state_file=STATE):
    s = np.load(Path(path) / 'metadata' / state_file, mmap_mode='r')
    if s.ndim != 2 or s.shape[1] != 4:
        raise ValueError('State columns must be [px, vx, py, vy]')
    p0 = s[0, [0, 2]].astype(np.float32)
    if not np.isfinite(p0).all():
        raise ValueError('Initial position must be finite')
    return p0


def spec_augment(x, freq_mask_param=24, time_mask_param=16, n_freq_masks=2, n_time_masks=2, rng=None):
    """SpecAugment on log-STFT features [T, F] without modifying original array in-place."""
    if rng is None:
        rng = np.random.default_rng()
    x_aug = x.copy()
    T, F = x_aug.shape
    for _ in range(n_freq_masks):
        f = rng.integers(0, freq_mask_param + 1)
        if f > 0 and F > f:
            f0 = rng.integers(0, F - f)
            x_aug[:, f0:f0 + f] = 0.0
    for _ in range(n_time_masks):
        t = rng.integers(0, time_mask_param + 1)
        if t > 0 and T > t:
            t0 = rng.integers(0, T - t)
            x_aug[t0:t0 + t, :] = 0.0
    return x_aug


def add_acoustic_noise(x, noise_level=0.02, rng=None):
    """Add small random acoustic background noise to log-STFT features."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, noise_level, size=x.shape).astype(np.float32)
    return np.maximum(x + noise, 0.0)


def load(path, bins=BINS, augment=False, rng=None):
    p = Path(path)
    x, ix, seconds = load_audio(p)
    if augment:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < 0.8:
            x = spec_augment(x, rng=rng)
        if rng.random() < 0.5:
            x = add_acoustic_noise(x, rng=rng)
    t = np.load(p / 'metadata/frame_times.npy')
    s = np.load(p / 'metadata' / STATE)
    if s.shape != (len(t), 4) or not np.isfinite(s).all():
        raise ValueError(f'{p}: expected finite state array [frames, 4]')
    p0 = initial_position(p)
    values = np.stack([np.interp(seconds, t, s[:, j]) for j in (1, 3, 0, 2)], 1)
    values[:, :2] *= 3.6
    for j, b in enumerate(bins):
        if np.any((values[:, j] < b[0]) | (values[:, j] > b[-1])):
            raise ValueError(f'{p}: head {j} target outside bins; widen BINS in config.py')
    y = np.stack([np.abs(values[:, j, None] - b).argmin(1)
                  for j, b in enumerate(bins)], 1).astype(np.int64)
    return x, y, ix, seconds, p0


def split(paths):
    groups = {}
    for p in paths:
        meta = json.loads((p / 'metadata/simulation_parameters.json').read_text())
        groups.setdefault(meta['source_clip'], []).append(p)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError('Need at least two distinct source_clip recordings for a held-out split')
    np.random.default_rng(SEED).shuffle(keys)
    n = max(1, int(0.2 * len(keys)))
    return ([p for key in keys[n:] for p in groups[key]],
            [p for key in keys[:n] for p in groups[key]])


def split3(paths, n_test_groups=2, n_val_groups=1):
    """Three-way train/val/test split grouped by source_clip to prevent leakage.

    Groups are sorted then shuffled with the global SEED for reproducibility.
    The first n_test_groups become the locked blind test set, the next
    n_val_groups become the validation set, and all remaining groups are
    used for training.  Call once; write the result to split.json and
    never recompute from a different dataset root or seed.
    """
    groups = {}
    for p in paths:
        meta = json.loads((p / 'metadata/simulation_parameters.json').read_text())
        groups.setdefault(meta['source_clip'], []).append(p)
    keys = sorted(groups)
    n_min = n_test_groups + n_val_groups + 1
    if len(keys) < n_min:
        raise ValueError(
            f'Need at least {n_min} distinct source_clip groups '
            f'(have {len(keys)}); reduce n_test_groups or n_val_groups')
    np.random.default_rng(SEED).shuffle(keys)
    test_keys  = keys[:n_test_groups]
    val_keys   = keys[n_test_groups:n_test_groups + n_val_groups]
    train_keys = keys[n_test_groups + n_val_groups:]
    train = [p for key in train_keys for p in groups[key]]
    val   = [p for key in val_keys   for p in groups[key]]
    test  = [p for key in test_keys  for p in groups[key]]
    return train, val, test
