# Vehicle motion with NeMo's streaming Conformer

Replace the old project code with the files in this folder. Keep your existing
`data/` sample directories. This is a new architecture: retrain it from scratch;
the GRU checkpoint cannot be reused.

The model is:

**Existing log1p magnitude STFT + known first position -> causal NeMo Conformer
-> four categorical heads (vx, vy, px, py) -> constrained beam-search trajectory.**

The NeMo encoder is imported from the installed library. You do not need to
copy or modify NVIDIA's `conformer_encoder.py`. The ASR tokenizer, transducer,
speech loss and transcription script are not used.

## Files

| File | Responsibility |
| --- | --- |
| `config.py` | Bins, encoder size, output timing, loss and decoder settings |
| `data.py` | Load existing simulator features, align labels, group source recordings |
| `model.py` | NeMo encoder, position conditioning, four heads, streaming caches |
| `loss.py` | Four CE losses and optional soft motion-consistency penalties |
| `train.py` | Training, validation, split/history files, best checkpoint |
| `decoder.py` | Beam search with retained histories and motion constraints |
| `predict.py` | Replay streaming chunks and save distributions plus trajectories |
| `plot.py` | Probability heatmaps, trajectory and optional ground-truth overlay |
| `smoke_check.py` | Causality, cache, gradient and decoder checks before training |
| `requirements.txt` | Dependencies |

## Install and run

Use Python 3.12 or newer. A Linux environment with a CUDA GPU is recommended
for training. Install a PyTorch build appropriate for your machine first using
[PyTorch's installation instructions](https://pytorch.org/get-started/locally/).
Then, from this project folder:

```bash
python -m pip install -r requirements.txt
python smoke_check.py --sample data/sample_0000001
python train.py
python predict.py data/sample_0000001
python plot.py probabilities.npz --sample data/sample_0000001
```

For another data root, epoch count or checkpoint path:

```bash
python train.py --data /path/to/data --epochs 30 --checkpoint model.pt
```

Give an explicit known initial position (metres):

```bash
python predict.py data/sample_0000001 -75 22.579128 --save-revisions
```

Without explicit coordinates, prediction reads only the first position from
`metadata/state_frames.npy`. Prediction does not load later motion labels.
Plotting with `--sample` loads ground truth solely for visualization.

The dependency file pins NeMo to 3.0.0. The official
[NeMo Speech installation guide](https://docs.nvidia.com/nemo/speech/nightly/starthere/install.html)
provides source/container alternatives if your environment needs them.

## Existing data format

Place samples under `data/`, e.g. `data/sample_0000001/`. Required files:

- `spectrograms/stft.npy`: finite nonnegative magnitude STFT, shape `[1025, frames]`.
- `metadata/frame_times.npy`: increasing frame timestamps in seconds, starting at zero.
- `metadata/state_frames.npy`: columns **[px, vx, py, vy]**, position metres, velocity m/s.
- `metadata/simulation_parameters.json`: `source_clip` and `phase1` feature settings.

Default feature settings match the supplied sample: 22,050 Hz analysis sample
rate, FFT 2048, hop 256. Data loading verifies these against metadata.
Inputs are `log1p(magnitude)`, not log-mel. No whole-clip normalization is used.

The centered-STFT assumption is inherited from the previous starter. Confirm it
against the simulator exporter. A frame is consumed only after its full right
window edge arrives; frames relying on initial left padding are skipped.
If exported frame timestamps represent left-aligned windows, set
`STFT_CENTERED = False` and ensure those timestamps denote window starts.
No resampling or microphone sample-rate change is performed by this project.

Training and validation are grouped by `source_clip` to avoid sharing source
recordings. At least two distinct source recordings are required. All renderings
of one source stay together. With one source per vehicle, this also holds out
vehicles. It is not an independent real-world test set.

## Encoder and timing

Defaults: 4 Conformer layers, width 128, 4 internal attention heads, causal
convolution of kernel size 15, channel-wise layer normalization.

Attention uses `[1024, 0]` with `regular` masking. The right context is zero.
Subsampling is disabled, preserving feature-to-label alignment and avoiding
NeMo's ASR buffer overlap logic. The encoder's linear pre-encoder maps 1027 input
channels (1025 audio + 2 position coordinates) to the hidden width.

The supplied starting position is divided by 100 and repeated as two conditioning
channels. It is the same first point throughout the clip. Future positions and
velocities are never model inputs. This conditions all four heads; it does not
force their raw distributions to equal the starting coordinate.

Training runs the full feature sequence through a causal attention mask.
Streaming uses nonoverlapping new feature chunks with NeMo's initialized
attention and convolution caches. Caches reset between clips. Attention caches
retain 1024 feature frames per layer, approximately 11.9 seconds at this hop.
This is bounded memory, not guaranteed lossless retention of all past audio.
The decoder separately keeps the histories of its surviving trajectories.

Outputs are scheduled at **0.1, 0.2, 0.3, ... seconds**, ten timestamps per second.
This is the motion output rate; it is not the waveform sampling rate.
At each timestamp only frames available by that time can be used. A centered
frame describes slightly earlier audio, so the model learns to estimate the
current state from that available history. No future chunk is consulted.
Runtime speed depends on hardware and must be measured before deployment.

## Targets and losses

Head order: **vx, vy, px, py**. Signed velocity components are converted to km/h;
positions remain in metres. Defaults:

- vx/vy: -120 to 120 km/h, step 2.
- px: -200 to 200 metres, step 2.
- py: -100 to 100 metres, step 2.

Edit `BINS` before training if necessary. Out-of-range targets raise an error
rather than being silently clipped. Labels are interpolated to output times,
then assigned to the nearest bin.

Each head has its own trained weights and CE loss. The shared encoder receives
gradients from their weighted mean. Optional additional terms compare expected
positions with integrated expected velocities and penalize excessive expected
vector acceleration. They are soft penalties, not hard guarantees. Means can
hide multimodal uncertainty, so the four full distributions are saved too.

Both physics weights default to zero: a plain four-CE baseline. The supplied
sample's interpolated velocity labels jump from zero at 0.2 s to approximately
52 km/h at 0.3 s, and have an abrupt change near the end. These exceed the
decoder's default acceleration limit. Check simulator boundary labels and
position/velocity timing before enabling physics regularization. A physically
constrained path cannot exactly reproduce such label discontinuities.
Set positive weights to enable those penalties after verifying the labels.
Set `HEAD_WEIGHTS` to change relative supervision strength.
No standard CTC or transducer loss is used: targets already have timestamps.

`history.json` records separate CE losses, combined losses and diagnostic MAE
against quantized target centres. These MAEs are not errors against the original
continuous labels. The best combined validation-loss checkpoint is `model.pt`.
Encoder and feature settings are stored in that checkpoint for inference.

## Probability graph and constrained path

The four heads produce a probability for every configured bin at every output
timestamp. Beam search retains multiple joint velocity/position trajectories,
adds scores from all four heads, and selects a feasible best path as audio arrives.

Positions on each candidate path start exactly at the known p(0), then integrate
velocity using the trapezoidal rule. Position-head distributions score those
continuous positions by interpolation. Their peaks do not directly overwrite
the integrated path. Initial velocity is unknown, so the first interval uses its
first candidate velocity without assuming zero velocity at t=0.

After the first interval, candidates obey maximum vector acceleration and speed.
Defaults are 8 m/s^2 and 120 km/h. Tune these limits to the simulator; overly tight
limits can exclude the true motion. Velocity bin spacing also limits which
changes can be represented at 10 Hz. Position bounds come from the position bins.

This is approximate beam search with candidate pruning, not exhaustive Viterbi
or a calibrated joint posterior. A better surviving history can replace the
previously preferred history. Histories already pruned cannot be recovered.
The printed current estimates are provisional; later evidence may revise their
past path. Raw network probabilities are not themselves recomputed for past frames.

## Saved predictions

`probabilities.npz` contains:

| Key | Content |
| --- | --- |
| `seconds` | Network output timestamps starting at 0.1 s |
| `vx`, `vy`, `px`, `py` | Four `[timestamps, bins]` probability arrays |
| `vx_bins`, etc. | Bin centres, in each head's units |
| `initial_position` | Exact known initial coordinates |
| `raw_argmax` | Independent most likely bins, `[timestamps, 4]` |
| `raw_expectation` | Independent head expectations, `[timestamps, 4]` |
| `online_estimates` | Best current state emitted at each streaming step |
| `path_seconds` | Final revised path timestamps, including time zero |
| `revised_path` | Final best history, `[timestamps + 1, 4]` |

The first row of `revised_path` is **[NaN, NaN, px0, py0]**. Velocities at time zero
are unknown; the supplied first position is exact. Network distributions begin
at 0.1 seconds and need not equal the position at time zero.

`--save-revisions` additionally saves the best complete history after each step
in `<output>.revisions.jsonl`, allowing you to inspect when older estimates change.

`predict.py` replays exported feature files as streaming input; it is not a
microphone recorder. For actual live acquisition, feed compatible newly available
`log1p` feature chunks to `MotionModel.predict_chunk(x, p0, cache)`, carry its returned
cache, and pass the resulting distributions to `MotionDecoder.step(probs, time)`.
Your acquisition frontend must use the same STFT conventions and avoid future data.

A single mono microphone may not identify every x/y trajectory uniquely. Accuracy
must be evaluated on held-out trajectories and recordings; architecture alone
cannot remove acoustic ambiguities or simulation-to-real differences.

## Checks

See `VERIFICATION.md` for the authoring checks and their limits. Run
`smoke_check.py` in your own installation before training, particularly after
changing NeMo, context settings, or convolution normalization.
