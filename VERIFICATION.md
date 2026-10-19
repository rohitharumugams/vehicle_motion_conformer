# Verification

Checked on CPU with Python 3.12, PyTorch 2.14.1+cpu and NeMo Toolkit 3.0.0.
The actual NeMo encoder was executed; these were not mocked encoder tests.

Passed:

- Python syntax checks for every Python file.
- Loading the supplied sample: 862 usable acoustic frames, 101 outputs at 10 Hz,
  first position [-75, 22.579128] metres.
- Changing future input frames does not change earlier encoder/head outputs.
- Cached chunk-by-chunk outputs match a full causal forward pass in evaluation
  mode, including a sequence longer than the test attention-cache capacity.
- Finite forward/backward loss, gradients on all four heads and the shared encoder.
  Optional nonzero physics losses were included in the gradient test.
- Beam decoding rejects physically impossible jumps, obeys vector acceleration
  limits, integrates velocity consistently and preserves the exact first position.
- Default-size encoder: 1,798,688 trainable parameters.
- CLI training, validation and checkpoint saving on two short plumbing fixtures
  made from the supplied sample. Their source names were changed ONLY to exercise
  the split code; they are duplicates and do not constitute a held-out evaluation.
- CLI checkpoint loading, streaming replay of the full supplied sample, normalized
  four-head distributions, saved exact anchor and JSONL revision histories.
- Plot generation with a ground-truth overlay.

No full-dataset training, GPU throughput test, real microphone acquisition or
real-world accuracy evaluation was performed. Test checkpoints and test data are
not included in this package. Train a new checkpoint on your own full dataset.

The supplied labels include discontinuous velocity changes exceeding the default
8 m/s^2 constraint. Physics training penalties are disabled by default pending
label/timing validation. The decoder still enforces its configurable constraints;
its path therefore may differ from those abrupt labels even with strong emissions.
