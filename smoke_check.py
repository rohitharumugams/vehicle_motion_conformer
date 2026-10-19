"""Run BEFORE training: real NeMo forward/backward, causality, cache equivalence."""
import argparse
import numpy as np
import torch
from config import BINS, model_config
from decoder import MotionDecoder
from loss import motion_loss
from model import MotionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', help='Also validate a simulator sample directory')
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    c = model_config()
    c.update(hidden=32, layers=2, left_context=64, conv_kernel=7, dropout=0.0)
    model = MotionModel(settings=c).eval()
    x = torch.randn(1, 75, c['feat_in'])
    p0 = torch.tensor([[-75.0, 22.579128]])
    with torch.no_grad():
        full, _ = model(x, p0)
        changed = x.clone()
        changed[:, 40:] += 20 * torch.randn_like(changed[:, 40:])
        other, _ = model(changed, p0)
        for a, b in zip(full, other):
            torch.testing.assert_close(a[:, :40], b[:, :40], atol=1e-5, rtol=1e-4)
        chunks, cache = [[] for _ in range(4)], None
        for start in range(0, x.shape[1], 9):
            out, cache = model.predict_chunk(x[:, start:start + 9], p0, cache)
            for j, z in enumerate(out):
                chunks[j].append(z)
        for a, b in zip(full, chunks):
            torch.testing.assert_close(a, torch.cat(b, 1), atol=3e-5, rtol=3e-4)
        assert cache[0].shape[2] == c['left_context']
        assert cache[2].item() == c['left_context']
    model.train()
    logits, _ = model(x[:, :27], p0)
    selected = [z[0, [8, 17, 26]] for z in logits]
    target = torch.tensor([[60, 60, 62, 61]] * 3)
    loss, terms = motion_loss(selected, target, model.bins, [0.1, 0.2, 0.3], p0[0],
                              kinematic_weight=0.05, acceleration_weight=0.01)
    loss.backward()
    assert torch.isfinite(loss)
    for head in model.heads:
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert model.encoder.pre_encode.weight.grad is not None
    # A high-probability impossible jump must not override acceleration limits.
    decoder = MotionDecoder(p0[0].numpy(), BINS)
    for n in range(20):
        probs = [np.full(len(b), 1e-8) for b in BINS]
        probs[0][np.abs(BINS[0] - (2 if n == 0 else 100)).argmin()] = 1
        probs[1][np.abs(BINS[1]).argmin()] = 1
        probs[2][:] = 1
        probs[3][:] = 1
        decoder.step(probs, (n + 1) / 10)
    t, path = decoder.best_path()
    np.testing.assert_array_equal(path[0, 2:], p0[0].numpy())
    a = np.linalg.norm(np.diff(path[1:, :2], axis=0) / 3.6, axis=1) / np.diff(t[1:])
    assert np.max(a) <= decoder.max_accel + 1e-5
    expected_delta = (path[2:, :2] + path[1:-1, :2]) / 7.2 * np.diff(t[1:])[:, None]
    np.testing.assert_allclose(np.diff(path[1:, 2:], axis=0), expected_delta, atol=1e-8)
    if args.sample:
        from data import load
        features, y, ix, seconds, start = load(args.sample)
        assert features.shape[1] == c['feat_in']
        assert len(y) == len(ix) == len(seconds) and ix[-1] < len(features)
        print(f'Sample OK: {len(features)} features, {len(seconds)} outputs, p0={start}')
    print('PASS: causal outputs, streaming/full equivalence beyond cache capacity, '
          'four-head gradients, position anchoring, and motion constraints')


if __name__ == '__main__':
    main()
