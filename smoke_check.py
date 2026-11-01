"""Run before training: real NeMo forward/backward, causality and cache equivalence."""
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
        full, _, pos_full = model(x, p0)
        changed = x.clone()
        changed[:, 40:] += 20 * torch.randn_like(changed[:, 40:])
        other, _, _ = model(changed, p0)
        for a, b in zip(full, other):
            torch.testing.assert_close(a[:, :40], b[:, :40], atol=1e-5, rtol=1e-4)
        assert pos_full.shape == (1, 75, 2), f"position_head shape wrong: {pos_full.shape}"
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
    logits, _, pos_pred_full = model(x[:, :27], p0)
    selected = [z[0, [8, 17, 26]] for z in logits]
    pos_pred = pos_pred_full[0, [8, 17, 26]]   # [3, 2]
    target = torch.tensor([[60, 60, 62, 61]] * 3)
    gt_position = torch.tensor([[-75.0, 22.0]] * 3)
    loss, terms = motion_loss(selected, target, model.bins, [0.1, 0.2, 0.3], p0[0],
                              pos_pred=pos_pred, gt_position=gt_position,
                              kinematic_weight=0.05, acceleration_weight=0.01)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(terms['pos_loss']) and terms['pos_loss'].item() > 0, \
        "position head loss should be nonzero"
    for head in model.heads:
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert model.position_head[0].weight.grad is not None, \
        "position_head should receive gradients"
    assert model.encoder.pre_encode.weight.grad is not None
    decoder = MotionDecoder(p0[0].numpy(), BINS)
    for n in range(5):
        probs = [np.full(len(b), 1e-8) for b in BINS]
        for j, b in enumerate(BINS):
            value = 2 if j == 0 else (p0[0, j - 2].item() if j >= 2 else 0)
            probs[j][np.abs(b - value).argmin()] = 1
        state = decoder.step(probs, (n + 1) / 10)
        assert np.isfinite(state).all()
    t_smooth, path_smooth = decoder.best_path(adaptive_noise=True)
    assert len(t_smooth) == 6 and np.isfinite(path_smooth).all()

    # Verify expectation + RTS decoder (no beam search)
    from decoder import expectation_rts_path
    exp = np.tile([[10.0, 0.0, -70.0, 22.0]], (5, 1))
    t_e, path_e = expectation_rts_path(p0[0].numpy(), np.arange(1, 6) / 10, exp)
    assert path_e.shape == (6, 4) and np.isfinite(path_e).all()
    assert np.allclose(path_e[0, 2:], p0[0].numpy(), atol=1e-4)

    # Verify SpectroTemporalCNN and FiLM conditioning
    c_film = dict(c, frontend='spectro_temporal_cnn', spatial_conditioning='film')
    m_film = MotionModel(settings=c_film).train()
    out_film, _, _ = m_film(x[:, :15], p0)
    loss_film = sum(z.sum() for z in out_film)
    loss_film.backward()
    assert m_film.film_gen[0].weight.grad is not None

    # Verify Data Augmentations
    from data import spec_augment, add_acoustic_noise
    spec_sample = np.ones((50, c['feat_in']), dtype=np.float32)
    aug1 = spec_augment(spec_sample)
    aug2 = add_acoustic_noise(spec_sample)
    assert aug1.shape == aug2.shape == spec_sample.shape and np.isfinite(aug1).all()

    if args.sample:
        from data import load
        features, y, ix, seconds, start = load(args.sample)
        assert features.shape[1] == c['feat_in']
        assert len(y) == len(ix) == len(seconds) and ix[-1] < len(features)
        print(f'Sample OK: {len(features)} features, {len(seconds)} outputs, p0={start}')
    print('PASS: causal outputs, streaming/full equivalence, four-head gradients, position head, endpoint weighting, 2D CNN, FiLM, augmentations, and adaptive decoder')


if __name__ == '__main__':
    main()
