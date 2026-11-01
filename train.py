import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import config as C
from data import load, samples, split3
from loss import motion_loss
from model import MotionModel


def prefetch_loader(path_list, max_prefetch=4, augment=False):
    """Prefetch samples in a background thread so disk I/O overlaps with GPU compute."""
    from queue import Queue
    from threading import Thread

    q = Queue(maxsize=max_prefetch)
    sentinel = object()

    def worker():
        for p in path_list:
            try:
                data = load(p, augment=augment)
                q.put((p, data, None))
            except Exception as e:
                q.put((p, None, e))
        q.put(sentinel)

    t = Thread(target=worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        p, data, err = item
        if err is not None:
            raise err
        yield p, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=C.DATA)
    parser.add_argument('--epochs', type=int, default=C.EPOCHS)
    parser.add_argument('--checkpoint', default='model.pt')
    parser.add_argument('--split', default='3way',
                        help='Path to existing split.json, or "2way" or "3way"')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    device = ('cuda' if torch.cuda.is_available()
              else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.split).is_file():
        partition = json.loads(Path(args.split).read_text())
        train = [Path(p) for p in partition['train']]
        val = [Path(p) for p in partition['val']]
    elif args.split == '2way':
        from data import split as split2
        train, val = split2(samples(args.data))
        partition = {'train': list(map(str, train)), 'val': list(map(str, val))}
    else:
        train, val, test = split3(samples(args.data))
        partition = {'train': list(map(str, train)), 'val': list(map(str, val)),
                     'test': list(map(str, test))}
    (run_dir / 'split.json').write_text(json.dumps(partition, indent=2))
    model = MotionModel().to(device)
    print(f'{device}: {sum(p.numel() for p in model.parameters()):,} parameters; '
          f'{len(train)} training clips, {len(val)} validation clips')
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    best = float('inf')
    history = []
    for epoch in range(args.epochs):
        record = {'epoch': epoch + 1}
        for paths, training in ((train, True), (val, False)):
            model.train(training)
            total, count = 0.0, 0
            heads, mae = np.zeros(4), np.zeros(4)
            if training:
                rng.shuffle(paths)
            with torch.set_grad_enabled(training):
                phase = 'train' if training else 'val'
                for p, (x, y, ix, seconds, p0) in tqdm(
                    prefetch_loader(paths, augment=training), total=len(paths),
                    desc=f'epoch {epoch + 1}/{args.epochs} {phase}',
                    unit='clip', leave=False
                ):
                    x = torch.from_numpy(x)[None].to(device)
                    y = torch.from_numpy(y).to(device)
                    ix = torch.from_numpy(ix).to(device)
                    p0 = torch.from_numpy(p0)[None].to(device)
                    output, _, pos_pred_full = model(x, p0)
                    logits = [z[0, ix] for z in output]
                    # pos_pred at the sampled timesteps; ground-truth px,py in metres
                    pos_pred = pos_pred_full[0, ix]          # [T, 2]
                    gt_position = torch.stack(
                        [model.bins[2][y[:, 2]], model.bins[3][y[:, 3]]], dim=-1
                    )                                        # [T, 2]
                    loss, terms = motion_loss(logits, y, model.bins, seconds, p0[0],
                                             pos_pred=pos_pred, gt_position=gt_position)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss on {p}')
                    if training:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    total += loss.item() * len(y)
                    heads += terms['ce'].detach().cpu().numpy() * len(y)
                    truth = torch.stack([b[y[:, j]] for j, b in enumerate(model.bins)], -1)
                    mae += (terms['means'].detach() - truth).abs().sum(0).cpu().numpy()
                    count += len(y)
            tag = 'train' if training else 'val'
            record[tag] = total / count
            record[tag + '_ce'] = (heads / count).tolist()
            record[tag + '_mae_quantized'] = (mae / count).tolist()
        print(f"{epoch + 1}: train={record['train']:.4f} val={record['val']:.4f} "
              f"val CE={np.round(record['val_ce'], 3)}")
        history.append(record)
        (run_dir / 'history.json').write_text(json.dumps(history, indent=2))
        if record['val'] < best:
            best = record['val']
            checkpoint = dict(model=model.state_dict(), bins=[b.tolist() for b in C.BINS],
                              model_config=model.settings, feature_config=C.feature_config(),
                              names=list(C.NAMES), state_file=C.STATE,
                              architecture='nemo_conformer_motion_v1',
                              epoch=epoch + 1, val_loss=best, split=partition,
                              loss_settings=dict(head_weights=list(C.HEAD_WEIGHTS),
                                                 kinematic_weight=C.KINEMATIC_WEIGHT,
                                                 acceleration_weight=C.ACCELERATION_WEIGHT,
                                                 label_smoothing_sigma=C.LABEL_SMOOTHING_SIGMA,
                                                 position_head_weight=C.POSITION_HEAD_WEIGHT,
                                                 endpoint_weight=C.ENDPOINT_WEIGHT))
            torch.save(checkpoint, ckpt_path)


if __name__ == '__main__':
    main()
