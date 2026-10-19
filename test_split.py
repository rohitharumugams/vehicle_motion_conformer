"""Run a checkpoint on every clip in a saved train/validation split."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from tqdm import tqdm

from config import NAMES
from data import initial_position, load_audio
from decoder import MotionDecoder
from model import MotionModel
from plot import plot_prediction, plot_xy_prediction


def predict_one(model, checkpoint, sample, device):
    bins = [np.asarray(b, dtype=np.float32) for b in checkpoint['bins']]
    x, ix, seconds = load_audio(sample, checkpoint['feature_config'])
    p0 = initial_position(sample, checkpoint['state_file'])
    position_input = torch.from_numpy(p0)[None].to(device)
    decoder = MotionDecoder(p0, bins)
    probabilities = {name: [] for name in NAMES}
    online, raw, means = [], [], []
    cache, start, last_probs = None, 0, None

    for second, end in zip(seconds, ix):
        if end >= start:
            chunk = torch.from_numpy(x[start:end + 1])[None].to(device)
            logits, cache = model.predict_chunk(chunk, position_input, cache)
            last_probs = [z[0, -1].softmax(-1).cpu().numpy() for z in logits]
            start = int(end) + 1
        if last_probs is None:
            raise RuntimeError('No acoustic prediction available yet')
        for name, probability in zip(NAMES, last_probs):
            probabilities[name].append(probability.copy())
        raw.append([b[p.argmax()] for b, p in zip(bins, last_probs)])
        means.append([float(b @ p) for b, p in zip(bins, last_probs)])
        online.append(decoder.step(last_probs, second))

    path_seconds, revised_path = decoder.best_path()
    return dict(
        seconds=seconds,
        initial_position=p0,
        online_estimates=np.asarray(online),
        raw_argmax=np.asarray(raw),
        raw_expectation=np.asarray(means),
        path_seconds=path_seconds,
        revised_path=revised_path,
        names=np.asarray(NAMES),
        **{name: np.stack(values) for name, values in probabilities.items()},
        **{name + '_bins': b for name, b in zip(NAMES, bins)},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/vs13_1000_varied',
                        help='Directory containing model.pt and split.json')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--output', help='Output directory (default: RUN/SPLIT_predictions)')
    parser.add_argument('--limit', type=int, help='Only process the first N clips')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    run = Path(args.run).expanduser().resolve()
    output = (Path(args.output).expanduser().resolve() if args.output
              else run / f'{args.split}_predictions')
    checkpoint_path = run / 'model.pt'
    split_path = run / 'split.json'
    if not checkpoint_path.is_file() or not split_path.is_file():
        parser.error(f'{run} must contain model.pt and split.json')

    paths = json.loads(split_path.read_text())[args.split]
    if args.limit is not None:
        paths = paths[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    xy_output = output / 'xy'
    xy_output.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'nemo_conformer_motion_v1':
        raise ValueError('Expected a nemo_conformer_motion_v1 checkpoint')
    model = MotionModel(checkpoint['bins'], checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    completed, failed = [], []
    for sample_name in tqdm(paths, desc=f'testing {args.split}', unit='clip'):
        sample = Path(sample_name)
        stem = sample.name
        prediction_path = output / f'{stem}.npz'
        image_path = output / f'{stem}.png'
        xy_image_path = xy_output / f'{stem}.png'
        try:
            result = predict_one(model, checkpoint, sample, device)
            np.savez(prediction_path, **result)
            plot_prediction(prediction_path, image_path, sample)
            plot_xy_prediction(prediction_path, xy_image_path, sample)
            completed.append(str(sample))
        except Exception as error:
            failed.append({'sample': str(sample), 'error': str(error)})

    summary = {
        'checkpoint': str(checkpoint_path),
        'split': args.split,
        'device': device,
        'requested': len(paths),
        'completed': len(completed),
        'failed': failed,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Completed {len(completed)}/{len(paths)} clips; results: {output}')
    if failed:
        raise SystemExit(f'{len(failed)} clips failed; see {output / "summary.json"}')


if __name__ == '__main__':
    main()
