"""Run a v1 checkpoint on every clip in its saved train/validation split."""
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
import config as C
from decoder import MotionDecoder, expectation_rts_path
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
    if C.DECODER_MODE == 'expectation_rts':
        path_seconds, revised_path = expectation_rts_path(p0, seconds, np.asarray(means))
    else:
        path_seconds, revised_path = decoder.best_path()
    return dict(seconds=seconds, initial_position=p0,
                online_estimates=np.asarray(online), raw_argmax=np.asarray(raw),
                raw_expectation=np.asarray(means), path_seconds=path_seconds,
                revised_path=revised_path, names=np.asarray(NAMES),
                **{name: np.stack(values) for name, values in probabilities.items()},
                **{name + '_bins': b for name, b in zip(NAMES, bins)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--output')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run = Path(args.run).expanduser().resolve()
    output_name = args.split
    output = (Path(args.output).expanduser().resolve() if args.output
              else run / 'outputs' / output_name)
    partition = json.loads((run / 'split.json').read_text())
    paths = partition[args.split]
    if args.limit is not None:
        paths = paths[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'xy').mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available()
              else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    checkpoint = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'nemo_conformer_motion_v1':
        raise ValueError('Expected a nemo_conformer_motion_v1 checkpoint')
    model = MotionModel(checkpoint['bins'], checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    from external_test import trajectory_metrics
    completed, failed, clip_metrics = [], [], []
    for sample_name in tqdm(paths, desc=f'testing {args.split}', unit='clip'):
        sample = Path(sample_name)
        prediction_path = output / f'{sample.name}.npz'
        try:
            result = predict_one(model, checkpoint, sample, device)
            metrics = trajectory_metrics(sample, result, checkpoint['state_file'])
            extra = {} if metrics is None else {
                key: value for key, value in metrics.items() if key != 'timestamps'
            }
            np.savez(prediction_path, **result, **extra)
            plot_prediction(prediction_path, output / f'{sample.name}.png', sample)
            plot_xy_prediction(prediction_path, output / 'xy' / f'{sample.name}.png', sample)
            completed.append(str(sample))
            if metrics is not None:
                clip_metrics.append({'sample': sample.name, **metrics})
        except Exception as error:
            failed.append({'sample': str(sample), 'error': str(error)})
    aggregate = None
    if clip_metrics:
        aggregate = {
            'primary_metric': 'final_path_ade_m',
            'final_path_ade_m': float(np.mean([row['final_path_ade_m'] for row in clip_metrics])),
            'median_clip_final_path_ade_m': float(np.median([row['final_path_ade_m'] for row in clip_metrics])),
            'online_ade_m': float(np.mean([row['online_ade_m'] for row in clip_metrics])),
            'fde_m': float(np.mean([row['fde_m'] for row in clip_metrics])),
            'x_mae_m': float(np.mean([row['x_mae_m'] for row in clip_metrics])),
            'y_mae_m': float(np.mean([row['y_mae_m'] for row in clip_metrics])),
        }
    summary = {'checkpoint': str(run / 'model.pt'), 'split': args.split,
               'device': device, 'requested': len(paths),
               'completed': len(completed), 'failed': failed, 'metrics': aggregate}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'metrics.json').write_text(json.dumps(
        {'summary': aggregate, 'clips': clip_metrics}, indent=2))
    print(f'Completed {len(completed)}/{len(paths)} clips; results: {output}')
    if aggregate is not None and clip_metrics:
        print("\n--- Evaluation Metrics Breakdown ---")
        for key, label in [
            ("final_path_ade_m", "ADE"),
            ("online_ade_m", "Online ADE"),
            ("fde_m", "FDE"),
            ("x_mae_m", "X MAE"),
            ("y_mae_m", "Y MAE"),
        ]:
            v = np.array([x[key] for x in clip_metrics])
            print(f"{label}: mean={v.mean():.3f}, median={np.median(v):.3f}, "
                  f"min={v.min():.3f}, max={v.max():.3f}, "
                  f"p90={np.percentile(v, 90):.3f}, p95={np.percentile(v, 95):.3f}")

        ade = np.array([x["final_path_ade_m"] for x in clip_metrics])
        print("\nADE thresholds:")
        for t in [1, 2, 5, 10, 20]:
            print(f"ADE < {t} m: {(ade < t).mean() * 100:.1f}%")
        print("------------------------------------\n")
    if failed:
        raise SystemExit(f'{len(failed)} clips failed; see {output / "summary.json"}')


if __name__ == '__main__':
    main()
