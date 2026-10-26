"""Evaluate a trained v1 checkpoint on an independent external dataset."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from tqdm import tqdm

from data import samples
from model import MotionModel
from plot import plot_prediction, plot_xy_prediction
from test_split import predict_one


def trajectory_metrics(sample, result, state_file):
    """Compare saved causal/final paths with timestamp-aligned ground truth."""
    metadata = Path(sample) / 'metadata'
    times_path = metadata / 'frame_times.npy'
    state_path = metadata / state_file
    if not times_path.is_file() or not state_path.is_file():
        return None
    frame_times = np.load(times_path)
    state = np.load(state_path)

    seconds = result['seconds']
    truth = np.stack([np.interp(seconds, frame_times, state[:, 0]),
                      np.interp(seconds, frame_times, state[:, 2])], axis=1)
    online_errors = np.linalg.norm(
        result['online_estimates'][:, 2:] - truth, axis=1)

    path_seconds = result['path_seconds'][1:]
    final_path = result['revised_path'][1:, 2:]
    path_truth = np.stack([np.interp(path_seconds, frame_times, state[:, 0]),
                           np.interp(path_seconds, frame_times, state[:, 2])], axis=1)
    final_errors = np.linalg.norm(final_path - path_truth, axis=1)
    return {
        'final_path_ade_m': float(final_errors.mean()),
        'online_ade_m': float(online_errors.mean()),
        'fde_m': float(final_errors[-1]),
        'x_mae_m': float(np.mean(np.abs(final_path[:, 0] - path_truth[:, 0]))),
        'y_mae_m': float(np.mean(np.abs(final_path[:, 1] - path_truth[:, 1]))),
        'timestamps': int(len(final_errors)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run a v1 model on every sample in an independent dataset.')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to the trained v1 model.pt')
    parser.add_argument('--input', required=True,
                        help='External dataset root containing sample directories')
    parser.add_argument('--output', required=True,
                        help='Directory for NPZ files, plots, and metrics')
    parser.add_argument('--limit', type=int,
                        help='Process only the first N clips for a quick check')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint not found: {checkpoint_path}')
    if not input_path.is_dir():
        parser.error(f'Input dataset not found: {input_path}')

    sample_paths = samples(input_path)
    if args.limit is not None:
        sample_paths = sample_paths[:args.limit]
    if not sample_paths:
        parser.error(f'No spectrograms/stft.npy samples found under {input_path}')

    output_path.mkdir(parents=True, exist_ok=True)
    xy_path = output_path / 'xy'
    xy_path.mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available()
              else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'nemo_conformer_motion_v1':
        raise ValueError('external_test.py requires a nemo_conformer_motion_v1 checkpoint')
    model = MotionModel(checkpoint['bins'], checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    completed, failed, clip_metrics = [], [], []
    for sample in tqdm(sample_paths, desc='external test', unit='clip'):
        prediction_path = output_path / f'{sample.name}.npz'
        try:
            result = predict_one(model, checkpoint, sample, device)
            metrics = trajectory_metrics(sample, result, checkpoint['state_file'])
            extra = {} if metrics is None else {
                key: value for key, value in metrics.items() if key != 'timestamps'
            }
            np.savez(prediction_path, **result, **extra)
            plot_prediction(prediction_path, output_path / f'{sample.name}.png', sample)
            plot_xy_prediction(prediction_path, xy_path / f'{sample.name}.png', sample)
            completed.append(str(sample))
            if metrics is not None:
                clip_metrics.append({'sample': sample.name, **metrics})
        except Exception as error:
            failed.append({'sample': str(sample), 'error': str(error)})

    aggregate = None
    if clip_metrics:
        aggregate = {
            'primary_metric': 'final_path_ade_m',
            'final_path_ade_m': float(np.mean(
                [row['final_path_ade_m'] for row in clip_metrics])),
            'median_clip_final_path_ade_m': float(np.median(
                [row['final_path_ade_m'] for row in clip_metrics])),
            'online_ade_m': float(np.mean(
                [row['online_ade_m'] for row in clip_metrics])),
            'fde_m': float(np.mean([row['fde_m'] for row in clip_metrics])),
            'x_mae_m': float(np.mean([row['x_mae_m'] for row in clip_metrics])),
            'y_mae_m': float(np.mean([row['y_mae_m'] for row in clip_metrics])),
        }
    summary = {
        'checkpoint': str(checkpoint_path),
        'input': str(input_path),
        'output': str(output_path),
        'device': device,
        'requested': len(sample_paths),
        'completed': len(completed),
        'failed': failed,
        'metrics': aggregate,
    }
    (output_path / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output_path / 'metrics.json').write_text(json.dumps(
        {'summary': aggregate, 'clips': clip_metrics}, indent=2))
    print(f'Completed {len(completed)}/{len(sample_paths)} clips; results: {output_path}')
    if aggregate is not None and clip_metrics:
        print(f"Final-path ADE: {aggregate['final_path_ade_m']:.3f} m; "
              f"Online ADE: {aggregate['online_ade_m']:.3f} m")
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
    else:
        print('Ground-truth state files were unavailable; metrics were not calculated.')
    if failed:
        raise SystemExit(f'{len(failed)} clips failed; see {output_path / "summary.json"}')


if __name__ == '__main__':
    main()
