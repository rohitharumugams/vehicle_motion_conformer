"""Replay saved probabilities through a decoder; no model needed.

Works on a single .npz or on a whole directory of .npz files (batch mode).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import config as C
from decoder import MotionDecoder, expectation_rts_path

NAMES = ('vx', 'vy', 'px', 'py')


def expectations(data):
    """Per-step expectation of each head, shape [T, 4]."""
    return np.stack([data[n] @ data[n + '_bins'] / data[n].sum(-1) for n in NAMES], axis=1)


def redecode_one(data, args):
    """Return (data, revisions) with online_estimates/revised_path recomputed."""
    revisions = []
    if args.decoder == 'expectation_rts':
        # Online (causal) estimate is still produced by the beam decoder.
        decoder = MotionDecoder(data['initial_position'], [data[n + '_bins'] for n in NAMES],
                                accel_weight=args.accel_weight, motion_weight=args.motion_weight)
        data['online_estimates'] = np.asarray(
            [decoder.step([data[n][i] for n in NAMES], t) for i, t in enumerate(data['seconds'])])
        data['path_seconds'], data['revised_path'] = expectation_rts_path(
            data['initial_position'], data['seconds'], expectations(data),
            q_accel=args.q_accel, r_pos=args.r_pos, r_vel=args.r_vel)
        data['decoder_version'] = np.asarray('expectation_rts_v1')
        data['decoder_weights'] = np.array([args.q_accel, args.r_pos, args.r_vel])
        return data, revisions

    decoder = MotionDecoder(data['initial_position'], [data[n + '_bins'] for n in NAMES],
                            accel_weight=args.accel_weight, motion_weight=args.motion_weight,
                            smooth_rts=args.smooth)
    online = []
    for i, t in enumerate(data['seconds']):
        online.append(decoder.step([data[n][i] for n in NAMES], t))
        if args.save_revisions:
            ts, path = decoder.best_path(adaptive_noise=args.adaptive_noise,
                                         q_accel=args.q_accel, r_pos=args.r_pos, r_vel=args.r_vel)
            revisions.append(dict(time=float(t), seconds=ts.tolist(),
                                  path=[[None if np.isnan(v) else float(v) for v in row] for row in path]))
    data['online_estimates'] = np.asarray(online)
    data['path_seconds'], data['revised_path'] = decoder.best_path(
        adaptive_noise=args.adaptive_noise, q_accel=args.q_accel,
        r_pos=args.r_pos, r_vel=args.r_vel)
    data['decoder_version'] = np.asarray('adaptive_continuous_expectation_v4')
    data['decoder_weights'] = np.array([args.accel_weight, args.motion_weight,
                                        args.q_accel, args.r_pos, args.r_vel])
    return data, revisions


def add_metrics(data, sample):
    from external_test import trajectory_metrics
    m = trajectory_metrics(sample, data, 'canonical_state_frames.npy')
    if m:
        for k, v in m.items():
            if k != 'timestamps':
                data[k] = np.asarray(v)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='.npz file, or a directory of .npz files')
    parser.add_argument('--output', required=True, help='output .npz, or directory in batch mode')
    parser.add_argument('--decoder', choices=['expectation_rts', 'beam_rts'],
                        default=C.DECODER_MODE)
    parser.add_argument('--save-revisions', action='store_true')
    parser.add_argument('--accel-weight', type=float, default=0.1)
    parser.add_argument('--motion-weight', type=float, default=1.5)
    parser.add_argument('--smooth', action='store_true', default=True)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth')
    parser.add_argument('--adaptive-noise', action='store_true', default=True)
    parser.add_argument('--no-adaptive-noise', action='store_false', dest='adaptive_noise')
    parser.add_argument('--q-accel', type=float, default=None)
    parser.add_argument('--r-pos', type=float, default=None)
    parser.add_argument('--r-vel', type=float, default=None)
    parser.add_argument('--sample', help='Sample directory with ground truth (single-file mode)')
    parser.add_argument('--samples-root',
                        help='Dataset root containing <name>/metadata for ground truth (batch mode)')
    args = parser.parse_args()
    if args.q_accel is None:
        args.q_accel = C.EXPECTATION_RTS_Q_ACCEL if args.decoder == 'expectation_rts' else 3.0
    if args.r_pos is None:
        args.r_pos = C.EXPECTATION_RTS_R_POS if args.decoder == 'expectation_rts' else 1.5
    if args.r_vel is None:
        args.r_vel = C.EXPECTATION_RTS_R_VEL if args.decoder == 'expectation_rts' else 2.0

    src, dst = Path(args.input).resolve(), Path(args.output).resolve()
    if src == dst:
        parser.error('Use a different output path to preserve the original run')

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        rows = []
        for f in tqdm(sorted(src.glob('*.npz')), desc='redecode', unit='clip'):
            with np.load(f) as z:
                data = {key: z[key].copy() for key in z.files}
            data, _ = redecode_one(data, args)
            if args.samples_root:
                m = add_metrics(data, Path(args.samples_root) / f.stem)
                if m:
                    rows.append(m)
            np.savez(dst / f.name, **data)
        print(f'{dst}: {len(list(dst.glob("*.npz")))} files')
        if rows:
            for key, label in [('final_path_ade_m', 'ADE'), ('online_ade_m', 'Online ADE'),
                               ('fde_m', 'FDE'), ('x_mae_m', 'X MAE'), ('y_mae_m', 'Y MAE')]:
                v = np.array([r[key] for r in rows])
                print(f'{label}: mean={v.mean():.3f} median={np.median(v):.3f} '
                      f'p90={np.percentile(v, 90):.3f} p95={np.percentile(v, 95):.3f}')
        return

    with np.load(src) as f:
        data = {key: f[key].copy() for key in f.files}
    data, revisions = redecode_one(data, args)
    if args.sample:
        m = add_metrics(data, args.sample)
        if m:
            print(f"ADE: {m['final_path_ade_m']:.3f} m, FDE: {m['fde_m']:.3f} m, "
                  f"Online ADE: {m['online_ade_m']:.3f} m")
    np.savez(args.output, **data)
    if args.save_revisions and revisions:
        with open(args.output + '.revisions.jsonl', 'w') as f:
            for row in revisions:
                f.write(json.dumps(row, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
