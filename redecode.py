"""Replay saved probabilities through the replacement decoder; no model needed."""
import argparse
import json
from pathlib import Path
import numpy as np
from decoder import MotionDecoder

NAMES = ('vx', 'vy', 'px', 'py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    parser.add_argument('--save-revisions', action='store_true')
    parser.add_argument('--accel-weight', type=float, default=0.1)
    parser.add_argument('--motion-weight', type=float, default=1.5)
    parser.add_argument('--smooth', action='store_true', default=True)
    parser.add_argument('--no-smooth', action='store_false', dest='smooth')
    parser.add_argument('--adaptive-noise', action='store_true', default=True)
    parser.add_argument('--no-adaptive-noise', action='store_false', dest='adaptive_noise')
    parser.add_argument('--q-accel', type=float, default=3.0)
    parser.add_argument('--r-pos', type=float, default=1.5)
    parser.add_argument('--r-vel', type=float, default=2.0)
    parser.add_argument('--sample', help='Optional sample directory to calculate trajectory metrics')
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error('Use a different output filename to preserve the original run')
    with np.load(args.input) as f:
        data = {key: f[key].copy() for key in f.files}
    decoder = MotionDecoder(data['initial_position'], [data[n + '_bins'] for n in NAMES],
                            accel_weight=args.accel_weight, motion_weight=args.motion_weight,
                            smooth_rts=args.smooth)
    online, revisions = [], []
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
        r_pos=args.r_pos, r_vel=args.r_vel
    )
    data['decoder_version'] = np.asarray('adaptive_continuous_expectation_v4')
    data['decoder_weights'] = np.array([args.accel_weight, args.motion_weight, args.q_accel, args.r_pos, args.r_vel])
    if args.sample:
        from external_test import trajectory_metrics
        m = trajectory_metrics(args.sample, data, 'canonical_state_frames.npy')
        if m:
            for k, v in m.items():
                if k != 'timestamps':
                    data[k] = np.asarray(v)
            print(f"ADE: {m['final_path_ade_m']:.3f} m, FDE: {m['fde_m']:.3f} m, Online ADE: {m['online_ade_m']:.3f} m")
    np.savez(args.output, **data)
    if args.save_revisions:
        with open(args.output + '.revisions.jsonl', 'w') as f:
            for row in revisions:
                f.write(json.dumps(row, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
