import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import NAMES, STATE


def plot_prediction(predictions, output, sample=None):
    """Render one saved prediction file, optionally with its ground truth."""
    d = np.load(predictions)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    truth = None
    if sample:
        from pathlib import Path
        p = Path(sample) / 'metadata'
        t = np.load(p / 'frame_times.npy')
        s = np.load(p / STATE)
        truth = s[:, [1, 3, 0, 2]].copy()
        truth[:, :2] *= 3.6
    for j, (name, ax) in enumerate(zip(NAMES, axes)):
        mesh = ax.pcolormesh(d['seconds'], d[name + '_bins'], d[name].T,
                            shading='nearest', cmap='Blues', vmin=0,
                            vmax=max(float(d[name].max()), 1e-8))
        fig.colorbar(mesh, ax=ax, label='Probability')
        ax.plot(d['seconds'], d['raw_expectation'][:, j], color='orange', lw=1,
                label='Head expectation')
        ax.plot(d['path_seconds'], d['revised_path'][:, j], color='red', lw=1.5,
                label='Revised feasible path')
        if j >= 2:
            ax.scatter([0], [d['initial_position'][j - 2]], color='black', s=30, zorder=5,
                       label='Known initial position')
        if truth is not None:
            ax.plot(t, truth[:, j], color='black', lw=1, linestyle='--', label='Ground truth')
        ax.set_ylabel(f'{name} ({"km/h" if j < 2 else "m"})')
        ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Motion probabilities and constrained trajectory')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    d.close()


def plot_xy_prediction(predictions, output, sample=None):
    """Render the final revised trajectory directly in the x-y plane."""
    d = np.load(predictions)
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    if sample:
        from pathlib import Path
        state = np.load(Path(sample) / 'metadata' / STATE)
        ax.plot(state[:, 0], state[:, 2], color='black', linestyle='--', lw=2,
                label='Ground truth')
        title = Path(sample).name
    else:
        title = 'Prediction'
    path = d['revised_path']
    ax.plot(path[:, 2], path[:, 3], color='#f59a00', lw=2.5, label='Predicted path')
    ax.scatter(d['initial_position'][0], d['initial_position'][1], color='black',
               s=70, zorder=5, label='Known start')
    ax.set_title(f'{title}: path in the xy plane')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    d.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', nargs='?', default='probabilities.npz')
    parser.add_argument('--sample', help='Optional sample directory for ground-truth overlay')
    parser.add_argument('--output', default='motion_predictions.png')
    parser.add_argument('--xy-output', help='Also save a direct x-y trajectory plot')
    args = parser.parse_args()
    plot_prediction(args.predictions, args.output, args.sample)
    print(args.output)
    if args.xy_output:
        plot_xy_prediction(args.predictions, args.xy_output, args.sample)
        print(args.xy_output)


if __name__ == '__main__':
    main()
