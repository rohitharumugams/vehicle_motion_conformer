import argparse
import json
import numpy as np
import torch
from config import NAMES
from data import initial_position, load_audio
from decoder import MotionDecoder
from model import MotionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('px', nargs='?', type=float)
    parser.add_argument('py', nargs='?', type=float)
    parser.add_argument('--checkpoint', default='model.pt')
    parser.add_argument('--output', default='probabilities.npz')
    parser.add_argument('--save-revisions', action='store_true')
    args = parser.parse_args()
    if (args.px is None) != (args.py is None):
        parser.error('Supply both px and py, or neither')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('architecture') != 'nemo_conformer_motion_v1':
        raise ValueError('This model needs a NEW Conformer checkpoint; retrain first')
    model = MotionModel(ckpt['bins'], ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    bins = [np.asarray(b, dtype=np.float32) for b in ckpt['bins']]
    x, ix, seconds = load_audio(args.sample, ckpt['feature_config'])
    # Reads no future motion labels. Explicit px/py avoids state-file input entirely.
    p0 = (np.array([args.px, args.py], dtype=np.float32) if args.px is not None
          else initial_position(args.sample, ckpt['state_file']))
    position_input = torch.from_numpy(p0)[None].to(device)
    decoder = MotionDecoder(p0, bins)
    probabilities = {name: [] for name in NAMES}
    online, raw, means, revisions = [], [], [], []
    cache, start, last_probs = None, 0, None
    for second, end in zip(seconds, ix):
        if end >= start:
            chunk = torch.from_numpy(x[start:end + 1])[None].to(device)
            logits, cache = model.predict_chunk(chunk, position_input, cache)
            last_probs = [z[0, -1].softmax(-1).cpu().numpy() for z in logits]
            start = int(end) + 1
        # If no new feature arrived, keep the most recent acoustic distributions.
        if last_probs is None:
            raise RuntimeError('No acoustic prediction available yet')
        for name, prob in zip(NAMES, last_probs):
            probabilities[name].append(prob.copy())
        raw.append([b[p.argmax()] for b, p in zip(bins, last_probs)])
        means.append([float(b @ p) for b, p in zip(bins, last_probs)])
        state = decoder.step(last_probs, second)
        online.append(state)
        print(f'{second:.1f}s: ' + ' '.join(f'{name}={v:.2f}' for name, v in zip(NAMES, state)))
        if args.save_revisions:
            t, path = decoder.best_path()
            revisions.append(dict(time=float(second), seconds=t.tolist(),
                                  path=[[None if np.isnan(v) else float(v) for v in row] for row in path]))
    path_seconds, revised_path = decoder.best_path()
    np.savez(args.output, seconds=seconds, initial_position=p0,
             online_estimates=np.asarray(online), raw_argmax=np.asarray(raw),
             raw_expectation=np.asarray(means), path_seconds=path_seconds,
             revised_path=revised_path, names=np.asarray(NAMES),
             **{name: np.stack(probs) for name, probs in probabilities.items()},
             **{name + '_bins': b for name, b in zip(NAMES, bins)})
    if args.save_revisions:
        with open(args.output + '.revisions.jsonl', 'w') as f:
            for revision in revisions:
                f.write(json.dumps(revision, allow_nan=False) + '\n')
    print(f'Saved {args.output}; revised_path starts at t=0 with the exact supplied position.')


if __name__ == '__main__':
    main()
