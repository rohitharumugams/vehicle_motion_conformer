from dataclasses import dataclass
import numpy as np
import config as C


@dataclass
class Node:
    time: float
    state: np.ndarray
    score: float
    parent: object = None


def huber(x):
    x = np.abs(x)
    return np.where(x <= 1, 0.5 * x ** 2, x - 0.5)


class MotionDecoder:
    """Approximate online beam search with SOFT motion consistency.

    Global observations are proposed every step, allowing recovery after pruning.
    Acceleration is penalized, not forbidden. Positions can correct integrated
    motion. This is a simulator-oriented estimator, not a physical guarantee.
    """
    def __init__(self, p0, bins, beam_size=C.BEAM_SIZE,
                 max_accel=C.MAX_ACCEL_MPS2, max_speed=C.MAX_SPEED_KMPH,
                 top_k=4, accel_weight=0.05, motion_weight=0.5, position_scale=2.0):
        self.bins = [np.asarray(b, dtype=np.float64) for b in bins]
        p0 = np.asarray(p0, dtype=np.float64)
        if len(self.bins) != 4 or any(b.ndim != 1 or len(b) < 2 or
                                     not np.isfinite(b).all() or np.any(np.diff(b) <= 0)
                                     for b in self.bins):
            raise ValueError('Need four finite, strictly increasing bin arrays')
        if p0.shape != (2,) or not np.isfinite(p0).all():
            raise ValueError('p0 must be two finite coordinates')
        if any(not b[0] <= p0[j] <= b[-1] for j, b in enumerate(self.bins[2:])):
            raise ValueError('Initial position is outside the position bins')
        if min(beam_size, top_k, max_accel, max_speed, position_scale) <= 0:
            raise ValueError('Sizes and scales must be positive')
        if min(accel_weight, motion_weight) < 0:
            raise ValueError('Penalty weights must be nonnegative')
        self.beam_size, self.top_k = int(beam_size), int(top_k)
        self.max_accel, self.max_speed = max_accel, max_speed
        self.accel_weight, self.motion_weight = accel_weight, motion_weight
        self.position_scale = position_scale
        self.beam = [Node(0.0, np.r_[np.nan, np.nan, p0], 0.0)]

    def step(self, probabilities, time):
        if len(probabilities) != 4 or not np.isfinite(time) or time <= self.beam[0].time:
            raise ValueError('Need four distributions and a strictly increasing finite timestamp')
        probs, values = [], []
        for p, b in zip(probabilities, self.bins):
            p = np.asarray(p, dtype=np.float64)
            if p.shape != b.shape or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
                raise ValueError('Invalid categorical probabilities')
            p = p / p.sum()
            probs.append(p)
            ix = np.argsort(-p, kind='stable')[:self.top_k]
            values.append(np.unique(np.r_[b[ix], b @ p]))
        dt = float(time) - self.beam[0].time
        position_grid = np.array(np.meshgrid(values[2], values[3], indexing='ij')).reshape(2, -1).T
        candidate_states, candidate_scores, parents = [], [], []
        for parent_id, node in enumerate(self.beam):
            vv = values[:2]
            if not np.isnan(node.state[0]):
                vv = [np.unique(np.r_[vv[j], node.state[j]]) for j in range(2)]
            velocities = np.array(np.meshgrid(*vv, indexing='ij')).reshape(2, -1).T
            velocities = velocities[np.linalg.norm(velocities, axis=1) <= self.max_speed + 1e-8]
            if not len(velocities):
                continue
            previous_v = velocities if np.isnan(node.state[0]) else node.state[:2]
            integrated = node.state[2:] + (velocities + previous_v) / 7.2 * dt
            # For each velocity, propose observed position pairs AND its integrated position.
            obs = np.broadcast_to(position_grid, (len(velocities), *position_grid.shape))
            positions = np.concatenate((obs, integrated[:, None, :]), axis=1)
            states = np.concatenate((np.broadcast_to(velocities[:, None, :], positions.shape), positions), axis=-1)
            residual = (positions - integrated[:, None, :]) / self.position_scale
            motion_cost = huber(residual).sum(-1)
            delta = np.linalg.norm((velocities - previous_v) / 3.6, axis=1)
            excess = np.maximum(delta / (self.max_accel * dt) - 1.0, 0.0)
            accel_cost = huber(excess)[:, None]
            states = states.reshape(-1, 4)
            scores = np.full(len(states), node.score)
            for j, (p, b) in enumerate(zip(probs, self.bins)):
                scores += np.log(np.maximum(np.interp(states[:, j], b, p, left=0, right=0), 1e-12))
            scores -= (self.motion_weight * motion_cost + self.accel_weight * accel_cost).ravel()
            valid = np.ones(len(states), dtype=bool)
            for j, b in enumerate(self.bins):
                valid &= (states[:, j] >= b[0]) & (states[:, j] <= b[-1])
            states, scores = states[valid], scores[valid]
            candidate_states.append(states)
            candidate_scores.append(scores)
            parents.extend([parent_id] * len(states))
        if not candidate_states:
            raise RuntimeError('No candidates within configured speed/bin bounds')
        states, scores = np.concatenate(candidate_states), np.concatenate(candidate_scores)
        # Keep only the highest-scoring parent for each rounded current state.
        rounded = np.round(states, 4)
        order = np.argsort(-scores, kind='stable')
        _, first = np.unique(rounded[order], axis=0, return_index=True)
        chosen = order[np.sort(first)[:self.beam_size]]
        if not len(chosen):
            raise RuntimeError('No candidates within configured speed/bin bounds')
        old_beam = self.beam
        self.beam = [Node(float(time), states[i].copy(), float(scores[i]), old_beam[parents[i]])
                     for i in chosen]
        return self.beam[0].state.copy()

    def best_path(self):
        nodes, node = [], self.beam[0]
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return np.array([n.time for n in nodes]), np.stack([n.state for n in nodes])
