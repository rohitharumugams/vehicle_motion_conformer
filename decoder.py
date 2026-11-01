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


def rts_smooth_path(times, states, q_accel=3.0, r_pos=1.5, r_vel=2.0,
                    measurement_stds=None, min_r_pos=0.5, min_r_vel=1.0):
    """Rauch-Tung-Striebel (RTS) Kalman smoother for vehicle trajectories.

    Eliminates discrete quantization steps and produces kinematically smooth,
    continuous paths adhering to vehicle dynamics. Supports adaptive measurement
    covariance based on categorical observation standard deviations.
    
    Args:
        times: (T,) array of timestamps in seconds.
        states: (T, 4) array of [vx_kmph, vy_kmph, px_m, py_m].
        q_accel: Process noise acceleration std in m/s^2.
        r_pos: Base measurement noise position std in meters.
        r_vel: Base measurement noise velocity std in km/h.
        measurement_stds: Optional (T, 4) array of observation stds for adaptive noise.
        min_r_pos: Minimum allowable measurement position std in meters.
        min_r_vel: Minimum allowable measurement velocity std in km/h.
    Returns:
        smoothed_states: (T, 4) array of smoothed states.
    """
    T = len(times)
    if T < 2:
        return states.copy()

    # Initial velocities (handle initial NaN in beam search node)
    v0_x = 0.0 if np.isnan(states[0, 0]) else states[0, 0] / 3.6
    v0_y = 0.0 if np.isnan(states[0, 1]) else states[0, 1] / 3.6
    if np.isnan(states[0, 0]) and T > 1 and not np.isnan(states[1, 0]):
        v0_x, v0_y = states[1, 0] / 3.6, states[1, 1] / 3.6

    # Forward Kalman Filter pass
    # State vector: [px (m), py (m), vx (m/s), vy (m/s)]
    xs_filt = [np.array([states[0, 2], states[0, 3], v0_x, v0_y], dtype=np.float64)]
    Ps_filt = [np.diag([0.01, 0.01, 25.0, 25.0])]
    Fs, Qs, P_preds = [], [], []

    for k in range(1, T):
        dt = max(float(times[k] - times[k - 1]), 0.01)
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        q = q_accel ** 2
        Q = np.array([
            [0.25 * dt**4, 0.0,          0.5 * dt**3,  0.0        ],
            [0.0,          0.25 * dt**4, 0.0,          0.5 * dt**3],
            [0.5 * dt**3,  0.0,          dt**2,        0.0        ],
            [0.0,          0.5 * dt**3,  0.0,          dt**2      ]
        ], dtype=np.float64) * q

        # Predict
        x_pred = F @ xs_filt[-1]
        P_pred = F @ Ps_filt[-1] @ F.T + Q
        P_preds.append(P_pred)

        # Measurement
        v_x_mps = states[k, 0] / 3.6 if not np.isnan(states[k, 0]) else 0.0
        v_y_mps = states[k, 1] / 3.6 if not np.isnan(states[k, 1]) else 0.0
        z = np.array([states[k, 2], states[k, 3], v_x_mps, v_y_mps], dtype=np.float64)
        H = np.eye(4, dtype=np.float64)

        if measurement_stds is not None and k < len(measurement_stds):
            s = measurement_stds[k]
            r_px = max(float(s[2]), min_r_pos)
            r_py = max(float(s[3]), min_r_pos)
            r_vx = max(float(s[0]) / 3.6, min_r_vel / 3.6)
            r_vy = max(float(s[1]) / 3.6, min_r_vel / 3.6)
            R = np.diag([r_px**2, r_py**2, r_vx**2, r_vy**2])
        else:
            R = np.diag([r_pos**2, r_pos**2, (r_vel / 3.6)**2, (r_vel / 3.6)**2])

        # Innovation and update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ y
        P_upd = (np.eye(4, dtype=np.float64) - K @ H) @ P_pred

        xs_filt.append(x_upd)
        Ps_filt.append(P_upd)
        Fs.append(F)
        Qs.append(Q)

    # Backward RTS smoother pass
    xs_smooth = [None] * T
    Ps_smooth = [None] * T
    xs_smooth[-1] = xs_filt[-1].copy()
    Ps_smooth[-1] = Ps_filt[-1].copy()

    for k in range(T - 2, -1, -1):
        F = Fs[k]
        P_pred = P_preds[k]
        C_k = Ps_filt[k] @ F.T @ np.linalg.inv(P_pred)
        xs_smooth[k] = xs_filt[k] + C_k @ (xs_smooth[k + 1] - F @ xs_filt[k])
        Ps_smooth[k] = Ps_filt[k] + C_k @ (Ps_smooth[k + 1] - P_pred) @ C_k.T

    # Convert back to [vx_kmph, vy_kmph, px, py]
    smoothed = np.array([[x[2] * 3.6, x[3] * 3.6, x[0], x[1]] for x in xs_smooth])
    # Keep initial position fixed to exact p0
    smoothed[0, 2:] = states[0, 2:]
    return smoothed


def expectation_rts_path(p0, seconds, expectation, q_accel=None, r_pos=None, r_vel=None):
    """RTS-smooth the per-step head expectations directly (no beam search).

    Args:
        p0: (2,) known starting position.
        seconds: (T,) output timestamps (excluding t=0).
        expectation: (T, 4) per-step [vx_kmph, vy_kmph, px_m, py_m] expectations.
    Returns:
        (path_seconds [T+1], path [T+1, 4]) in the same layout as MotionDecoder.best_path().
    """
    q_accel = C.EXPECTATION_RTS_Q_ACCEL if q_accel is None else q_accel
    r_pos = C.EXPECTATION_RTS_R_POS if r_pos is None else r_pos
    r_vel = C.EXPECTATION_RTS_R_VEL if r_vel is None else r_vel
    seconds = np.asarray(seconds, dtype=np.float64)
    expectation = np.asarray(expectation, dtype=np.float64)
    times = np.r_[0.0, seconds]
    states = np.vstack([np.r_[np.nan, np.nan, np.asarray(p0, dtype=np.float64)], expectation])
    if len(states) < 3:
        return times, states
    return times, rts_smooth_path(times, states, q_accel=q_accel, r_pos=r_pos, r_vel=r_vel)


class MotionDecoder:
    """Continuous-expectation beam search with soft motion consistency and RTS smoothing.

    Uses continuous expectation (\\sum b_i p_i) for position proposals instead of
    discrete bin centers, completely eliminating 2-meter staircasing artifacts.
    Enforces kinematic acceleration and trajectory smoothness.
    """
    def __init__(self, p0, bins, beam_size=C.BEAM_SIZE,
                 max_accel=C.MAX_ACCEL_MPS2, max_speed=C.MAX_SPEED_KMPH,
                 top_k=4, accel_weight=getattr(C, 'DECODER_ACCEL_PENALTY', 0.1),
                 motion_weight=getattr(C, 'DECODER_MOTION_WEIGHT', 1.5),
                 position_scale=1.5, smooth_rts=True):
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
        self.smooth_rts = smooth_rts
        self.beam = [Node(0.0, np.r_[np.nan, np.nan, p0], 0.0)]
        self.history_stds = [np.array([0.0, 0.0, 0.01, 0.01], dtype=np.float64)]

    def step(self, probabilities, time):
        if len(probabilities) != 4 or not np.isfinite(time) or time <= self.beam[0].time:
            raise ValueError('Need four distributions and a strictly increasing finite timestamp')
        probs, means, stds = [], [], []
        for p, b in zip(probabilities, self.bins):
            p = np.asarray(p, dtype=np.float64)
            if p.shape != b.shape or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
                raise ValueError('Invalid categorical probabilities')
            p = p / p.sum()
            probs.append(p)
            mu = float(b @ p)
            means.append(mu)
            var = float(np.sum((b - mu) ** 2 * p))
            stds.append(np.sqrt(max(var, 0.01)))

        self.history_stds.append(np.array(stds, dtype=np.float64))

        dt = float(time) - self.beam[0].time
        # Continuous position observation from expectation
        obs_pos = np.array([means[2], means[3]], dtype=np.float64)

        # Propose top-k velocities + expectation
        v_candidates = []
        for j in range(2):
            ix = np.argsort(-probs[j], kind='stable')[:self.top_k]
            v_candidates.append(np.unique(np.r_[means[j], self.bins[j][ix]]))

        velocities = np.array(np.meshgrid(v_candidates[0], v_candidates[1], indexing='ij')).reshape(2, -1).T
        velocities = velocities[np.linalg.norm(velocities, axis=1) <= self.max_speed + 1e-8]
        if not len(velocities):
            velocities = np.array([[0.0, 0.0]])

        candidate_states, candidate_scores, parents = [], [], []
        for parent_id, node in enumerate(self.beam):
            prev_v = velocities if np.isnan(node.state[0]) else node.state[:2]
            integrated = node.state[2:] + (velocities + prev_v) / 7.2 * dt

            # Continuous position proposals:
            # - Kinematically integrated position from previous state and current velocity
            # - Continuous expectation from acoustic observation
            # - Convex combinations to smoothly blend acoustic evidence and kinematic motion
            pos_proposals = np.stack([
                integrated,
                np.broadcast_to(obs_pos, integrated.shape),
                0.5 * integrated + 0.5 * obs_pos,
                0.7 * integrated + 0.3 * obs_pos,
                0.3 * integrated + 0.7 * obs_pos,
            ], axis=1)

            V, P, _ = pos_proposals.shape
            v_expanded = np.broadcast_to(velocities[:, None, :], (V, P, 2))
            states = np.concatenate([v_expanded, pos_proposals], axis=-1).reshape(-1, 4)

            cur_integrated = np.broadcast_to(integrated[:, None, :], (V, P, 2)).reshape(-1, 2)
            cur_positions = states[:, 2:]
            residual = (cur_positions - cur_integrated) / self.position_scale
            motion_cost = huber(residual).sum(-1)

            if np.isnan(node.state[0]):
                accel_cost = np.zeros(len(states))
            else:
                delta_v = np.linalg.norm((states[:, :2] - node.state[:2]) / 3.6, axis=1)
                excess_acc = np.maximum(delta_v / (self.max_accel * dt) - 1.0, 0.0)
                accel_cost = huber(excess_acc)

            log_lik = np.zeros(len(states))
            for j in range(4):
                val = states[:, j]
                prob_val = np.interp(val, self.bins[j], probs[j], left=1e-12, right=1e-12)
                log_lik += np.log(np.maximum(prob_val, 1e-12))

            scores = node.score + log_lik - (self.motion_weight * motion_cost + self.accel_weight * accel_cost)

            valid = np.ones(len(states), dtype=bool)
            for j, b in enumerate(self.bins):
                valid &= (states[:, j] >= b[0]) & (states[:, j] <= b[-1])

            states, scores = states[valid], scores[valid]
            if len(states):
                candidate_states.append(states)
                candidate_scores.append(scores)
                parents.extend([parent_id] * len(states))

        if not candidate_states:
            raise RuntimeError('No candidates within configured speed/bin bounds')
        states, scores = np.concatenate(candidate_states), np.concatenate(candidate_scores)

        # Diverse beam pruning
        rounded = np.round(states, 3)
        order = np.argsort(-scores, kind='stable')
        _, first = np.unique(rounded[order], axis=0, return_index=True)
        chosen = order[np.sort(first)[:self.beam_size]]
        if not len(chosen):
            raise RuntimeError('No candidates within configured speed/bin bounds')

        old_beam = self.beam
        self.beam = [Node(float(time), states[i].copy(), float(scores[i]), old_beam[parents[i]])
                     for i in chosen]
        return self.beam[0].state.copy()

    def best_path(self, smooth=None, adaptive_noise=True, q_accel=3.0, r_pos=1.5, r_vel=2.0):
        """Retrieve the best-scoring trajectory, optionally smoothed with RTS Kalman filter."""
        if smooth is None:
            smooth = self.smooth_rts
        nodes, node = [], self.beam[0]
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        times = np.array([n.time for n in nodes])
        states = np.stack([n.state for n in nodes])
        if smooth and len(states) >= 3:
            meas_stds = None
            if adaptive_noise and hasattr(self, 'history_stds') and len(self.history_stds) == len(states):
                meas_stds = np.array(self.history_stds)
            states = rts_smooth_path(times, states, q_accel=q_accel, r_pos=r_pos, r_vel=r_vel,
                                    measurement_stds=meas_stds)
        return times, states


class KalmanRTSDecoder:
    """Pure Kalman filter & RTS smoother directly integrating velocity and position expectations.

    Operates in O(T) time and produces smooth, continuous, physically consistent trajectories.
    """
    def __init__(self, p0, bins, q_accel=3.0, r_pos_base=1.5, r_vel_base=2.0):
        self.bins = [np.asarray(b, dtype=np.float64) for b in bins]
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.q_accel = q_accel
        self.r_pos_base = r_pos_base
        self.r_vel_base = r_vel_base
        # State: [px (m), py (m), vx (m/s), vy (m/s)]
        self.x = np.array([p0[0], p0[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([0.01, 0.01, 25.0, 25.0])
        self.history_t = [0.0]
        self.history_x = [self.x.copy()]
        self.history_P = [self.P.copy()]
        self.history_F = []
        self.history_Q = []
        self.history_P_pred = []

    def step(self, probabilities, time):
        means, stds = [], []
        for p, b in zip(probabilities, self.bins):
            p = np.asarray(p, dtype=np.float64)
            p = p / p.sum()
            mu = float(b @ p)
            var = float(np.sum((b - mu) ** 2 * p))
            means.append(mu)
            stds.append(np.sqrt(max(var, 0.1)))

        dt = float(time) - self.history_t[-1]
        if dt <= 0:
            dt = 0.1

        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        q = self.q_accel ** 2
        Q = np.array([
            [0.25 * dt**4, 0.0,          0.5 * dt**3,  0.0        ],
            [0.0,          0.25 * dt**4, 0.0,          0.5 * dt**3],
            [0.5 * dt**3,  0.0,          dt**2,        0.0        ],
            [0.0,          0.5 * dt**3,  0.0,          dt**2      ]
        ], dtype=np.float64) * q

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Measurement
        z = np.array([means[2], means[3], means[0] / 3.6, means[1] / 3.6], dtype=np.float64)
        H = np.eye(4, dtype=np.float64)
        R = np.diag([
            max(stds[2], self.r_pos_base)**2,
            max(stds[3], self.r_pos_base)**2,
            max(stds[0] / 3.6, self.r_vel_base / 3.6)**2,
            max(stds[1] / 3.6, self.r_vel_base / 3.6)**2,
        ])

        # Innovation and update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4, dtype=np.float64) - K @ H) @ P_pred

        self.history_t.append(float(time))
        self.history_x.append(self.x.copy())
        self.history_P.append(self.P.copy())
        self.history_F.append(F)
        self.history_Q.append(Q)
        self.history_P_pred.append(P_pred)

        # Return [vx_kmph, vy_kmph, px, py]
        return np.array([self.x[2] * 3.6, self.x[3] * 3.6, self.x[0], self.x[1]])

    def best_path(self):
        n = len(self.history_x)
        xs = [None] * n
        Ps = [None] * n
        xs[-1] = self.history_x[-1].copy()
        Ps[-1] = self.history_P[-1].copy()

        for k in range(n - 2, -1, -1):
            F = self.history_F[k]
            P_pred = self.history_P_pred[k]
            C_k = self.history_P[k] @ F.T @ np.linalg.inv(P_pred)
            xs[k] = self.history_x[k] + C_k @ (xs[k + 1] - F @ self.history_x[k])
            Ps[k] = self.history_P[k] + C_k @ (Ps[k + 1] - P_pred) @ C_k.T

        states = np.array([[x[2] * 3.6, x[3] * 3.6, x[0], x[1]] for x in xs])
        states[0, 2:] = self.p0
        return np.array(self.history_t), states
