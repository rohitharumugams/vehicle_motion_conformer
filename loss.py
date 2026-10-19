import torch
import torch.nn.functional as F
from config import (HEAD_WEIGHTS, KINEMATIC_WEIGHT, ACCELERATION_WEIGHT,
                    MAX_ACCEL_MPS2)


def motion_loss(logits, target, bins, seconds, p0,
                kinematic_weight=KINEMATIC_WEIGHT, acceleration_weight=ACCELERATION_WEIGHT):
    """Four individually computed CE losses + optional soft physics penalties.

    logits: four [timestamps, bins] tensors, target: [timestamps, 4].
    Bins: signed km/h for v; metres for p. No CTC alignment/collapse is used.
    """
    ce = torch.stack([F.cross_entropy(z, target[:, j]) for j, z in enumerate(logits)])
    weights = ce.new_tensor(HEAD_WEIGHTS)
    total = (ce * weights).sum() / weights.sum()
    means = torch.stack([(z.softmax(-1) * b).sum(-1) for z, b in zip(logits, bins)], -1)
    times = torch.as_tensor(seconds, device=means.device, dtype=means.dtype)
    velocity = means[:, :2] / 3.6
    position = means[:, 2:]
    dt = torch.diff(torch.cat((times.new_zeros(1), times)))
    previous_p = torch.cat((p0.reshape(1, 2), position[:-1]), 0)
    # The starting velocity is UNKNOWN: use first predicted velocity for interval 1.
    previous_v = torch.cat((velocity[:1], velocity[:-1]), 0)
    residual = position - previous_p - 0.5 * (velocity + previous_v) * dt[:, None]
    # Scale metres to a fixed, meaningful range; do not divide quantized positions by dt.
    kinematic = F.smooth_l1_loss(residual / 2.0, torch.zeros_like(residual))
    acceleration = means.new_zeros(())
    if len(times) > 1:
        a = torch.linalg.vector_norm(torch.diff(velocity, dim=0), dim=-1) / dt[1:]
        acceleration = (F.relu(a / MAX_ACCEL_MPS2 - 1.0) ** 2).mean()
    total = total + kinematic_weight * kinematic + acceleration_weight * acceleration
    return total, dict(ce=ce, kinematic=kinematic, acceleration=acceleration, means=means)
