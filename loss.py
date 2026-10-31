import torch
import torch.nn.functional as F
from config import (HEAD_WEIGHTS, KINEMATIC_WEIGHT, ACCELERATION_WEIGHT,
                    MAX_ACCEL_MPS2, LABEL_SMOOTHING_SIGMA)


def gaussian_soft_target(target_indices, num_classes, sigma=1.0, device='cpu'):
    """Generate Gaussian-smoothed probability targets over discrete ordinal classes."""
    classes = torch.arange(num_classes, device=device, dtype=torch.float32)
    dist_sq = (classes[None, :] - target_indices[:, None].float()) ** 2
    return F.softmax(-dist_sq / (2.0 * max(float(sigma), 0.1) ** 2), dim=-1)


def motion_loss(logits, target, bins, seconds, p0,
                kinematic_weight=KINEMATIC_WEIGHT, acceleration_weight=ACCELERATION_WEIGHT,
                label_smoothing_sigma=LABEL_SMOOTHING_SIGMA):
    """Four categorical CE losses (with Gaussian soft targets) plus soft physics penalties."""
    ce_losses = []
    for j, z in enumerate(logits):
        if label_smoothing_sigma is not None and label_smoothing_sigma > 0:
            soft_target = gaussian_soft_target(target[:, j], z.shape[-1], sigma=label_smoothing_sigma, device=z.device)
            ce_losses.append(F.cross_entropy(z, soft_target))
        else:
            ce_losses.append(F.cross_entropy(z, target[:, j]))
    ce = torch.stack(ce_losses)
    weights = ce.new_tensor(HEAD_WEIGHTS)
    total = (ce * weights).sum() / weights.sum()
    means = torch.stack([(z.softmax(-1) * b).sum(-1) for z, b in zip(logits, bins)], -1)
    times = torch.as_tensor(seconds, device=means.device, dtype=means.dtype)
    velocity = means[:, :2] / 3.6
    position = means[:, 2:]
    dt = torch.diff(torch.cat((times.new_zeros(1), times)))
    previous_p = torch.cat((p0.reshape(1, 2), position[:-1]), 0)
    previous_v = torch.cat((velocity[:1], velocity[:-1]), 0)
    residual = position - previous_p - 0.5 * (velocity + previous_v) * dt[:, None]
    kinematic = F.smooth_l1_loss(residual / 2.0, torch.zeros_like(residual))
    acceleration = means.new_zeros(())
    if len(times) > 1:
        a = torch.linalg.vector_norm(torch.diff(velocity, dim=0), dim=-1) / dt[1:]
        acceleration = (F.relu(a / MAX_ACCEL_MPS2 - 1.0) ** 2).mean()
    total = total + kinematic_weight * kinematic + acceleration_weight * acceleration
    return total, dict(ce=ce, kinematic=kinematic, acceleration=acceleration, means=means)
