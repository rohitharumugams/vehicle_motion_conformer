import torch
import torch.nn.functional as F
from config import (HEAD_WEIGHTS, KINEMATIC_WEIGHT, ACCELERATION_WEIGHT,
                    MAX_ACCEL_MPS2, LABEL_SMOOTHING_SIGMA,
                    POSITION_HEAD_WEIGHT, ENDPOINT_WEIGHT)


def gaussian_soft_target(target_indices, num_classes, sigma=1.0, device='cpu'):
    """Generate Gaussian-smoothed probability targets over discrete ordinal classes."""
    classes = torch.arange(num_classes, device=device, dtype=torch.float32)
    dist_sq = (classes[None, :] - target_indices[:, None].float()) ** 2
    return F.softmax(-dist_sq / (2.0 * max(float(sigma), 0.1) ** 2), dim=-1)


def endpoint_weights(T, endpoint_weight=ENDPOINT_WEIGHT, device='cpu'):
    """Linear ramp from 1.0 at t=0 to endpoint_weight at t=T-1, normalised to sum=T."""
    if T == 1:
        return torch.ones(1, device=device)
    w = torch.linspace(1.0, float(endpoint_weight), T, device=device)
    return w * T / w.sum()


def motion_loss(logits, target, bins, seconds, p0, pos_pred=None,
                gt_position=None,
                kinematic_weight=KINEMATIC_WEIGHT,
                acceleration_weight=ACCELERATION_WEIGHT,
                label_smoothing_sigma=LABEL_SMOOTHING_SIGMA,
                position_head_weight=POSITION_HEAD_WEIGHT,
                endpoint_weight=ENDPOINT_WEIGHT):
    """Four categorical CE losses (with Gaussian soft targets and endpoint weighting)
    plus soft physics penalties plus optional auxiliary direct-position regression loss."""
    T = logits[0].shape[0]
    w = endpoint_weights(T, endpoint_weight=endpoint_weight, device=logits[0].device)

    ce_losses = []
    for j, z in enumerate(logits):
        if label_smoothing_sigma is not None and label_smoothing_sigma > 0:
            soft_target = gaussian_soft_target(target[:, j], z.shape[-1],
                                               sigma=label_smoothing_sigma,
                                               device=z.device)
            per_step = F.cross_entropy(z, soft_target, reduction='none')
        else:
            per_step = F.cross_entropy(z, target[:, j], reduction='none')
        ce_losses.append((per_step * w).mean())

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

    # Auxiliary direct position regression loss.
    pos_loss = means.new_zeros(())
    if pos_pred is not None and gt_position is not None:
        # pos_pred: [T, 2] raw regression output (px, py) in metres
        # gt_position: [T, 2] ground-truth (px, py) in metres
        # Weight by endpoint ramp so the final position matters more
        pos_residual = (pos_pred - gt_position) * w[:, None]
        pos_loss = F.smooth_l1_loss(pos_residual / 10.0, torch.zeros_like(pos_residual))
        total = total + position_head_weight * pos_loss

    return total, dict(ce=ce, kinematic=kinematic, acceleration=acceleration,
                       pos_loss=pos_loss, means=means)
