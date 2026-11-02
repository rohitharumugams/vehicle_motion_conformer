import torch
import torch.nn.functional as F
from config import (HEAD_WEIGHTS, KINEMATIC_WEIGHT, ACCELERATION_WEIGHT,
                    MAX_ACCEL_MPS2, LABEL_SMOOTHING_SIGMA,
                    POSITION_HEAD_WEIGHT, ENDPOINT_WEIGHT,
                    PATH_INTEGRATION_WEIGHT, POLAR_HEAD_WEIGHT,
                    GEOMETRIC_WEIGHT)


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
                gt_position=None, polar_pred=None, gt_polar=None,
                kinematic_weight=KINEMATIC_WEIGHT,
                acceleration_weight=ACCELERATION_WEIGHT,
                label_smoothing_sigma=LABEL_SMOOTHING_SIGMA,
                position_head_weight=POSITION_HEAD_WEIGHT,
                endpoint_weight=ENDPOINT_WEIGHT,
                path_int_weight=PATH_INTEGRATION_WEIGHT,
                polar_weight=POLAR_HEAD_WEIGHT,
                geometric_weight=GEOMETRIC_WEIGHT):
    """Four categorical CE losses (with Gaussian soft targets and endpoint weighting)
    plus soft physics penalties, cumulative path integration loss, and polar consistency."""
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

    # 1. Cumulative velocity integration loss (penalizes long-term heading drift)
    path_int_loss = means.new_zeros(())
    if gt_position is not None:
        mid_v = 0.5 * (velocity + previous_v)
        disp = torch.cumsum(mid_v * dt[:, None], dim=0)
        p_int = p0.reshape(1, 2) + disp
        path_res = (p_int - gt_position) * w[:, None]
        path_int_loss = F.smooth_l1_loss(path_res / 5.0, torch.zeros_like(path_res))
        total = total + path_int_weight * path_int_loss

    # 2. Auxiliary continuous position regression loss (anchored residual from p0)
    pos_loss = means.new_zeros(())
    if pos_pred is not None and gt_position is not None:
        pos_residual = (pos_pred - gt_position) * w[:, None]
        pos_loss = F.smooth_l1_loss(pos_residual / 10.0, torch.zeros_like(pos_residual))
        total = total + position_head_weight * pos_loss

    # 3. Auxiliary polar physics loss: range (r) and radial velocity (vr)
    polar_loss = means.new_zeros(())
    geo_loss = means.new_zeros(())
    if polar_pred is not None and gt_polar is not None:
        r_res = (polar_pred[:, 0] - gt_polar[:, 0]) / 50.0
        vr_res = (polar_pred[:, 1] - gt_polar[:, 1]) / 20.0
        polar_loss = F.smooth_l1_loss(r_res, torch.zeros_like(r_res)) + 0.5 * F.smooth_l1_loss(vr_res, torch.zeros_like(vr_res))
        total = total + polar_weight * polar_loss

        # Geometric consistency: norm of position should equal acoustic range
        if pos_pred is not None:
            pred_dist = torch.linalg.vector_norm(pos_pred, dim=-1)
            geo_res = (pred_dist - gt_polar[:, 0]) / 50.0
            geo_loss = F.smooth_l1_loss(geo_res, torch.zeros_like(geo_res))
            total = total + geometric_weight * geo_loss

    return total, dict(ce=ce, kinematic=kinematic, acceleration=acceleration,
                       path_int_loss=path_int_loss, pos_loss=pos_loss,
                       polar_loss=polar_loss, geo_loss=geo_loss, means=means)
