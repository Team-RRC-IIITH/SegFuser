import torch
from utils.camera import *
import torch.nn.functional as F

def normalize_logit(logit, percentile=90.0, eps=1e-6):
    if not torch.is_tensor(logit):
        logit = torch.as_tensor(
            logit,
            dtype=torch.float32
        )

    scale = torch.quantile(
        torch.abs(logit).flatten(),
        percentile / 100.0
    )

    return logit / (scale + eps)

def fuse_logits_cuda(ref_idx, frame_idxs, track_ids, obj_logits, depth_gpu, normal_gpu,
                      depth_jump_gpu, normal_jump_gpu, depth_tolerance,
                      min_witnesses=2, max_contribution=3.0, epsilon=3.0):

    if ref_idx not in track_ids:
        raise ValueError(f"Reference frame {ref_idx} not found in track_ids")

    ref_seg = track_ids[ref_idx]
    if ref_idx not in obj_logits or ref_seg >= len(obj_logits[ref_idx]):
        raise ValueError(f"Invalid reference segment {ref_seg} for frame {ref_idx}")

    ref_key = str(ref_idx + 1)
    fx, fy, cx, cy = get_intrinsics(ref_key)
    Rw2c_ref, tw2c_ref = get_extrinsics(ref_key)

    depth_ref = depth_gpu[ref_idx]
    H, W = depth_ref.shape
    device = depth_ref.device

    ref_logit = obj_logits[ref_idx][ref_seg]

    if not torch.is_tensor(ref_logit):
        ref_logit = torch.from_numpy(
            ref_logit
        ).float().to(device)
    else:
        ref_logit = ref_logit.to(device)

    if ref_logit.shape != (H, W):
        ref_logit = F.interpolate(
    ref_logit[None, None],
    size=(H, W),
    mode="bilinear",
    align_corners=False
        )[0, 0]

    ref_logit = normalize_logit(ref_logit, percentile=90.0)

    R_ref = torch.as_tensor(Rw2c_ref, dtype=torch.float32, device=device)
    t_ref = torch.as_tensor(tw2c_ref, dtype=torch.float32, device=device)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth_ref.dtype),
        torch.arange(W, device=device, dtype=depth_ref.dtype),
        indexing="ij",
    )

    x_cam = (xs - cx) * depth_ref / fx
    y_cam = (ys - cy) * depth_ref / fy
    points_cam_ref = torch.stack([x_cam, y_cam, depth_ref], dim=-1)

    points_world = torch.einsum("ij,hwj->hwi", R_ref, points_cam_ref) + t_ref

    fused_logit = ref_logit.clone()
    witness_count = torch.zeros((H, W), device=device, dtype=torch.float32)

    for f_idx in frame_idxs:
        if f_idx == ref_idx or f_idx not in track_ids:
            continue
        if f_idx not in obj_logits:
            continue

        f_seg = track_ids[f_idx]
        if f_seg >= len(obj_logits[f_idx]):
            continue

        f_key = str(f_idx + 1)
        fx_f, fy_f, cx_f, cy_f = get_intrinsics(f_key)
        Rw2c_f, tw2c_f = get_extrinsics(f_key)

        depth_f = depth_gpu[f_idx]
        normal_f = normal_gpu[f_idx]
        depth_jump_f = depth_jump_gpu[f_idx]
        normal_jump_f = normal_jump_gpu[f_idx]

        logits_f = obj_logits[f_idx][f_seg]

        if not torch.is_tensor(logits_f):
            logits_f = torch.from_numpy(
                logits_f
            ).float().to(device)
        else:
            logits_f = logits_f.to(device)
    
        logits_f = normalize_logit(
            logits_f,
            percentile=90.0
        )

        Hf, Wf = depth_f.shape
        R_f = torch.as_tensor(Rw2c_f, dtype=torch.float32, device=device)
        t_f = torch.as_tensor(tw2c_f, dtype=torch.float32, device=device)

        points_cam_f = torch.einsum("ij,hwj->hwi", R_f.T, points_world - t_f)

        z_f = points_cam_f[..., 2]
        u_f = fx_f * points_cam_f[..., 0] / (z_f + 1e-8) + cx_f
        v_f = fy_f * points_cam_f[..., 1] / (z_f + 1e-8) + cy_f

        valid = (
            torch.isfinite(z_f) & (z_f > 0) & torch.isfinite(u_f) & torch.isfinite(v_f)
            & (u_f >= 0) & (u_f < Wf) & (v_f >= 0) & (v_f < Hf)
        )

        u = u_f.long().clamp(0, Wf - 1)
        v = v_f.long().clamp(0, Hf - 1)

        measured_depth = depth_f[v, u]
        valid &= (torch.isfinite(measured_depth) & (measured_depth > 0)
                  & (torch.abs(measured_depth - z_f) < depth_tolerance))

        sampled_logit = logits_f[v, u].clamp(-max_contribution, max_contribution)

        normal_f_cam = normal_f[v, u]
        normal_f_world = torch.einsum("ij,hwj->hwi", R_f, normal_f_cam)

        cam_center_f = t_f
        view_ray = F.normalize(points_world - cam_center_f, dim=-1)
        grazing = 1.0 - torch.abs((normal_f_world * view_ray).sum(dim=-1))

        depth_jump = depth_jump_f[v, u]
        normal_jump = normal_jump_f[v, u]
        geometric_ambiguity = torch.maximum(torch.maximum(grazing, depth_jump), normal_jump)

        agrees = (sampled_logit > 0) == (ref_logit > 0)
        weight = torch.where(agrees, torch.ones_like(geometric_ambiguity), 1.0 - geometric_ambiguity)
        weight = torch.clamp(weight, min=0.0)
        weight = torch.where(valid, weight, torch.zeros_like(weight))

        witness_count += valid.float()
        fused_logit += weight * sampled_logit

    near_boundary = torch.abs(ref_logit) < epsilon
    apply_fusion = (witness_count >= min_witnesses) & near_boundary
    fused_logit = torch.where(apply_fusion, fused_logit, ref_logit)

    return fused_logit