import numpy as np
import cv2
from utils.camera import *

def compute_normal_map(depth_map, fx, fy, cx, cy, step=2, relative_depth_threshold=0.05, eps=1e-8):
    depth = depth_map.astype(np.float32)
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    X = (uu - cx) * depth / fx
    Y = (vv - cy) * depth / fy
    points = np.stack([X, Y, depth], axis=-1)

    normals = np.zeros_like(points)
    sl = np.s_[step:-step, step:-step]
    p_left = points[step:-step, :-2 * step]
    p_right = points[step:-step, 2 * step:]
    p_up = points[:-2 * step, step:-step]
    p_down = points[2 * step:, step:-step]
    du = p_right - p_left
    dv = p_down - p_up

    z = depth[step:-step, step:-step]
    z_left = depth[step:-step, :-2 * step]
    z_right = depth[step:-step, 2 * step:]
    z_up = depth[:-2 * step, step:-step]
    z_down = depth[2 * step:, step:-step]
    threshold = relative_depth_threshold * np.maximum(z, eps)

    valid = (
        np.isfinite(z) & (z > eps) & np.isfinite(z_left) & np.isfinite(z_right)
        & np.isfinite(z_up) & np.isfinite(z_down)
        & (np.abs(z_left - z) < threshold) & (np.abs(z_right - z) < threshold)
        & (np.abs(z_up - z) < threshold) & (np.abs(z_down - z) < threshold)
    )

    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid &= norm[..., 0] > eps
    n /= np.maximum(norm, eps)
    n[~valid] = 0.0
    normals[sl] = n
    normals[~(np.isfinite(depth) & (depth > eps))] = 0.0
    return normals

def compute_depth_jump_map(depth_map, window=3, max_depth_jump=50.0):
    gx = np.abs(np.diff(depth_map, axis=1, prepend=depth_map[:, :1]))
    gy = np.abs(np.diff(depth_map, axis=0, prepend=depth_map[:1, :]))
    grad = np.maximum(gx, gy)
    jump = cv2.dilate(grad, np.ones((window, window), np.float32))
    return np.clip(jump / max_depth_jump, 0, 1)


def compute_normal_jump_map(normal_map):
    dot_x = np.abs(np.sum(normal_map * np.roll(normal_map, 1, axis=1), axis=-1))
    dot_y = np.abs(np.sum(normal_map * np.roll(normal_map, 1, axis=0), axis=-1))
    return 1 - np.minimum(dot_x, dot_y)

def geometric_match_score(frame_i, seg_i, frame_j, seg_j, obj_logits, depth_maps,
                           depth_tolerance=20.0, stride=8):
    mask_i = obj_logits[frame_i][seg_i] > 0
    mask_j = obj_logits[frame_j][seg_j] > 0
    depth_i = depth_maps[frame_i]

    ys, xs = np.where(mask_i)
    keep = ((ys % stride) == 0) & ((xs % stride) == 0)
    ys, xs = ys[keep], xs[keep]
    if len(xs) == 0:
        return 0.0

    z = depth_i[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if len(z) == 0:
        return 0.0

    key_i = str(frame_i + 1)
    key_j = str(frame_j + 1)
    fx_i, fy_i, cx_i, cy_i = get_intrinsics(key_i)
    R_i, t_i = get_extrinsics(key_i)
    fx_j, fy_j, cx_j, cy_j = get_intrinsics(key_j)
    R_j, t_j = get_extrinsics(key_j)

    x = (xs.astype(np.float32) - cx_i) * z / fx_i
    y = (ys.astype(np.float32) - cy_i) * z / fy_i
    points_i = np.stack([x, y, z], axis=-1)

    points_world = points_i @ R_i.T + t_i
    points_j = (points_world - t_j) @ R_j

    zj = points_j[:, 2]
    valid = np.isfinite(zj) & (zj > 0)
    u = fx_j * points_j[:, 0] / (zj + 1e-8) + cx_j
    v = fy_j * points_j[:, 1] / (zj + 1e-8) + cy_j

    H, W = mask_j.shape
    valid &= (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H))
    if not np.any(valid):
        return 0.0

    u = np.clip(u.astype(np.int32), 0, W - 1)
    v = np.clip(v.astype(np.int32), 0, H - 1)

    depth_j = depth_maps[frame_j][v, u]
    valid &= (np.isfinite(depth_j) & (depth_j > 0) & (np.abs(depth_j - zj) < depth_tolerance))
    if not np.any(valid):
        return 0.0

    inside = mask_j[v, u] & valid
    return float(inside.sum() / (valid.sum() + 1e-6))