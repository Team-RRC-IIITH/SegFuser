import numpy as np

def cam2w(points, rotation, translation):
    return points @ rotation + translation[None, :]

def w2cam(point, rotation, translation):
    point = point[..., np.newaxis]
    transformed_point = rotation @ point + translation[..., np.newaxis]
    return transformed_point[:3].reshape(3)

def get_intrinsics(data, key):
    fx = data[key]['cam_K'][0][0]
    fy = data[key]['cam_K'][1][1]
    cx = data[key]['cam_K'][0][2]
    cy = data[key]['cam_K'][1][2]
    return fx, fy, cx, cy

def get_extrinsics(data, key):
    R = np.array(data[key]['cam_R_w2c']).reshape((3, 3))
    t = np.array(data[key]['cam_t_w2c'])
    return R, t

def depth_mask_to_world_points(depth_map, mask, fx, fy, cx, cy, Rw2c, tw2c):
    valid = mask & np.isfinite(depth_map) & (depth_map > 0)
    vs, us = np.where(valid)
    if len(us) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth_map[vs, us]
    x = (us.astype(np.float32) - cx) * z / fx
    y = (vs.astype(np.float32) - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=-1)
    points_world = points_cam @ Rw2c.T + tw2c
    return points_world.astype(np.float32)


def voxelize_points(points, voxel_size):
    if len(points) == 0:
        return set()
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    voxel_coords = np.unique(voxel_coords, axis=0)
    return {tuple(v) for v in voxel_coords}