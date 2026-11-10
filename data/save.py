import os
import json
import open3d as o3d
import cv2

from PIL import Image
from utils.camera import *
from segfuser.logits import *

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_scene_point_cloud(frame_indices, depth_maps, images, output_dir):
    all_points = []
    all_colors = []

    for idx in frame_indices:
        depth = depth_maps[idx]
        image = images[idx]

        key = str(idx + 1)
        fx, fy, cx, cy = get_intrinsics(key)
        Rw2c, tw2c = get_extrinsics(key)

        mask = np.isfinite(depth) & (depth > 0)
        points = depth_mask_to_world_points(depth, mask, fx, fy, cx, cy, Rw2c, tw2c)

        if len(points) == 0:
            continue

        all_points.append(points / 1000.0)
        all_colors.append(image[mask].astype(np.uint8))

    if not all_points:
        return

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    path = os.path.join(output_dir, "scene_point_cloud.ply")

    o3d.io.write_point_cloud(path, pcd)

    print(f"Saved point cloud: {path} " f"({len(points):,} points)" )


def save_original_sam_masks(frame_indices, obj_logits, output_dir):
    mask_dir = os.path.join(output_dir, "original_sam_masks")
    os.makedirs(mask_dir, exist_ok=True)

    metadata = {}

    for idx in frame_indices:
        logits = obj_logits.get(idx, [])

        if not logits:
            metadata[str(idx)] = []
            continue

        frame_dir = os.path.join(
            mask_dir,
            f"{idx + 1:06d}",
        )
        os.makedirs(frame_dir, exist_ok=True)

        frame_info = []

        for seg_idx, logit in enumerate(logits):
            logit = np.asarray(logit)

            if logit.ndim != 2:
                continue

            binary = logit > 0
            Image.fromarray((binary.astype(np.uint8) * 255)).save(os.path.join(frame_dir, f"mask_{seg_idx:04d}.png"))
            frame_info.append({"segment": seg_idx, "mask_pixels": int(binary.sum())})

        metadata[str(idx)] = frame_info

    with open(os.path.join(mask_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)

    print(f"Saved original SAM masks: {mask_dir}")

def get_instance_color(instance_id):
    hue = (instance_id * 0.618033988749895) % 1.0
    hsv = np.array([[[hue * 179.0, 220.0, 255.0]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]


def save_refined_masks_and_pointcloud(frame_indices, images, track_ids, obj_logits_merged, depth_maps, depth_gpu, normal_gpu, depth_jump_gpu, normal_jump_gpu, output_dir, alpha=0.35):
    mask_dir = os.path.join(output_dir, "refined_masks")
    os.makedirs(mask_dir, exist_ok=True)

    point_dir = os.path.join(output_dir, "refined_pointcloud")
    os.makedirs(point_dir, exist_ok=True)

    track_ids_all = sorted({track_id for frame_dict in track_ids.values() for track_id in frame_dict.values()})
    instance_colors = {track_id: get_instance_color(track_id) for track_id in track_ids_all}

    metadata = {}
    all_points = []
    all_colors = []

    for idx in frame_indices:
        image = images[idx].copy()

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        overlay = image.copy()
        frame_info = {}

        for track_id in track_ids_all:
            track = {frame_idx: seg_idx for frame_idx in track_ids for seg_idx, tid in track_ids[frame_idx].items() if tid == track_id}

            if idx not in track:
                continue

            seg_idx = track[idx]

            fused_logit = fuse_logits_cuda(
                ref_idx=idx,
                frame_idxs=list(track.keys()),
                track_ids=track,
                obj_logits=obj_logits_merged,
                depth_gpu=depth_gpu,
                normal_gpu=normal_gpu,
                depth_jump_gpu=depth_jump_gpu,
                normal_jump_gpu=normal_jump_gpu,
                depth_tolerance=20.0,
                min_witnesses=3,
                max_contribution=1.0,
                epsilon=0.10,
            )

            mask = (fused_logit > 0).detach().cpu().numpy()

            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask = mask.astype(bool)

            if not np.any(mask):
                continue

            color = instance_colors[track_id]
            overlay[mask] = ((1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color.astype(np.float32)).astype(np.uint8)

            key = str(idx + 1)
            fx, fy, cx, cy = get_intrinsics(key)
            Rw2c, tw2c = get_extrinsics(key)

            points = depth_mask_to_world_points(depth_maps[idx], mask, fx, fy, cx, cy, Rw2c, tw2c)

            if len(points) > 0:
                all_points.append(points)
                all_colors.append(np.repeat(color[None, :],len(points),axis=0,))

            frame_info[str(track_id)] = {"segment": int(seg_idx), "mask_pixels": int(mask.sum()), "color": color.tolist()}

        metadata[str(idx)] = frame_info
        Image.fromarray(overlay).save(os.path.join(mask_dir, f"{idx + 1:06d}.png"))

    with open(os.path.join(mask_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        pointcloud_path = os.path.join( point_dir,"combined_instances.ply")
        o3d.io.write_point_cloud(pointcloud_path, pcd)

        print(f"Saved combined instance point cloud: " f"{pointcloud_path} " f"({len(points):,} points)")
    else:
        print("No refined instance points found.")

    print(
        f"Saved combined refined masks: "
        f"{mask_dir}"
    )

    return pointcloud_path if all_points else None