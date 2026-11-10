from utils.camera import *
from utils.geometry import *
from utils.semantic import *
from logits import *
from metrics import *
from data.loader import *
from data.save import *
import argparse
import time

SCENES_ROOT = "/scratch2/GraspClutter6D-2/scenes/"
RESULTS_JSON = "scene_metrics_normalized_704.json"

CONFIG_PATH = "/scratch2/Rath/clutt3r-seg/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CHECKPOINT_PATH = "/scratch2/Rath/clutt3r-seg/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "/scratch2/Rath/clutt3r-seg/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

DEVICE = "cuda"

import os
import sys
sys.path.append(os.path.join("../", "Grounded-Segment-Anything"))
sys.path.append(os.path.join("../", "Grounded-Segment-Anything/GroundingDINO"))
sys.path.append(os.path.join("../", "Grounded-Segment-Anything/segment_anything"))

from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict

torch.cuda.empty_cache()
model = load_model(CONFIG_PATH, GD_CHECKPOINT_PATH)
use_sam_hq = True
if use_sam_hq:
    sam_predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE))
else:
    sam_predictor = SamPredictor(sam_model_registry['vit_b'](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE))

from torchvision.ops import box_convert


def build_track_ids(frame_indices, obj_logits, clip_embeddings,
                     clip_threshold=0.65, merge_clip_threshold=0.80,
                     merge_overlap_threshold=0.50, track_threshold=0.55,
                     lambda_clip=0.7, lambda_geom=0.3,
                     max_frame_gap=5, depth_tolerance=20.0):

    tracks = {}
    track_ids = {}
    merged_obj_logits = {}
    merged_clip_embeddings = {}
    active = {}
    next_track_id = 0

    for frame_idx in frame_indices:
        if frame_idx not in obj_logits or frame_idx not in clip_embeddings:
            continue

        logits = obj_logits[frame_idx]
        embeddings = clip_embeddings[frame_idx]
        if len(logits) == 0:
            continue

        embeddings = F.normalize(torch.as_tensor(embeddings).float(), dim=-1)
        n = len(logits)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        masks = [np.asarray(logits[i]) > 0 for i in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                clip_score = torch.dot(embeddings[i], embeddings[j]).item()
                if clip_score < merge_clip_threshold:
                    continue
                inter = np.logical_and(masks[i], masks[j]).sum()
                small = min(masks[i].sum(), masks[j].sum())
                overlap = (inter / (small + 1e-6)) if small > 0 else 0.0
                if overlap >= merge_overlap_threshold:
                    union(i, j)

        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        merged_logits = []
        merged_embeddings = []
        original_to_merged = {}

        for merged_idx, members in enumerate(groups.values()):
            stack = np.stack([logits[i] for i in members], axis=0)
            merged_logit = np.max(stack, axis=0)
            emb = F.normalize(embeddings[members].mean(dim=0), dim=0)
            merged_logits.append(merged_logit)
            merged_embeddings.append(emb)
            for i in members:
                original_to_merged[i] = merged_idx

        merged_obj_logits[frame_idx] = merged_logits
        merged_clip_embeddings[frame_idx] = torch.stack(merged_embeddings)

        n_seg = len(merged_logits)
        track_ids[frame_idx] = {}
        current_embeddings = merged_clip_embeddings[frame_idx]

        candidates = [tid for tid, info in active.items() if frame_idx - info["frame"] <= max_frame_gap]

        if not candidates:
            for seg_idx in range(n_seg):
                tracks[next_track_id] = {frame_idx: seg_idx}
                track_ids[frame_idx][seg_idx] = next_track_id
                active[next_track_id] = {
                    "frame": frame_idx, "segment": seg_idx,
                    "embedding": current_embeddings[seg_idx].detach().cpu(),
                }
                next_track_id += 1
            continue

        pairs = []
        for seg_idx in range(n_seg):
            for track_id in candidates:
                info = active[track_id]
                track_emb = info["embedding"].to(current_embeddings.device)
                clip_score = torch.dot(track_emb, current_embeddings[seg_idx]).item()
                if clip_score < clip_threshold:
                    continue
                geom_score = geometric_match_score(
                    info["frame"], info["segment"], frame_idx, seg_idx,
                    merged_obj_logits, depth_maps, depth_tolerance=depth_tolerance,
                )
                score = lambda_clip * clip_score + lambda_geom * geom_score
                pairs.append((score, track_id, seg_idx, clip_score, geom_score))

        pairs.sort(reverse=True)
        used_tracks = set()
        used_segments = set()

        for score, track_id, seg_idx, clip_score, geom_score in pairs:
            if track_id in used_tracks or seg_idx in used_segments or score < track_threshold:
                continue
            tracks[track_id][frame_idx] = seg_idx
            track_ids[frame_idx][seg_idx] = track_id
            old_emb = active[track_id]["embedding"].to(current_embeddings.device)
            new_emb = current_embeddings[seg_idx]
            updated_emb = F.normalize(0.8 * old_emb + 0.2 * new_emb, dim=-1).detach().cpu()
            active[track_id] = {"frame": frame_idx, "segment": seg_idx, "embedding": updated_emb}
            used_tracks.add(track_id)
            used_segments.add(seg_idx)

        for seg_idx in range(n_seg):
            if seg_idx in used_segments:
                continue
            tracks[next_track_id] = {frame_idx: seg_idx}
            track_ids[frame_idx][seg_idx] = next_track_id
            active[next_track_id] = {
                "frame": frame_idx, "segment": seg_idx,
                "embedding": current_embeddings[seg_idx].detach().cpu(),
            }
            next_track_id += 1

    return tracks, track_ids, merged_obj_logits, merged_clip_embeddings

def run_detection(idx):
    image_source_i, image_i = load_image(image_paths[idx])
    boxes_i, _, _ = predict(
        model=model, image=image_i, caption="object",
        box_threshold=0.2, text_threshold=0.3, device=DEVICE,
    )

    h_i, w_i, _ = image_source_i.shape
    boxes_i = boxes_i * torch.tensor([w_i, h_i, w_i, h_i])
    filt_t = torch.full((boxes_i.shape[0],), 0.3 * w_i)
    filt_i = boxes_i[:, 2] < filt_t
    boxes_i = boxes_i[filt_i]

    sam_predictor.set_image(image_source_i)
    xyxy_boxes_i = box_convert(boxes=boxes_i, in_fmt="cxcywh", out_fmt="xyxy")
    transformed_boxes_i = sam_predictor.transform.apply_boxes_torch(
        xyxy_boxes_i, image_source_i.shape[:2],
    ).to(DEVICE)

    masks_i, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=transformed_boxes_i,
        multimask_output=False, return_logits=True,
    )

    sam_predictor.reset_image()
    torch.cuda.empty_cache()

    logits = [masks_i[k, 0].detach().cpu().numpy() for k in range(masks_i.shape[0])]
    boxes = [box.detach().cpu().numpy() for box in xyxy_boxes_i]

    return logits, boxes, image_source_i

def run_scene_pipeline(depth_maps, normal_maps, frame_indices):
    depth_jump_maps = {}
    normal_jump_maps = {}
    for idx in frame_indices:
        depth_jump_maps[idx] = compute_depth_jump_map(depth_maps[idx])
        normal_jump_maps[idx] = compute_normal_jump_map(normal_maps[idx])

    depth_gpu = {i: torch.from_numpy(depth_maps[i]).float().cuda() for i in frame_indices}
    normal_gpu = {i: torch.from_numpy(normal_maps[i]).float().cuda() for i in frame_indices}
    depth_jump_gpu = {i: torch.from_numpy(depth_jump_maps[i]).float().cuda() for i in frame_indices}
    normal_jump_gpu = {i: torch.from_numpy(normal_jump_maps[i]).float().cuda() for i in frame_indices}

    obj_logits = {}
    obj_boxes = {}
    clip_embeddings = {}
    images = {}
    for idx in frame_indices:
        logits_i, boxes_i, image_i = run_detection(idx)
        obj_logits[idx] = logits_i
        obj_boxes[idx] = boxes_i
        images[idx] = image_i
        clip_embeddings[idx] = compute_clip_embeddings(image_i, boxes_i).detach()

    tracks, track_ids, obj_logits_merged, clip_embeddings_merged = build_track_ids(
        frame_indices, obj_logits, clip_embeddings,
        clip_threshold=0.65, merge_clip_threshold=0.80, merge_overlap_threshold=0.48,
        track_threshold=0.5, lambda_clip=0.7, lambda_geom=0.3, max_frame_gap=20,
        depth_tolerance=20.0,
    )

    voxel_size = 5.0
    depth_tolerance = 20.0

    track_ids_all = sorted({
        track_id for frame_dict in track_ids.values() for track_id in frame_dict.values()
    })

    start = time.time()
    pred_clouds = {}
    gt_clouds = {}

    for track_id in track_ids_all:
        track = {
            frame_idx: seg_idx
            for frame_idx in track_ids
            for seg_idx, tid in track_ids[frame_idx].items()
            if tid == track_id
        }
        if not track:
            continue

        pred_points_list = []
        for idx, seg_idx in track.items():
            fused_logit = fuse_logits_cuda(
                ref_idx=idx,
                frame_idxs=list(track.keys()),
                track_ids=track,
                obj_logits=obj_logits_merged,
                depth_gpu=depth_gpu,
                normal_gpu=normal_gpu,
                depth_jump_gpu=depth_jump_gpu,
                normal_jump_gpu=normal_jump_gpu,
                depth_tolerance=depth_tolerance,
                min_witnesses=3,
                max_contribution=1.0,
                epsilon=0.10,
            )
            fused_mask = (fused_logit > 0).cpu().numpy()
            depth = depth_maps[idx]
            key = str(idx + 1)
            fx, fy, cx, cy = get_intrinsics(key)
            Rw2c, tw2c = get_extrinsics(key)

            if fused_mask.shape != depth.shape:
                fused_mask = cv2.resize(
                    fused_mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            pts = depth_mask_to_world_points(depth, fused_mask, fx, fy, cx, cy, Rw2c, tw2c)
            if len(pts) > 0:
                pred_points_list.append(pts)

        pred_clouds[track_id] = (
            np.concatenate(pred_points_list, axis=0) if pred_points_list else np.empty((0, 3), dtype=np.float32)
        )

    for idx in frame_indices:
        label = load_label(idx)
        depth = depth_maps[idx]
        if label.shape != depth.shape:
            label = cv2.resize(label, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        key = str(idx + 1)
        fx, fy, cx, cy = get_intrinsics(key)
        Rw2c, tw2c = get_extrinsics(key)

        for obj_id in np.unique(label):
            obj_id = int(obj_id)
            if obj_id == 0:
                continue
            gt_mask = label == obj_id
            pts = depth_mask_to_world_points(depth, gt_mask, fx, fy, cx, cy, Rw2c, tw2c)
            if len(pts) > 0:
                gt_clouds.setdefault(obj_id, []).append(pts)

    gt_clouds = {obj_id: np.concatenate(points, axis=0) for obj_id, points in gt_clouds.items()}

    pred_voxels = {track_id: voxelize_points(points, voxel_size) for track_id, points in pred_clouds.items()}
    gt_voxels = {obj_id: voxelize_points(points, voxel_size) for obj_id, points in gt_clouds.items()}
    gt_object_ids = sorted(gt_voxels.keys())

    iou_matrix = np.zeros((len(track_ids_all), len(gt_object_ids)), dtype=np.float32)
    for i, track_id in enumerate(track_ids_all):
        pred_v = pred_voxels[track_id]
        for j, obj_id in enumerate(gt_object_ids):
            gt_v = gt_voxels[obj_id]
            intersection = pred_v & gt_v
            union = pred_v | gt_v
            iou_matrix[i, j] = len(intersection) / len(union) if union else 0.0

    track_scores = {track_id: len(track) for track_id, track in tracks.items()}

    precision, recall, f1, matched_results = compute_detection_metrics(
        iou_matrix,
        track_ids_all,
        track_scores,
        gt_object_ids,
        iou_threshold=0.50,
    )

    matched_ious = [x["iou"] for x in matched_results.values()]
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    ap_metrics = compute_ap_metrics(
        iou_matrix,
        track_ids_all,
        gt_object_ids,
        track_scores,
    )

    elapsed = time.time() - start

    scene_metrics = {
        "num_gt_objects": len(gt_object_ids),
        "num_pred_tracks": len(track_ids_all),
        "mean_iou": mean_iou,
        "mean_precision": precision,
        "mean_recall": recall,
        "mean_f1": f1,
        "AP@25": ap_metrics["AP@25"],
        "AP@50": ap_metrics["AP@50"],
        "AP": ap_metrics["AP"],
        "time_seconds": elapsed,
    }

    return scene_metrics

def main():
    global DATA_FOLDER
    global OUTPUT_PATH
    global data, image_paths, frame_indices
    global depth_maps, normal_maps, gt_masks_by_obj
    global depth_gpu, normal_gpu
    global depth_jump_gpu, normal_jump_gpu
    global obj_logits, obj_boxes, clip_embeddings, images
    global tracks, track_ids
    global obj_logits_merged, clip_embeddings_merged

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        required=True,
        help="Path to a single GraspClutter6D scene",
    )
    parser.add_argument(
        "--output",
        default="./scene_output",
        help="Directory for saved point cloud and masks",
    )
    args = parser.parse_args()

    DATA_FOLDER = os.path.abspath(args.scene)
    OUTPUT_PATH = os.path.abspath(args.output)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 60)
    print(f"Scene: {DATA_FOLDER}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    load_scene(DATA_FOLDER)

    scene_metrics = run_scene_pipeline()

    save_scene_point_cloud(OUTPUT_PATH)
    save_original_sam_masks(OUTPUT_PATH)
    save_refined_masks_and_pointcloud(OUTPUT_PATH, alpha=0.35)

    with open(
        os.path.join(OUTPUT_PATH, "metrics.json"),
        "w",
    ) as f:
        json.dump(
            scene_metrics,
            f,
            indent=2,
            cls=NumpyJSONEncoder,
        )

    del depth_gpu
    del normal_gpu
    del depth_jump_gpu
    del normal_jump_gpu
    torch.cuda.empty_cache()

    print("\nDone.")
