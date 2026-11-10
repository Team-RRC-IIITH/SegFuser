import os
import glob
import json
from PIL import Image
from utils.camera import *
from utils.geometry import *

def load_label(idx):
    label_path = os.path.join(LABEL_FOLDER, f"{str(idx + 1).zfill(6)}.png")
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        if np.all(label[..., 0] == label[..., 1]) and np.all(label[..., 1] == label[..., 2]):
            label = label[..., 0]
        else:
            raise ValueError(f"Label {label_path} is RGB with non-identical channels: shape={label.shape}")
    return label

def load_scene(scene_folder):
    global DATA_FOLDER, data, GT_MASK_FOLDER, LABEL_FOLDER, image_paths, frame_indices
    global depth_maps, normal_maps, gt_masks_by_obj

    DATA_FOLDER = scene_folder
    GT_MASK_FOLDER = os.path.join(DATA_FOLDER, "mask")
    LABEL_FOLDER = os.path.join(DATA_FOLDER, "label")

    with open(os.path.join(DATA_FOLDER, "scene_camera.json"), 'r') as file:
        data = json.load(file)

    num_images = 52
    image_paths = [os.path.join(DATA_FOLDER, f"rgb/{str(i+1).zfill(6)}.png") for i in range(num_images)]
    frame_indices = list(range(0,52,4))

    depth_maps = {}
    for idx in frame_indices:
        depth_path = os.path.join(DATA_FOLDER, f"depth/{str(idx+1).zfill(6)}.png")
        depth_png = np.array(Image.open(depth_path)).astype(np.float32)
        depth_maps[idx] = depth_png * data[str(idx + 1)]['depth_scale']

    normal_maps = {}
    for idx in frame_indices:
        fx, fy, cx, cy = get_intrinsics(str(idx + 1))
        normal_maps[idx] = compute_normal_map(depth_maps[idx], fx, fy, cx, cy)

    gt_masks_by_obj = {}
    for idx in frame_indices:
        frame_id = str(idx + 1).zfill(6)
        pattern = os.path.join(GT_MASK_FOLDER, f"{frame_id}_*.png")
        mask_paths = sorted(glob.glob(pattern))
        gt_masks_by_obj[idx] = {}
        for mask_path in mask_paths:
            obj_id = os.path.basename(mask_path).split("_")[1].split(".")[0]
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            gt_masks_by_obj[idx][obj_id] = mask > 0