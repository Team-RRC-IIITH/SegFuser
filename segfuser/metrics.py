import numpy as np

def compute_ap_at_iou(iou_matrix, pred_ids, gt_ids, pred_scores, iou_threshold,):
    num_preds = len(pred_ids)
    num_gts = len(gt_ids)

    if num_gts == 0:
        return 0.0

    if num_preds == 0:
        return 0.0

    order = sorted(range(num_preds),key=lambda i: pred_scores[pred_ids[i]],reverse=True,)

    matched_gt = set()

    tp = np.zeros(num_preds, dtype=np.float32)
    fp = np.zeros(num_preds, dtype=np.float32)

    for k, p_idx in enumerate(order):
        candidate_ious = iou_matrix[p_idx]
        eligible_gt = [gt_idx for gt_idx in range(num_gts) if gt_idx not in matched_gt and candidate_ious[gt_idx] >= iou_threshold]

        if eligible_gt:
            # Among eligible GTs, choose the one with highest IoU.
            gt_idx = max(eligible_gt, key=lambda g: candidate_ious[g] )

            tp[k] = 1.0
            matched_gt.add(gt_idx)

        else:
            fp[k] = 1.0

    # Cumulative TP/FP.
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / float(num_gts)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    # 101-point interpolated precision.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    recall_points = np.linspace(0.0, 1.0, 101)

    ap = 0.0

    for r in recall_points:
        valid = precision[recall >= r]

        if len(valid) > 0:
            ap += np.max(valid)

    return float(ap / 101.0)


def compute_ap_metrics(iou_matrix, pred_ids, gt_ids, pred_scores):
    ap25 = compute_ap_at_iou(iou_matrix, pred_ids, gt_ids, pred_scores, 0.25)
    ap50 = compute_ap_at_iou(iou_matrix, pred_ids, gt_ids, pred_scores, 0.50)
    aps = [compute_ap_at_iou(iou_matrix, pred_ids, gt_ids, pred_scores, t) for t in np.arange(0.50, 0.96, 0.05)]
    return {"AP@25": ap25, "AP@50": ap50, "AP": float(np.mean(aps))}

def compute_detection_metrics(iou_matrix, pred_ids, pred_scores, gt_ids, iou_threshold=0.50):
    num_preds = len(pred_ids)
    num_gts = len(gt_ids)

    if num_preds == 0 and num_gts == 0:
        return 1.0, 1.0, 1.0, {}

    if num_preds == 0:
        return 0.0, 0.0, 0.0, {}

    if num_gts == 0:
        return 0.0, 0.0, 0.0, {}

    order = sorted(range(num_preds), key=lambda i: pred_scores[pred_ids[i]], reverse=True)

    matched_gt = set()
    matched_results = {}
    tp = 0

    for p_idx in order:
        candidate_ious = iou_matrix[p_idx]
        eligible_gt = [gt_idx for gt_idx in range(num_gts) if gt_idx not in matched_gt and candidate_ious[gt_idx] >= iou_threshold]

        if not eligible_gt:
            continue

        gt_idx = max(eligible_gt, key=lambda g: candidate_ious[g])

        matched_gt.add(gt_idx)
        tp += 1

        track_id = pred_ids[p_idx]
        gt_id = gt_ids[gt_idx]

        matched_results[track_id] = {"gt_id": gt_id, "iou": float(iou_matrix[p_idx, gt_idx])}

    fp = num_preds - tp
    fn = num_gts - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(precision), float(recall), float(f1), matched_results