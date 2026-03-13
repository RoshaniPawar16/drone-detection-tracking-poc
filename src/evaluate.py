"""
evaluate.py
-----------
Evaluation module for the C-UAS detection and tracking pipeline.

Loads ground-truth annotations and model predictions in YOLO format,
computes standard detection metrics (Precision, Recall, F1), and reports
average inference throughput (FPS).

YOLO annotation format (per .txt file, one row per object):
    <class_id> <x_centre> <y_centre> <width> <height>
    All values normalised to [0, 1] relative to image dimensions.

Metrics are saved to outputs/evaluation_results.csv for inclusion in
technical reporting and comparison across model checkpoints.

Usage:
    python src/evaluate.py \
        --gt_dir   data/ground_truth/ \
        --pred_dir outputs/predictions/ \
        --fps      28.3
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate drone detection predictions against ground truth."
    )
    parser.add_argument(
        "--gt_dir", type=str, required=True,
        help="Directory containing ground-truth YOLO .txt annotation files."
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True,
        help="Directory containing predicted YOLO .txt annotation files "
             "(one file per frame, matching ground-truth filenames)."
    )
    parser.add_argument(
        "--iou_thresh", type=float, default=0.5,
        help="IoU threshold for a prediction to count as a true positive. "
             "Default: 0.5 (standard COCO metric threshold)."
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Average inference FPS measured during detect_track.py run. "
             "If omitted, FPS column will show N/A."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# YOLO annotation I/O
# ---------------------------------------------------------------------------

def load_yolo_annotations(txt_path: str) -> np.ndarray:
    """
    Load a single YOLO-format annotation file.

    Returns an (N, 5) array of [class_id, cx, cy, w, h] rows.
    Returns an empty array if the file is absent or empty (no objects in frame).
    """
    if not os.path.isfile(txt_path):
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                rows.append([float(p) for p in parts])

    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


# ---------------------------------------------------------------------------
# IoU calculation
# ---------------------------------------------------------------------------

def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two YOLO-format boxes.

    Both boxes are [cx, cy, w, h] (normalised).
    Converted internally to [x1, y1, x2, y2] for intersection calculation.

    IoU is the primary spatial matching criterion used in standard object
    detection benchmarks (PASCAL VOC, COCO). In a C-UAS context, a threshold
    of 0.5 is standard; tighter thresholds (0.75) penalise poor localisation.
    """
    # Convert centre-format to corner-format
    def to_corners(b):
        cx, cy, w, h = b[0], b[1], b[2], b[3]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax1, ay1, ax2, ay2 = to_corners(box_a)
    bx1, by1, bx2, by2 = to_corners(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ---------------------------------------------------------------------------
# Per-frame matching
# ---------------------------------------------------------------------------

def match_frame(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_thresh: float,
) -> tuple[int, int, int]:
    """
    Match predictions to ground-truth boxes for a single frame.

    Uses a greedy matching strategy (highest IoU first), consistent with
    the PASCAL VOC evaluation protocol.

    Returns:
        tp (int): True positives  — predicted box matched a GT box at >= iou_thresh
        fp (int): False positives — predicted box had no matching GT box
        fn (int): False negatives — GT box was not matched by any prediction
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    gt_matched   = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)

    # Build IoU matrix: shape (n_pred, n_gt)
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = box_iou(pb[1:], gb[1:])  # columns 1-4 are box coords

    # Greedily assign highest-IoU pairs
    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_thresh:
            break
        pi, gi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        pred_matched[pi] = True
        gt_matched[gi]   = True
        # Prevent re-use of matched rows/columns
        iou_matrix[pi, :] = -1
        iou_matrix[:, gi] = -1

    tp = sum(pred_matched)
    fp = sum(not m for m in pred_matched)
    fn = sum(not m for m in gt_matched)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate(gt_dir: str, pred_dir: str, iou_thresh: float) -> dict:
    """
    Evaluate all frames in gt_dir against corresponding predictions in pred_dir.

    Discovers ground-truth files by globbing *.txt in gt_dir, then looks for
    matching prediction files by filename. Missing prediction files are treated
    as empty (all ground-truth boxes become false negatives).

    Returns a dict with: precision, recall, f1, tp, fp, fn, n_frames.
    """
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

    if not gt_files:
        print(f"[Warning] No ground-truth .txt files found in: {gt_dir}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "n_frames": 0}

    total_tp, total_fp, total_fn = 0, 0, 0

    for gt_path in gt_files:
        filename  = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)

        gt_boxes   = load_yolo_annotations(gt_path)
        pred_boxes = load_yolo_annotations(pred_path)

        tp, fp, fn = match_frame(gt_boxes, pred_boxes, iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro-level precision / recall / F1 from aggregated TP/FP/FN
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        total_tp,
        "fp":        total_fp,
        "fn":        total_fn,
        "n_frames":  len(gt_files),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(metrics: dict, fps: float | None) -> None:
    """
    Print a formatted results table to the console.

    This mirrors the tabular format expected in technical reports for C-UAS
    system performance documentation.
    """
    sep = "+" + "-" * 22 + "+" + "-" * 12 + "+"
    print("\n" + sep)
    print(f"| {'Metric':<20} | {'Value':>10} |")
    print(sep)
    print(f"| {'Frames evaluated':<20} | {metrics['n_frames']:>10} |")
    print(f"| {'True Positives':<20} | {metrics['tp']:>10} |")
    print(f"| {'False Positives':<20} | {metrics['fp']:>10} |")
    print(f"| {'False Negatives':<20} | {metrics['fn']:>10} |")
    print(sep)
    print(f"| {'Precision':<20} | {metrics['precision']:>10.4f} |")
    print(f"| {'Recall':<20} | {metrics['recall']:>10.4f} |")
    print(f"| {'F1 Score':<20} | {metrics['f1']:>10.4f} |")
    fps_str = f"{fps:.2f}" if fps is not None else "N/A"
    print(f"| {'Average FPS':<20} | {fps_str:>10} |")
    print(sep + "\n")


def save_results_csv(metrics: dict, fps: float | None, out_path: str) -> None:
    """
    Persist evaluation results to CSV for reproducibility and experiment tracking.

    In a full KTP research context, this CSV would feed into a results
    dashboard comparing multiple model checkpoints over time.
    """
    os.makedirs(os.path.dirname(out_path) or "outputs", exist_ok=True)
    row = {
        "n_frames":  metrics["n_frames"],
        "tp":        metrics["tp"],
        "fp":        metrics["fp"],
        "fn":        metrics["fn"],
        "precision": round(metrics["precision"], 4),
        "recall":    round(metrics["recall"],    4),
        "f1":        round(metrics["f1"],        4),
        "avg_fps":   fps if fps is not None else "N/A",
    }
    df = pd.DataFrame([row])
    df.to_csv(out_path, index=False)
    print(f"[Output] Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"[Evaluate] Ground truth : {args.gt_dir}")
    print(f"[Evaluate] Predictions  : {args.pred_dir}")
    print(f"[Evaluate] IoU threshold: {args.iou_thresh}")

    metrics = evaluate(args.gt_dir, args.pred_dir, args.iou_thresh)
    print_results_table(metrics, args.fps)
    save_results_csv(metrics, args.fps, "outputs/evaluation_results.csv")


if __name__ == "__main__":
    main()
