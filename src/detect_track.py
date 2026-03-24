"""
detect_track.py
---------------
Core detection and tracking pipeline for Counter-UAS (C-UAS) PoC.

Pipeline:
    Input Video → Frame Extraction → YOLOv8 Detection → ByteTrack Tracking
    → Annotated Frame → Output Video

Model:  YOLOv8n (Ultralytics) — lightweight, real-time capable.
Tracker: ByteTrack via the `supervision` library — robust to occlusion and
         missed detections, which is critical when drones briefly pass behind
         obstacles or are lost in sky clutter.

Usage:
    python src/detect_track.py --input data/sample_clips/flight.mp4 \
                                --output outputs/tracked.mp4 \
                                --conf 0.35 --show
"""

import argparse
import os
import time

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> str:
    """
    Automatically select the best available compute device.

    In a C-UAS deployment context, edge hardware (e.g. Jetson Orin) will
    typically expose CUDA. Development machines may use MPS (Apple Silicon)
    or fall back to CPU. The pipeline must run on all three without
    code changes.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[Device] Using: {device.upper()}")
    return device


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 + ByteTrack drone detection and tracking pipeline."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input video file (e.g. data/sample_clips/drone.mp4)."
    )
    parser.add_argument(
        "--output", type=str, default="outputs/output.mp4",
        help="Path to save annotated output video. Default: outputs/output.mp4"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold (0.0–1.0). Lower values increase "
             "recall at the cost of more false positives. Default: 0.35"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display a live preview window during processing."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def load_model(device: str) -> YOLO:
    """
    Load the fine-tuned VisDrone model if available, otherwise fall back to
    pretrained YOLOv8n (COCO).

    For operational C-UAS use, this model should be fine-tuned on a labelled
    UAS dataset. The 'n' (nano) variant is selected here for real-time
    feasibility on edge hardware; 's' or 'm' variants offer higher accuracy
    at the cost of throughput.
    """
    model_path = 'runs/detect/outputs/training/drone_finetune_1k/weights/best.pt'
    if not os.path.exists(model_path):
        model_path = 'yolov8n.pt'
    print(f'Loading model: {model_path}')
    model = YOLO(model_path)
    model.to(device)
    print("[Model] Loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Tracker initialisation
# ---------------------------------------------------------------------------

def build_tracker() -> sv.ByteTrack:
    """
    Initialise ByteTrack with supervision's default parameters.

    ByteTrack is preferred over SORT/DeepSORT for this application because:
    - It associates both high- and low-confidence detections, reducing ID
      switches when a drone partially enters cloud cover or background clutter.
    - It does not require a re-identification network, keeping inference
      latency low — important for real-time C-UAS alerting.
    """
    tracker = sv.ByteTrack()
    return tracker


# ---------------------------------------------------------------------------
# Annotators
# ---------------------------------------------------------------------------

def build_annotators():
    """
    Build supervision annotators for bounding boxes and track labels.

    Colours are assigned per track ID so operators can visually follow
    individual UAS targets across frames.
    """
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        color_lookup=sv.ColorLookup.TRACK,
    )
    return box_annotator, label_annotator


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_video(
    model: YOLO,
    tracker: sv.ByteTrack,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    args: argparse.Namespace,
    device: str,
) -> None:
    """
    Read video frame by frame, run detection + tracking, write annotated output.

    Each frame passes through:
      1. YOLO inference  — produces bounding boxes, class IDs, confidences
      2. ByteTrack update — assigns persistent track IDs across frames
      3. Annotation       — overlays boxes, track IDs, and confidence scores
      4. Write / display  — saves to output file, optionally shows preview
    """

    # --- Open input video ---
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Input]  {args.input} | {width}x{height} @ {fps_in:.1f} FPS | "
          f"{total_frames} frames")

    # --- Prepare output directory and video writer ---
    os.makedirs(os.path.dirname(args.output) or "outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))

    frame_count = 0
    total_time  = 0.0

    print(f"[Config] Confidence threshold: {args.conf} | Device: {device}")
    print("[Processing] Starting frame-by-frame inference...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream

        t_start = time.perf_counter()

        # ---- 1. YOLOv8 inference ----------------------------------------
        # `verbose=False` suppresses per-frame console noise — important when
        # processing thousands of frames in a C-UAS surveillance session.
        results = model.predict(
            source=frame,
            conf=args.conf,
            device=device,
            verbose=False,
        )
        result = results[0]

        # ---- 2. Convert to supervision Detections -----------------------
        # supervision.Detections is the common format consumed by ByteTracker
        # and the annotators, decoupling detector and tracker implementations.
        detections = sv.Detections.from_ultralytics(result)

        # ---- 3. ByteTrack update ----------------------------------------
        # tracker.update_with_detections returns the same Detections object
        # with tracker_id populated for each surviving track.
        detections = tracker.update_with_detections(detections)

        # ---- 4. Build per-detection labels (Track ID + confidence) -------
        # Operators need both the persistent ID (to follow a specific drone
        # across the scene) and the confidence score (to assess alert quality).
        labels = []
        for i in range(len(detections)):
            tid  = detections.tracker_id[i] if detections.tracker_id is not None else "?"
            conf = detections.confidence[i] if detections.confidence is not None else 0.0
            cls  = result.names[int(detections.class_id[i])] if detections.class_id is not None else "UAS"
            labels.append(f"ID:{tid} {cls} {conf:.2f}")

        # ---- 5. Annotate frame -------------------------------------------
        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
        )
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        total_time  += elapsed
        frame_count += 1
        inst_fps     = 1.0 / elapsed if elapsed > 0 else 0.0

        # Overlay FPS counter on frame for real-time feasibility assessment
        cv2.putText(
            annotated,
            f"FPS: {inst_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        writer.write(annotated)

        # ---- 6. Optional live preview ------------------------------------
        if args.show:
            cv2.imshow("C-UAS Detection & Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Preview] User quit early.")
                break

        # Progress report every 50 frames
        if frame_count % 50 == 0:
            avg_fps = frame_count / total_time if total_time > 0 else 0.0
            n_det   = len(detections)
            print(f"  Frame {frame_count:>5}/{total_frames} | "
                  f"Avg FPS: {avg_fps:5.1f} | Detections this frame: {n_det}")

    # --- Cleanup ---
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # --- Final report ---
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    print(f"\n[Done] Processed {frame_count} frames.")
    print(f"[Performance] Average FPS: {avg_fps:.2f}")
    print(f"[Output] Annotated video saved to: {args.output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = select_device()
    model  = load_model(device)
    tracker, box_ann, label_ann = (
        build_tracker(),
        *build_annotators(),
    )
    process_video(model, tracker, box_ann, label_ann, args, device)


if __name__ == "__main__":
    main()
