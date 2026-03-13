"""
visualise.py
------------
Trajectory visualisation for the C-UAS detection and tracking pipeline.

Takes an annotated output video (produced by detect_track.py) or a directory
of annotated image frames, re-renders each track with a unique colour, and
produces a static trajectory plot — a top-down 2D "flight path" overlay
showing every tracked object's motion across the scene.

Trajectory plots are valuable in C-UAS after-action review: they let operators
understand the flight path of each detected UAS without scrubbing through video
frame by frame.

Usage:
    python src/visualise.py \
        --input  outputs/tracked_output.mp4 \
        --output outputs/trajectory_plot.png
"""

import argparse
import os
from collections import defaultdict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import supervision as sv


# Use a non-interactive backend so the script works headlessly on servers
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise drone tracks and generate trajectory plots."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to annotated output video (.mp4) or directory of frames."
    )
    parser.add_argument(
        "--output", type=str, default="outputs/trajectory_plot.png",
        help="Path to save the trajectory plot PNG. Default: outputs/trajectory_plot.png"
    )
    parser.add_argument(
        "--max_frames", type=int, default=0,
        help="Maximum frames to process (0 = all). Useful for large files."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Predefined high-contrast colours for track IDs.
# In a C-UAS display system, colour consistency per track ID is essential so
# operators can immediately associate a colour with a specific drone.
TRACK_COLOURS = [
    (255,  60,  60),   # red
    (60,  180, 255),   # sky blue
    (60,  220,  60),   # green
    (255, 200,  30),   # amber
    (200,  60, 255),   # purple
    (255, 140,  30),   # orange
    (30,  220, 200),   # cyan
    (255, 100, 180),   # pink
    (100, 100, 255),   # blue
    (180, 255, 100),   # lime
]


def get_colour(track_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given track ID."""
    return TRACK_COLOURS[int(track_id) % len(TRACK_COLOURS)]


def get_colour_rgb(track_id: int) -> tuple[float, float, float]:
    """Return the same colour in [0,1] RGB for matplotlib."""
    b, g, r = get_colour(track_id)
    return r / 255.0, g / 255.0, b / 255.0


# ---------------------------------------------------------------------------
# Frame source abstraction
# ---------------------------------------------------------------------------

def iter_frames(source: str, max_frames: int):
    """
    Yield frames from either a video file or an image directory.

    Abstracts over the two possible input types so the rest of the pipeline
    does not need to branch on input format.
    """
    if os.path.isdir(source):
        # Collect image files sorted by name (assumes numeric/timestamp naming)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(
            f for f in os.listdir(source)
            if os.path.splitext(f)[1].lower() in exts
        )
        for i, fname in enumerate(files):
            if max_frames and i >= max_frames:
                break
            frame = cv2.imread(os.path.join(source, fname))
            if frame is not None:
                yield frame
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {source}")
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and i >= max_frames:
                break
            yield frame
            i += 1
        cap.release()


# ---------------------------------------------------------------------------
# Track history extraction
# ---------------------------------------------------------------------------

def extract_track_history(source: str, max_frames: int) -> dict[int, list[tuple[float, float]]]:
    """
    Re-run supervision's ByteTracker on the raw annotated video to recover
    the (cx, cy) centroid history for each track ID.

    NOTE: Because detect_track.py embeds track IDs as text overlays rather
    than in a sidecar file, we re-run the tracker here to reconstruct
    trajectories. In a production system, track history would be serialised
    (e.g. as JSON or CSV) by detect_track.py and loaded directly.

    Returns:
        dict mapping track_id → list of (cx_normalised, cy_normalised) tuples
    """
    from ultralytics import YOLO
    import torch

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model   = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    history: dict[int, list[tuple[float, float]]] = defaultdict(list)

    frame_w, frame_h = None, None

    for frame in iter_frames(source, max_frames):
        if frame_w is None:
            frame_h, frame_w = frame.shape[:2]

        results    = model.predict(source=frame, conf=0.25, device=device, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is None:
            continue

        for i, tid in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = detections.xyxy[i]
            cx = ((x1 + x2) / 2) / frame_w   # normalise to [0,1]
            cy = ((y1 + y2) / 2) / frame_h
            history[int(tid)].append((float(cx), float(cy)))

    return history, frame_w, frame_h


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectories(
    history: dict[int, list[tuple[float, float]]],
    output_path: str,
    frame_w: int | None,
    frame_h: int | None,
) -> None:
    """
    Render all tracked UAS trajectories on a single matplotlib figure.

    Each track ID gets a unique colour. The path is drawn as a continuous line
    with a filled circle at the final position, mimicking a radar/EO track
    display. This format is immediately interpretable by C-UAS operators.
    """
    os.makedirs(os.path.dirname(output_path) or "outputs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Set axis limits in pixel space if dimensions are available
    ax.set_xlim(0, frame_w if frame_w else 1)
    ax.set_ylim(frame_h if frame_h else 1, 0)   # invert Y so (0,0) is top-left
    ax.set_aspect("equal")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#1a1a2e")

    ax.set_xlabel("X position (pixels)", color="white", fontsize=11)
    ax.set_ylabel("Y position (pixels)", color="white", fontsize=11)
    ax.set_title(
        "UAS Track Trajectories — C-UAS Detection PoC",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    legend_patches = []

    if not history:
        ax.text(
            0.5, 0.5, "No tracks detected",
            ha="center", va="center", color="white", fontsize=14,
            transform=ax.transAxes
        )
    else:
        fw = frame_w if frame_w else 1
        fh = frame_h if frame_h else 1

        for tid, points in history.items():
            colour_rgb = get_colour_rgb(tid)
            xs = [p[0] * fw for p in points]   # denormalise to pixel coords
            ys = [p[1] * fh for p in points]

            # Draw trajectory line
            ax.plot(xs, ys, color=colour_rgb, linewidth=1.5, alpha=0.8)

            # Mark start and end positions
            if len(xs) >= 1:
                ax.plot(xs[0],  ys[0],  "o", color=colour_rgb, markersize=6, alpha=0.5)
                ax.plot(xs[-1], ys[-1], "o", color=colour_rgb, markersize=9)

                # Arrow to show direction of travel at trajectory end
                if len(xs) >= 2:
                    dx = xs[-1] - xs[-2]
                    dy = ys[-1] - ys[-2]
                    ax.annotate(
                        "", xy=(xs[-1], ys[-1]),
                        xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(
                            arrowstyle="->", color=colour_rgb, lw=1.5
                        ),
                    )

            legend_patches.append(
                mpatches.Patch(color=colour_rgb, label=f"Track ID {tid}")
            )

    ax.legend(
        handles=legend_patches,
        loc="upper right",
        facecolor="#222233",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Output] Trajectory plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"[Visualise] Source : {args.input}")
    print(f"[Visualise] Output : {args.output}")

    if not os.path.exists(args.input):
        print(f"[Warning] Input not found: {args.input}")
        print("[Warning] Generating empty trajectory plot.")
        plot_trajectories({}, args.output, frame_w=1280, frame_h=720)
        return

    print("[Visualise] Extracting track histories (re-running tracker)...")
    history, frame_w, frame_h = extract_track_history(args.input, args.max_frames)
    print(f"[Visualise] Found {len(history)} unique tracks.")

    plot_trajectories(history, args.output, frame_w, frame_h)


if __name__ == "__main__":
    main()
