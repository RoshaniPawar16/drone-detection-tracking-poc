# Drone Detection and Tracking — Proof of Concept

> Built as part of a KTP Associate application to the University of Central Lancashire (UCLan) in partnership with Operational Solutions Ltd (OSL).

---

## Overview

Unmanned Aerial Systems (UAS) — commonly known as drones — present an increasingly complex threat to critical infrastructure, public safety, and restricted airspace. OSL develops Counter-UAS (C-UAS) solutions that detect, track, and classify drone threats using multimodal sensor arrays. This proof-of-concept (PoC) demonstrates the core machine learning pipeline required for such a system: real-time object detection and multi-object tracking applied to aerial video footage under diverse environmental conditions.

This PoC uses **YOLOv8** for deep-learning-based single-frame detection and **ByteTrack** for robust multi-frame association, running end-to-end in Python with PyTorch. It is designed to be extended toward a full C-UAS capability including thermal imaging, radar fusion, and 3D geolocation.

---

## Pipeline

```
Input Video
    │
    ▼
Frame Extraction         (OpenCV — frame-by-frame decode)
    │
    ▼
YOLOv8 Detection         (Ultralytics — per-frame bounding boxes + confidence)
    │
    ▼
ByteTrack Tracking       (Supervision — cross-frame ID assignment)
    │
    ▼
Classification           (Class label from YOLO head — extensible to fine-grained UAS type)
    │
    ▼
Annotated Output Video   (Track ID + confidence overlaid on each frame)
```

---

## Results

| Model                   | Precision                                      | Recall                                         | F1  | Avg FPS |
|-------------------------|------------------------------------------------|------------------------------------------------|-----|---------|
| YOLOv8n (pretrained)    | TBD — ground truth annotations required        | TBD — ground truth annotations required        | TBD | **35.49**   |
| YOLOv8s (fine-tuned)    | TBD — ground truth annotations required        | TBD — ground truth annotations required        | TBD | TBD     |
| YOLOv8m (fine-tuned)    | TBD — ground truth annotations required        | TBD — ground truth annotations required        | TBD | TBD     |

> FPS measured on Apple M-series (MPS) running YOLOv8n at 640×360 over 9184 frames. Precision and Recall require labelled ground-truth annotations — to be produced by `evaluate.py` once a UAS dataset is curated.

---

## Sample Outputs

### Annotated Detection and Tracking
![Sample Frame](outputs/sample_frame.png)

### Track Trajectories
![Trajectories](outputs/trajectories.png)

### Training Curves (Day 2)
![Training](outputs/training/drone_finetune/results.png)

---

## Repository Structure

```
drone-detection-tracking-poc/
├── README.md
├── requirements.txt
├── src/
│   ├── detect_track.py     # Main detection + tracking pipeline
│   ├── evaluate.py         # Precision / Recall / F1 / FPS evaluation
│   └── visualise.py        # Trajectory visualisation and annotated replay
├── notebooks/
│   └── exploration.ipynb   # Interactive dataset exploration and inference demo
└── data/
    └── .gitkeep            # Placeholder — populate with sample_clips/ before running
```

Output files (videos, plots, CSVs) are written to `outputs/` which is created automatically at runtime.

---

## Installation

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU will be selected automatically.

---

## Run Instructions

### 1. Detection and Tracking

```bash
python src/detect_track.py \
    --input  data/sample_clips/drone_flight.mp4 \
    --output outputs/tracked_output.mp4 \
    --conf   0.35 \
    --show
```

| Argument   | Description                                      | Default              |
|------------|--------------------------------------------------|----------------------|
| `--input`  | Path to input video file                         | required             |
| `--output` | Path to save annotated output video              | `outputs/output.mp4` |
| `--conf`   | Detection confidence threshold (0.0 – 1.0)       | `0.35`               |
| `--show`   | Display live preview window during processing    | off                  |

### 2. Evaluation

```bash
python src/evaluate.py \
    --gt_dir     data/ground_truth/ \
    --pred_dir   outputs/predictions/ \
    --fps        28.3
```

Results are printed to the console and saved to `outputs/evaluation_results.csv`.

### 3. Visualisation

```bash
python src/visualise.py \
    --input  outputs/tracked_output.mp4 \
    --output outputs/trajectory_plot.png
```

### 4. Notebook Exploration

```bash
jupyter notebook notebooks/exploration.ipynb
```

Populate `data/sample_clips/` with `.jpg` or `.png` frames before running.

---

## Limitations and Next Steps

### Current Limitations
- Detection operates on **visible-spectrum video only** — does not generalise to thermal (LWIR/MWIR) or depth imagery without retraining.
- Tracking is **2D** (image plane); no 3D geolocation or bearing/elevation estimation.
- Pretrained weights are trained on general object categories — UAS-specific fine-tuning on annotated drone datasets is required for operational precision.
- No **sensor fusion**: radar, RF spectrum, acoustic, and electro-optical feeds are not yet combined.
- Not yet validated for **real-time edge deployment** (e.g. Jetson Orin, Raspberry Pi 5).

### Next Steps
1. **Sensor fusion** — integrate thermal (LWIR) cameras, mmWave radar, and RF spectrum analysers to enable detection under occlusion, at night, and at long range.
2. **Custom dataset curation** — collect and annotate a labelled UAS dataset covering multiple drone classes (multirotor, fixed-wing, micro-UAS), altitudes, lighting conditions, and backgrounds.
3. **Model fine-tuning** — train YOLOv8 (and compare against RT-DETR, YOLOv9) on the curated dataset; apply data augmentation simulating fog, motion blur, and sun glare.
4. **3D geolocation** — fuse stereo camera / LiDAR / radar returns to produce WGS-84 coordinate estimates for detected UAS.
5. **Real-time edge deployment** — quantise and export models to TensorRT / ONNX for Jetson-class hardware; target >30 FPS at detection quality.
6. **Classification granularity** — extend the YOLO classification head to distinguish drone make/model, predict intent, and flag threat level.
7. **Continuous evaluation framework** — integrate automated metric tracking (mAP@0.5, MOTA, IDF1) against a held-out test set as part of a CI pipeline.

---

## Relevance to KTP Objectives

This PoC directly demonstrates competency in each technical area specified in the KTP Associate job description:

| JD Requirement                                  | Demonstrated By                                      |
|-------------------------------------------------|------------------------------------------------------|
| Object detection and tracking of UAS            | `detect_track.py` — YOLOv8 + ByteTrack              |
| Deep learning model design and training         | YOLOv8 architecture; fine-tuning workflow outlined   |
| ML across diverse conditions                    | Confidence thresholding; augmentation roadmap        |
| Dataset curation and performance evaluation     | `evaluate.py` — precision, recall, F1, FPS           |
| Python, PyTorch, real-time feasibility          | Full PyTorch backend; FPS measurement                |
| Technical documentation                         | This README, inline code comments, notebook          |

---

*University of Central Lancashire / Operational Solutions Ltd — KTP Associate Application, 2026*

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add drone footage
Place any `.mp4` file into `data/sample_clips/`

### 3. Run detection and tracking
```bash
python src/detect_track.py --input data/sample_clips/drone_test.mp4 --output outputs/annotated_drone.mp4 --conf 0.3
```

### 4. Visualise trajectories
```bash
python src/visualise.py --input outputs/annotated_drone.mp4 --output outputs/trajectories.png
```

### 5. Explore dataset
```bash
jupyter notebook notebooks/exploration.ipynb
```
