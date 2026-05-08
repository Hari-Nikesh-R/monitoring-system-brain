# Cattle Health Monitoring System

AI-powered cattle health monitoring using computer vision. The system detects cows via YOLO, tracks them across frames with a centroid tracker, analyses movement patterns, and raises health alerts — all viewable through a live Streamlit dashboard.

## Features

- **Real-time cow detection** using YOLOv8
- **Multi-object tracking** with persistent IDs across frames
- **Movement analysis** — classifies each cow as Active or Idle
- **Health alerts** — flags prolonged inactivity, abnormal speed, and herd isolation
- **Appearance-based abnormality (optional)** — plug in a custom trained model (e.g., wounds / weak body condition) and automatically mark cows as unhealthy
- **SQLite storage** — logs every status update and alert
- **Streamlit dashboard** — live video feed, status table, statistics, alert history
- **Standalone mode** — lightweight OpenCV window without Streamlit
- **Demo mode** — detect all COCO objects for testing without live cows

## Project Structure

```
monitoring-system/
├── app.py                 # Streamlit dashboard (main entry point)
├── run_standalone.py      # OpenCV standalone viewer
├── config.py              # All configurable parameters
├── detector.py            # YOLO detection wrapper
├── tracker.py             # Centroid-based multi-object tracker
├── health_monitor.py      # Movement analysis & health logic
├── video_processor.py     # Pipeline orchestrator
├── database.py            # SQLite storage layer
├── abnormality_detector.py# Optional: abnormality model wrapper (wound/weak etc.)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- A webcam **or** a sample video file (`.mp4`, `.avi`, `.mov`)

## Installation

### 1. Clone / navigate to the project

```bash
cd monitoring-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run the YOLOv8 model (`yolov8n.pt`) will be downloaded automatically (~6 MB).

## Usage

### Option A: Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

#### Optional: enable appearance-based abnormality detection (wound/weak)

1. Place your trained model weights somewhere accessible (example: `models/abnormality_best.pt`).
2. Set the path in `config.py`:

```python
ABNORMALITY_MODEL = "models/abnormality_best.pt"
```

3. The app supports two kinds of models:
   - **Detection model (YOLO task=detect)**: classes like `wound`, `injury`, `weak`, etc. (configured via keyword lists in `config.py`)
   - **Classification model (YOLO task=classify)**: classes like `healthy` / `unhealthy` (configured via `UNHEALTHY_CLASS_*` in `config.py`)

4. Ensure your model’s class names contain keywords that map to wound/weak (configurable in `config.py`):
   - Wound keywords default to things like: `wound`, `injury`, `lesion`, `ulcer`
   - Weak keywords default to things like: `weak`, `thin`, `emaciated`, `bcs_low`

### Training a simple `healthy` vs `unhealthy` appearance classifier (starter)

This repo already contains example images:

- `trained/cow/` → treated as **healthy**
- `trained/unhealthy_cow/` → treated as **unhealthy**

#### 1) Prepare a dataset (train/val folders)

```bash
python3 scripts/prepare_health_dataset.py
```

This creates `datasets/cow_health/train/{healthy,unhealthy}` and `datasets/cow_health/val/{healthy,unhealthy}` using symlinks.

#### 2) Train with Ultralytics (classification)

```bash
yolo classify train model=yolov8n-cls.pt data=datasets/cow_health imgsz=224 epochs=50
```

After training, set `ABNORMALITY_MODEL` to the resulting `best.pt` (usually under `runs/classify/.../weights/best.pt`).

This opens a browser tab at `http://localhost:8501` with:

1. **Sidebar** — choose Webcam or upload a video file, adjust detection confidence, toggle demo mode.
2. **Live Feed** — annotated video with bounding boxes, cow IDs, and health labels.
3. **Cow Status Table** — real-time per-cow status.
4. **Statistics** — total cows, active count, idle count, alert count.
5. **Alert Log** — timestamped alerts from the database.

**Steps:**

1. Select your video source in the sidebar (Webcam or upload a file).
2. Optionally enable **Detect all objects (demo mode)** if you don't have a cow video.
3. Check **▶ Start Monitoring** to begin processing.
4. Uncheck it to stop.

### Option B: Standalone OpenCV Window

```bash
# Use webcam
python run_standalone.py

# Use a video file
python run_standalone.py --source path/to/video.mp4

# Demo mode (detect all objects)
python run_standalone.py --detect-all

# Custom confidence
python run_standalone.py --confidence 0.6
```

Press **q** or **ESC** to quit.

> **Note:** The standalone mode requires `opencv-python` (with GUI support). If you installed `opencv-python-headless`, replace it:
> ```bash
> pip uninstall opencv-python-headless
> pip install opencv-python
> ```

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `0` | Webcam index or file path |
| `YOLO_MODEL` | `"yolov8n.pt"` | YOLO model name/path |
| `DETECTION_CONFIDENCE` | `0.45` | Minimum detection score |
| `DETECT_ALL_CLASSES` | `False` | Track all objects, not just cows |
| `ABNORMALITY_MODEL` | `None` | Optional custom model path to detect abnormalities (wounds/weak) |
| `ABNORMALITY_CONFIDENCE` | `0.35` | Confidence threshold for abnormality detections |
| `ABNORMALITY_MIN_HITS` | `1` | Min number of abnormal hits before marking unhealthy |
| `MAX_DISAPPEARED_FRAMES` | `40` | Frames before dropping a lost track |
| `MAX_MATCH_DISTANCE` | `120` | Max pixels for centroid matching |
| `IDLE_SPEED_THRESHOLD` | `5.0` | px/s — below this is idle |
| `IDLE_WARNING_SECONDS` | `30` | Seconds idle → Warning |
| `IDLE_ALERT_SECONDS` | `60` | Seconds idle → Alert |
| `ISOLATION_DISTANCE_PX` | `300` | Pixels from herd centroid to flag isolation |
| `ABNORMAL_SPEED_THRESHOLD` | `200.0` | px/s — above this is abnormal |
| `ALERT_COOLDOWN_SECONDS` | `30` | Min gap between repeated alerts |

## How It Works

```
Camera / Video File
       ↓
  Video Capture (OpenCV)
       ↓
  Cow Detection (YOLOv8)
       ↓
  Centroid Tracking (persistent IDs)
       ↓
  Movement Analysis (speed, idle duration)
       ↓
  Health Monitoring (rules engine)
       ↓
  Appearance Abnormality (optional)
  (custom YOLO model on cow crops)
       ↓
  SQLite Storage (statuses + alerts)
       ↓
  Dashboard / Alerts (Streamlit)
```

1. **Detection** — YOLOv8n scans each frame for cows (COCO class 19).
2. **Tracking** — A centroid tracker matches detections across frames using nearest-neighbour distance, assigning stable IDs.
3. **Health Analysis** — For each tracked cow, the system computes speed over a rolling window. Prolonged low speed triggers Idle/Warning/Alert. Abnormally high speed or isolation from the herd also raises alerts.
4. **Appearance Abnormality (optional)** — If `ABNORMALITY_MODEL` is set, each tracked cow’s bounding box is cropped and passed to the abnormality model. If a wound/weak label is detected, the cow is marked `Alert` (unhealthy) and an alert is logged.
5. **Storage** — Status snapshots and alerts are written to a local SQLite database (`cattle_monitoring.db`).
6. **Dashboard** — Streamlit renders the annotated video, a live status table (including an **Appearance** column), aggregate metrics, and an alert log.

## Testing Without Cows

Enable **demo mode** to detect and track all 80 COCO classes (people, cars, dogs, etc.):

- **Dashboard:** Check *Detect all objects (demo mode)* in the sidebar.
- **Standalone:** Pass `--detect-all` flag.

You can point your webcam at any scene or use any video file to verify the pipeline works end-to-end.

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Webcam not opening | Check camera index in `config.py` or try `--source 1` |
| No cows detected | Enable demo mode, or lower the confidence threshold |
| `cv2.imshow` crash | Use the Streamlit dashboard (`streamlit run app.py`) instead |
| Slow FPS | Use a smaller YOLO model or reduce input resolution |
| Abnormality model not triggering | Confirm `ABNORMALITY_MODEL` points to a valid `.pt` file and reduce `ABNORMALITY_CONFIDENCE` |
| Wound/Weak never shows up | Update `WOUND_CLASS_KEYWORDS` / `WEAK_CLASS_KEYWORDS` in `config.py` to match your model’s class names |
