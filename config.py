import os

# ---------------------------------------------------------------------------
# Video source: 0 for webcam, or a file path like "sample.mp4"
# ---------------------------------------------------------------------------
VIDEO_SOURCE = 0

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"
DETECTION_CONFIDENCE = 0.45
COW_CLASS_NAME = "cow"

# Set True to track *all* detected objects (useful for testing without cows)
DETECT_ALL_CLASSES = False

# ---------------------------------------------------------------------------
# Centroid tracker
# ---------------------------------------------------------------------------
MAX_DISAPPEARED_FRAMES = 40   # frames with no match before dropping a track
MAX_MATCH_DISTANCE = 120      # max pixel distance for centroid matching

# ---------------------------------------------------------------------------
# Health monitoring thresholds
# ---------------------------------------------------------------------------
IDLE_SPEED_THRESHOLD = 5.0          # pixels/sec – below this is "idle"
IDLE_WARNING_SECONDS = 30           # seconds idle before Warning status
IDLE_ALERT_SECONDS = 60             # seconds idle before Alert status
ISOLATION_DISTANCE_PX = 300         # pixels from herd centroid to flag isolation
ABNORMAL_SPEED_THRESHOLD = 200.0    # pixels/sec – above this is abnormal
ALERT_COOLDOWN_SECONDS = 30         # min gap between repeated alerts per cow
POSITION_HISTORY_SECONDS = 120      # rolling window of position history

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cattle_monitoring.db")

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DB_LOG_INTERVAL_FRAMES = 30  # write status to DB every N frames
