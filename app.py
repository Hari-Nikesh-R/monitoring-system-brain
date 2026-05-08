"""Streamlit dashboard for the Cattle Health Monitoring System."""

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from database import Database
from video_processor import VideoProcessor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cattle Health Monitor",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .alert-healthy  { color: #28a745; font-weight: 600; }
    .alert-warning  { color: #ffc107; font-weight: 600; }
    .alert-danger   { color: #dc3545; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🐄 Cattle Health Monitor")
    st.markdown("---")

    source_type = st.radio("Video Source", ["Webcam", "Video File"], horizontal=True)

    uploaded_file = None
    if source_type == "Video File":
        uploaded_file = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv"]
        )

    st.markdown("---")
    st.subheader("Detection Settings")

    detect_all = st.checkbox(
        "Detect all objects (demo mode)",
        value=False,
        help="Track every detected object, not only cows. Useful for testing.",
    )
    confidence = st.slider("Confidence threshold", 0.10, 1.00, 0.45, 0.05)
    loop_video = st.checkbox("Loop video file", value=True)

    st.markdown("---")
    run = st.checkbox("▶ Start Monitoring", value=False)

    st.markdown("---")
    if st.button("🗑 Clear Database"):
        Database().clear_all()
        st.success("Database cleared.")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.header("Live Monitoring Dashboard")

col_video, col_stats = st.columns([3, 2], gap="large")

with col_video:
    st.subheader("Camera Feed")
    frame_slot = st.empty()

with col_stats:
    st.subheader("Cow Status")
    status_slot = st.empty()
    st.subheader("Statistics")
    metric_cols = st.columns(4)
    metric_slots = [c.empty() for c in metric_cols]

st.markdown("---")
st.subheader("Recent Alerts")
alert_slot = st.empty()

# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------
if run:
    # Resolve video source
    cap = None
    tmp_path = None

    if source_type == "Webcam":
        cap = cv2.VideoCapture(0)
    elif uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
    else:
        st.warning("Please upload a video file or select Webcam.")
        st.stop()

    if not cap.isOpened():
        st.error("Could not open video source.")
        st.stop()

    processor = VideoProcessor(detect_all=detect_all, confidence=confidence)
    db = Database()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            if loop_video and source_type == "Video File" and tmp_path:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        annotated, results = processor.process_frame(frame)

        # -- video feed -------------------------------------------------------
        display = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_slot.image(display, channels="RGB", use_container_width=True)

        # -- cow status table -------------------------------------------------
        if results:
            rows = []
            for cid, info in results.items():
                appearance = info.get("appearance_flags") or []
                rows.append(
                    {
                        "Cow ID": cid,
                        "Movement": info["movement_status"],
                        "Health": info["health_status"],
                        "Appearance": ", ".join(appearance) if appearance else "-",
                        "X": int(info["centroid"][0]),
                        "Y": int(info["centroid"][1]),
                    }
                )
            status_slot.dataframe(
                pd.DataFrame(rows).set_index("Cow ID"),
                use_container_width=True,
            )
        else:
            status_slot.info("No cows detected yet.")

        # -- statistics -------------------------------------------------------
        total = len(results)
        active = sum(1 for r in results.values() if r["movement_status"] == "Active")
        idle = sum(1 for r in results.values() if r["movement_status"] == "Idle")
        alert_cnt = sum(1 for r in results.values() if r["health_status"] == "Alert")

        metric_slots[0].metric("Total", total)
        metric_slots[1].metric("Active", active)
        metric_slots[2].metric("Idle", idle)
        metric_slots[3].metric("Alerts", alert_cnt)

        # -- alerts -----------------------------------------------------------
        recent_alerts = db.get_recent_alerts(limit=10)
        if recent_alerts:
            alert_slot.dataframe(
                pd.DataFrame(recent_alerts)[
                    ["timestamp", "cow_id", "alert_type", "message"]
                ],
                use_container_width=True,
            )
        else:
            alert_slot.info("No alerts yet.")

        time.sleep(0.01)

    cap.release()
else:
    st.info(
        "Toggle **▶ Start Monitoring** in the sidebar to begin. "
        "Select a video source first."
    )

    # Show historical data when not running
    db = Database()
    statuses = db.get_latest_cow_statuses()
    if statuses:
        st.subheader("Last Known Statuses")
        st.dataframe(pd.DataFrame(statuses), use_container_width=True)

    alerts = db.get_recent_alerts(limit=20)
    if alerts:
        st.subheader("Alert History")
        st.dataframe(
            pd.DataFrame(alerts)[["timestamp", "cow_id", "alert_type", "message"]],
            use_container_width=True,
        )
