"""Standalone OpenCV viewer — runs without Streamlit.

Usage:
    python run_standalone.py                    # webcam
    python run_standalone.py --source video.mp4 # video file
    python run_standalone.py --detect-all       # demo: track all objects
"""

import argparse
import sys

import cv2

from config import VIDEO_SOURCE, DETECTION_CONFIDENCE
from video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Cattle Health Monitor – standalone")
    parser.add_argument(
        "--source",
        default=str(VIDEO_SOURCE),
        help="Video source: 0 for webcam, or path to a video file (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DETECTION_CONFIDENCE,
        help="YOLO confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--detect-all",
        action="store_true",
        help="Detect all COCO classes instead of only cows",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: cannot open video source '{source}'")
        sys.exit(1)

    processor = VideoProcessor(
        detect_all=args.detect_all, confidence=args.confidence
    )

    print("Press 'q' or ESC to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, results = processor.process_frame(frame)

        cv2.imshow("Cattle Health Monitor", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
