"""
Runs the trained fire-detection model live on a video source: a local
webcam, or a network camera stream (RTSP/HTTP, e.g. a CCTV feed).

Usage:
    python detect_webcam.py
    python detect_webcam.py --source 1
    python detect_webcam.py --source "rtsp://admin:password@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0"
Press 'q' to quit.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5
RECONNECT_DELAY_SECONDS = 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0, 1, ...) or a stream URL (rtsp://, http://). "
             "Default is 0, the first local webcam.",
    )
    return parser.parse_args()

def open_capture(source):
    return cv2.VideoCapture(int(source) if source.isdigit() else source)

def main():
    args = parse_args()
    model = YOLO(MODEL_PATH)
    cap = open_capture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the video stream, reconnecting...")
            cap.release()
            time.sleep(RECONNECT_DELAY_SECONDS)
            cap = open_capture(args.source)
            continue

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        fire_detected = len(results[0].boxes) > 0
        if fire_detected:
            cv2.putText(
                annotated, "FIRE DETECTED - HAZARD", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
            )

        cv2.imshow("CANARY", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
