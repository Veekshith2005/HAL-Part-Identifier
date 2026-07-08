# core.py
# Shared models, camera, metadata, and logging.
# Imported by app1.py AND both blueprints — never imports app1 itself.

from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import threading
import os
from datetime import datetime

PART_MODEL_PATH  = "trained_models/retrained.pt"
THUMB_MODEL_PATH = "trained_models/thumb_detector.pt"
CSV_PATH         = "metadata/parts_master.csv"
LOG_FILE         = "logs/detection_log.xlsx"
RAW_IMAGES_DIR   = "raw_images1"
DEBUG_MODE       = False

def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

os.makedirs("logs", exist_ok=True)


# =====================
# STRICT DETECTION CONSTANTS
# Used by all three detection paths (upload, partial manual, live)
# =====================

CONF_THRESHOLD   = 0.82   # minimum confidence to accept any detection
MIN_BOX_RATIO    = 0.03   # box must cover at least 3% of frame area
MAX_BOX_RATIO    = 0.90   # box must NOT cover 90%+ of frame (papers, random)


def is_valid_box(box_xyxy, frame_w, frame_h, conf):
    """
    Returns True only if:
    - confidence meets the threshold
    - bounding box is not too small (noise) or too large (paper/background)
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    box_area   = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    ratio      = box_area / frame_area

    if conf < CONF_THRESHOLD:
        debug_print(f"  Rejected: conf {conf:.3f} < {CONF_THRESHOLD}")
        return False
    if ratio < MIN_BOX_RATIO:
        debug_print(f"  Rejected: box too small ({ratio:.1%})")
        return False
    if ratio > MAX_BOX_RATIO:
        debug_print(f"  Rejected: box too large ({ratio:.1%}) — likely paper/background")
        return False
    return True


# =====================
# MODELS & METADATA
# =====================

part_model  = YOLO(PART_MODEL_PATH)
thumb_model = YOLO(THUMB_MODEL_PATH)

metadata = pd.read_csv(CSV_PATH)
metadata["yolo_class"] = metadata["yolo_class"].astype(str).str.strip()

print("MODEL CLASSES")
print(part_model.names)


def get_reference_image(label):
    """
    Returns the path to a reference image for the given detected label,
    picked from raw_images1/<label>/ folder.
    """
    folder = os.path.join(RAW_IMAGES_DIR, label)
    if not os.path.isdir(folder):
        return "N/A"

    images = [f for f in os.listdir(folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        return "N/A"

    return os.path.join(folder, sorted(images)[0]).replace("\\", "/")


# =====================
# WARM-UP
# Forces one-time layer fusion on the MAIN thread, before any
# camera/request threads exist.
# =====================

_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
part_model(_dummy, verbose=False)
thumb_model(_dummy, verbose=False)
print(">>> Models warmed up and fused successfully")


# =====================
# MODEL LOCK
# =====================

model_lock = threading.Lock()
log_lock   = threading.Lock()


# =====================
# LOG
# =====================

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "Timestamp", "Source", "Part Number", "Part Name",
        "Issue", "Confidence", "Count", "Reference Path",
        "Original Details", "User Edited Details"
    ]).to_excel(LOG_FILE, index=False)


def save_log(source, pn, name, issue, conf, count=1,
             reference_path="N/A",
             original_details=None, user_edited_details=None):
    with log_lock:
        df   = pd.read_excel(LOG_FILE)
        orig = original_details or f"{name} | {pn} | {issue}"
        edited = user_edited_details or orig

        row = pd.DataFrame([{
            "Timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Source":              source,
            "Part Number":         pn,
            "Part Name":           name,
            "Issue":               issue,
            "Confidence":          round(conf, 2),
            "Count":               count,
            "Reference Path":      reference_path,
            "Original Details":    orig,
            "User Edited Details": edited
        }])
        df = pd.concat([df, row], ignore_index=True)
        df.to_excel(LOG_FILE, index=False)


# =====================
# SHARED CAMERA
# =====================

camera           = None
camera_active    = False
latest_raw_frame = None
latest_frame     = None
camera_lock      = threading.Lock()


def start_camera():
    global camera, camera_active
    with camera_lock:
        if camera_active:
            return
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera_active = True
        threading.Thread(target=_camera_loop, daemon=True).start()


def stop_camera():
    global camera_active, camera
    with camera_lock:
        camera_active = False
        if camera:
            camera.release()
            camera = None


def _camera_loop():
    global latest_frame, latest_raw_frame
    from app1 import detect
    while camera_active:
        ret, frame = camera.read()
        if ret:
            latest_raw_frame = frame.copy()
            latest_frame     = detect(frame, "WEBCAM")


def get_raw_frame():
    """Blueprints call this to grab the latest unprocessed camera frame."""
    return latest_raw_frame