from flask import Blueprint, render_template, jsonify, Response, request
import cv2
import time
import core
from collections import defaultdict

live_identification_bp = Blueprint(
    "live_identification",
    __name__,
    template_folder="templates"
)

live_detections     = {}
_frame_counts       = defaultdict(list)
_buffer_start       = None
BUFFER_SECONDS      = 10
results_ready       = False
recognition_started = False

parts_active  = True
no_part_start = None
PART_TIMEOUT  = 5
status_text   = "DETECTION ACTIVE"

CONFIDENCE_THRESHOLD = 0.75   # strict threshold
MIN_BOX_AREA_RATIO   = 0.03   # box must be at least 3% of frame
MAX_BOX_AREA_RATIO   = 0.92   # reject full-frame detections


@live_identification_bp.route("/live_identification/")
def live_id_page():
    global parts_active, no_part_start, status_text
    global _buffer_start, results_ready, live_detections
    global _frame_counts, recognition_started
    parts_active        = True
    no_part_start       = None
    status_text         = "DETECTION ACTIVE"
    _buffer_start       = None
    results_ready       = False
    live_detections     = {}
    recognition_started = False
    _frame_counts.clear()
    return render_template("live_identification.html")


@live_identification_bp.route("/live_identification/start", methods=["POST"])
def start_recognition():
    global recognition_started, _buffer_start, results_ready
    global live_detections, _frame_counts
    recognition_started = True
    _buffer_start       = None
    results_ready       = False
    live_detections     = {}
    _frame_counts.clear()
    core.start_camera()
    return jsonify({"ok": True})


@live_identification_bp.route("/live_identification/rerecognize", methods=["POST"])
def rerecognize():
    global _buffer_start, results_ready, live_detections, _frame_counts
    global parts_active, no_part_start, status_text
    _buffer_start   = None
    results_ready   = False
    live_detections = {}
    _frame_counts.clear()
    parts_active  = True
    no_part_start = None
    status_text   = "DETECTION ACTIVE"
    return jsonify({"ok": True})


@live_identification_bp.route("/live_identification/video_feed")
def li_video_feed():

    def generate():
        global _buffer_start, live_detections, results_ready
        global parts_active, no_part_start, status_text, _frame_counts

        while core.camera_active:
            frame = core.get_raw_frame()
            if frame is None:
                continue

            now  = time.time()
            h, w = frame.shape[:2]
            frame_area = w * h

            if not recognition_started:
                _, buf = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
                continue

            if not parts_active:
                with core.model_lock:
                    results = core.thumb_model(frame, conf=0.5, verbose=False)
                annotated = results[0].plot()
                if len(results[0].boxes) > 0:
                    parts_active  = True
                    no_part_start = None
                    status_text   = "DETECTION RESUMED"
                else:
                    status_text = "SHOW THUMB TO RESUME"
                cv2.putText(annotated, status_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                _, buf = cv2.imencode(".jpg", annotated)
                yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
                continue

            if results_ready:
                annotated = frame.copy()
                cv2.putText(annotated, "RESULTS READY",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 200, 0), 2)
                _, buf = cv2.imencode(".jpg", annotated)
                yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
                continue

            with core.model_lock:
                results = core.part_model(
                    frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            boxes     = results[0].boxes
            annotated = results[0].plot()

            if len(boxes) == 0:
                if no_part_start is None:
                    no_part_start = now
                elif now - no_part_start >= PART_TIMEOUT:
                    parts_active  = False
                    no_part_start = None
                    status_text   = "SHOW THUMB TO RESUME"
            else:
                no_part_start = None
                status_text   = "DETECTING..."

            if _buffer_start is None:
                _buffer_start = now

            frame_class_counts = defaultdict(int)

            for box in boxes:
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = results[0].names[cls].strip()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)

                if box_area < frame_area * MIN_BOX_AREA_RATIO:
                    continue
                if box_area > frame_area * MAX_BOX_AREA_RATIO:
                    continue

                frame_class_counts[label] += 1
                _frame_counts[f"{label}_conf"].append(conf)

            for label, cnt in frame_class_counts.items():
                _frame_counts[label].append(cnt)

            elapsed   = now - _buffer_start
            remaining = max(0, BUFFER_SECONDS - elapsed)
            cv2.putText(annotated,
                        f"Scanning: {remaining:.1f}s",
                        (30, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            if elapsed >= BUFFER_SECONDS:
                new_detections = {}
                for label, frame_cnt_list in list(_frame_counts.items()):
                    if label.endswith("_conf"):
                        continue
                    # require part appeared in at least 30% of frames
                    total_frames = max(1, elapsed * 10)
                    if len(frame_cnt_list) < total_frames * 0.20:
                        continue

                    true_count = max(frame_cnt_list) if frame_cnt_list else 1
                    conf_list  = _frame_counts.get(f"{label}_conf", [CONFIDENCE_THRESHOLD])
                    best_conf  = max(conf_list)

                    row = core.metadata[core.metadata["yolo_class"] == label]
                    if not row.empty:
                        new_detections[label] = {
                            "yolo_class":  label,
                            "part_number": row.iloc[0]["part_number"],
                            "part_name":   row.iloc[0]["part_name"],
                            "issue":       row.iloc[0]["issue"],
                            "confidence":  round(best_conf * 100, 2),
                            "count":       true_count
                        }
                    else:
                        new_detections[label] = {
                            "yolo_class":  label,
                            "part_number": "Unknown",
                            "part_name":   label,
                            "issue":       "—",
                            "confidence":  round(best_conf * 100, 2),
                            "count":       true_count
                        }

                live_detections = new_detections
                _frame_counts.clear()
                _buffer_start   = None
                results_ready   = True

            _, buf = cv2.imencode(".jpg", annotated)
            yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@live_identification_bp.route("/live_identification/detections")
def get_detections():
    return jsonify({
        "detections":    list(live_detections.values()),
        "results_ready": results_ready,
        "time_left":     max(0, BUFFER_SECONDS - (
            time.time() - _buffer_start if _buffer_start else BUFFER_SECONDS
        ))
    })


@live_identification_bp.route("/live_identification/status")
def get_status():
    return jsonify({
        "status":        status_text,
        "parts_active":  parts_active,
        "results_ready": results_ready
    })


@live_identification_bp.route("/live_identification/update_count", methods=["POST"])
def update_count():
    body       = request.get_json()
    yolo_class = body.get("yolo_class")
    count      = int(body.get("count", 0))
    if yolo_class and yolo_class in live_detections:
        live_detections[yolo_class]["count"] = count
    return jsonify({"ok": True})


@live_identification_bp.route("/live_identification/confirm", methods=["POST"])
def confirm_detections():
    import pandas as pd
    data   = request.get_json()
    items  = data.get("items", [])
    logged = 0

    for item in items:
        # camera-detected (original)
        orig_name   = item.get("orig_name",   "").strip()
        orig_number = item.get("orig_number", "").strip()
        orig_issue  = item.get("orig_issue",  "").strip()

        # user-edited (final)
        part_name   = item.get("part_name",   "").strip()
        part_number = item.get("part_number", "").strip()
        issue       = item.get("issue",       "").strip()
        confidence  = float(item.get("confidence", 0))
        count       = int(item.get("count", 1))
        yolo_class  = item.get("yolo_class",  part_name).strip()

        if not part_name:
            continue

        existing = core.metadata[
            (core.metadata["part_number"] == part_number) &
            (core.metadata["issue"]       == issue)
        ]
        if existing.empty and part_number not in ("Unknown", "", "—"):
            new_row = {
                "class_id":    len(core.metadata),
                "part_number": part_number,
                "issue":       issue,
                "part_name":   part_name,
                "yolo_class":  yolo_class
            }
            core.metadata = pd.concat(
                [core.metadata, pd.DataFrame([new_row])], ignore_index=True
            )
            core.metadata.to_csv(core.CSV_PATH, index=False)

        ref_path = core.get_reference_image(yolo_class).replace("\\", "/")

        original_details    = f"{orig_name} | {orig_number} | {orig_issue}"
        user_edited_details = f"{part_name} | {part_number} | {issue}"

        core.save_log(
            "LIVE-IDENTIFICATION",
            part_number, part_name, issue,
            confidence, count, ref_path,
            original_details, user_edited_details
        )
        logged += 1

    return jsonify({"ok": True, "logged": logged})


@live_identification_bp.route("/live_identification/add_part", methods=["POST"])
def add_part():
    import pandas as pd
    data        = request.get_json()
    part_name   = data.get("part_name",   "").strip()
    part_number = data.get("part_number", "").strip()
    issue       = data.get("issue",       "").strip()
    yolo_class  = data.get("yolo_class",  part_name).strip()

    if not part_name or not part_number:
        return jsonify({"error": "part_name and part_number required"}), 400

    existing = core.metadata[
        (core.metadata["part_number"] == part_number) &
        (core.metadata["issue"]       == issue)
    ]
    if existing.empty:
        new_row = {
            "class_id":    len(core.metadata),
            "part_number": part_number,
            "issue":       issue,
            "part_name":   part_name,
            "yolo_class":  yolo_class
        }
        core.metadata = pd.concat(
            [core.metadata, pd.DataFrame([new_row])], ignore_index=True
        )
        core.metadata.to_csv(core.CSV_PATH, index=False)

    records = core.metadata[[
        "part_name", "part_number", "issue", "yolo_class"
    ]].drop_duplicates().to_dict("records")
    return jsonify({"ok": True, "parts": records})


@live_identification_bp.route("/live_identification/stop")
def li_stop():
    core.stop_camera()
    return jsonify({"ok": True})