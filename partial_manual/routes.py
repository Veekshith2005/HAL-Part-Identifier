from flask import Blueprint, render_template, jsonify, Response, request
import cv2
import core
from collections import defaultdict

partial_manual_bp = Blueprint(
    "partial_manual",
    __name__,
    template_folder="templates"
)

CONFIDENCE_THRESHOLD = 0.75   # strict threshold for partial manual


@partial_manual_bp.route("/partial_manual/")
def partial_manual_page():
    return render_template("partial_manual.html")


@partial_manual_bp.route("/partial_manual/parts_list")
def parts_list():
    records = core.metadata[[
        "part_name", "part_number", "issue", "yolo_class"
    ]].drop_duplicates().to_dict("records")
    return jsonify(records)


@partial_manual_bp.route("/partial_manual/video_feed")
def pm_video_feed():
    core.start_camera()

    def generate():
        while core.camera_active:
            frame = core.get_raw_frame()
            if frame is None:
                continue
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                   + buffer.tobytes() + b"\r\n")

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@partial_manual_bp.route("/partial_manual/capture_count", methods=["POST"])
def capture_count():
    frame = core.get_raw_frame()
    if frame is None:
        return jsonify({"error": "No camera frame available"}), 400

    h, w = frame.shape[:2]
    frame_area = w * h
    MIN_BOX_AREA = frame_area * 0.03   # box must cover at least 3% of frame
    MAX_BOX_AREA = frame_area * 0.92   # reject full-frame detections (papers)

    with core.model_lock:
        results = core.part_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    boxes = results[0].boxes
    if len(boxes) == 0:
        return jsonify({"detections": [], "message": "No parts detected"})

    class_confs = defaultdict(list)
    for box in boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = results[0].names[cls].strip()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)

        # reject boxes that are too small or fill the whole frame
        if box_area < MIN_BOX_AREA or box_area > MAX_BOX_AREA:
            continue

        class_confs[label].append(conf)

    if not class_confs:
        return jsonify({"detections": [], "message": "No valid parts detected"})

    results_out = []
    for label, confs in class_confs.items():
        # require at least 2 detections of same class OR very high confidence
        if len(confs) < 1:
            continue
        count     = len(confs)
        best_conf = max(confs)

        row = core.metadata[core.metadata["yolo_class"] == label]
        if not row.empty:
            results_out.append({
                "yolo_class":  label,
                "part_name":   row.iloc[0]["part_name"],
                "part_number": row.iloc[0]["part_number"],
                "issue":       row.iloc[0]["issue"],
                "confidence":  round(best_conf * 100, 2),
                "count":       count
            })
        else:
            results_out.append({
                "yolo_class":  label,
                "part_name":   label,
                "part_number": "Unknown",
                "issue":       "—",
                "confidence":  round(best_conf * 100, 2),
                "count":       count
            })

    if not results_out:
        return jsonify({"detections": [], "message": "No valid parts detected"})

    return jsonify({"detections": results_out})


@partial_manual_bp.route("/partial_manual/confirm", methods=["POST"])
def confirm():
    import pandas as pd

    data  = request.get_json()
    items = data.get("items", [])

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
        yolo_class  = item.get("yolo_class", part_name).strip()
        count       = int(item.get("count", 1))

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
            "PARTIAL-MANUAL",
            part_number, part_name, issue,
            confidence, count, ref_path,
            original_details, user_edited_details
        )
        logged += 1

    return jsonify({"ok": True, "logged": logged})


@partial_manual_bp.route("/partial_manual/add_part", methods=["POST"])
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


@partial_manual_bp.route("/partial_manual/stop")
def pm_stop():
    core.stop_camera()
    return jsonify({"ok": True})