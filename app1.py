from flask import Flask, render_template, request, Response, jsonify, url_for
import core
import cv2
import numpy as np
import os
import logging

app = Flask(__name__, template_folder="templates")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================
# STRICT THRESHOLDS
# =====================
UPLOAD_CONF_THRESHOLD  = 0.82   # single-pass minimum for upload
UPLOAD_VOTE_THRESHOLD  = 0.75   # per-pass minimum during voting
UPLOAD_VOTE_PASSES     = 5      # number of inference passes
UPLOAD_VOTE_MIN_WINS   = 4      # same class must win this many passes
UPLOAD_MIN_BOX_RATIO   = 0.03   # box must cover ≥3% of image area
UPLOAD_MAX_BOX_RATIO   = 0.90   # box must NOT cover ≥90% (full-frame = paper)

# =====================
# DETECTION GLOBALS
# =====================

latest_detection = {
    "part_number": "",
    "part_name":   "",
    "issue":       "",
    "confidence":  ""
}

detection_source = "NONE"
last_part        = None


# =====================
# RESET
# =====================

def reset_detection():
    global latest_detection, last_part
    latest_detection = {
        "part_number": "",
        "part_name":   "",
        "issue":       "",
        "confidence":  ""
    }
    last_part = None


# =====================
# STRICT UPLOAD DETECT
# Runs multiple inference passes and only accepts a detection if:
#   1. Confidence is above UPLOAD_CONF_THRESHOLD in at least one pass
#   2. The same class wins in UPLOAD_VOTE_MIN_WINS out of UPLOAD_VOTE_PASSES
#   3. The bounding box is not too small or too large (rejects papers/random)
# =====================

def detect_upload(img):
    """Multi-pass voted detection for uploaded images. Returns (label, conf) or (None, 0)."""
    h, w   = img.shape[:2]
    area   = w * h
    votes  = {}   # label -> list of confidences

    for i in range(UPLOAD_VOTE_PASSES):
        # slight augmentation on passes 1-4 to simulate viewpoint variance
        if i == 0:
            frame = img.copy()
        else:
            # random tiny brightness/contrast jitter — keeps real parts stable,
            # destabilises random textures
            alpha = np.random.uniform(0.92, 1.08)
            beta  = np.random.randint(-8, 8)
            frame = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        with core.model_lock:
            results = core.part_model(frame, conf=UPLOAD_VOTE_THRESHOLD, verbose=False)

        boxes = results[0].boxes
        if len(boxes) == 0:
            continue

        best = boxes.conf.argmax()
        conf = float(boxes.conf[best])
        cls  = int(boxes.cls[best])
        x1, y1, x2, y2 = map(int, boxes.xyxy[best])
        box_area = (x2 - x1) * (y2 - y1)

        # reject full-frame boxes (papers, random backgrounds)
        if box_area > area * UPLOAD_MAX_BOX_RATIO:
            core.debug_print(f"Pass {i}: rejected — box too large ({box_area/area:.0%})")
            continue

        # reject tiny boxes
        if box_area < area * UPLOAD_MIN_BOX_RATIO:
            core.debug_print(f"Pass {i}: rejected — box too small ({box_area/area:.0%})")
            continue

        label = results[0].names[cls].strip()
        votes.setdefault(label, []).append(conf)
        core.debug_print(f"Pass {i}: {label} @ {conf:.3f}")

    if not votes:
        return None, 0

    # find which label won the most passes
    best_label = max(votes, key=lambda l: len(votes[l]))
    win_count  = len(votes[best_label])
    best_conf  = max(votes[best_label])

    core.debug_print(f"Voting result: {best_label} won {win_count}/{UPLOAD_VOTE_PASSES} passes, best conf={best_conf:.3f}")

    # require minimum wins AND minimum confidence
    if win_count < UPLOAD_VOTE_MIN_WINS:
        core.debug_print(f"Rejected — only {win_count} wins, need {UPLOAD_VOTE_MIN_WINS}")
        return None, 0

    if best_conf < UPLOAD_CONF_THRESHOLD:
        core.debug_print(f"Rejected — best conf {best_conf:.3f} < threshold {UPLOAD_CONF_THRESHOLD}")
        return None, 0

    return best_label, best_conf


# =====================
# DETECT (webcam / live feed annotator — not used for upload anymore)
# =====================

def detect(frame, source):
    global latest_detection, last_part

    with core.model_lock:
        results = core.part_model(frame, conf=0.65, verbose=False)

    boxes     = results[0].boxes
    annotated = results[0].plot()
    core.debug_print("NUMBER OF BOXES:", len(boxes))

    if len(boxes) > 0:
        h, w     = frame.shape[:2]
        area     = w * h

        best     = boxes.conf.argmax()
        conf     = float(boxes.conf[best])
        cls      = int(boxes.cls[best])
        x1, y1, x2, y2 = map(int, boxes.xyxy[best])
        box_area = (x2 - x1) * (y2 - y1)
        label    = results[0].names[cls].strip()

        # reject full-frame and tiny boxes for live feed too
        if box_area > area * UPLOAD_MAX_BOX_RATIO or box_area < area * UPLOAD_MIN_BOX_RATIO:
            core.debug_print(f"Live: rejected box area {box_area/area:.0%}")
        elif conf >= 0.75:
            row = core.metadata[core.metadata["yolo_class"] == label]
            if not row.empty:
                pn    = row.iloc[0]["part_number"]
                name  = row.iloc[0]["part_name"]
                issue = row.iloc[0]["issue"]
                latest_detection = {
                    "part_number": pn,
                    "part_name":   name,
                    "issue":       issue,
                    "confidence":  round(conf * 100, 2)
                }
                if last_part != pn and source in ("UPLOAD",):
                    ref_path = core.get_reference_image(label).replace("\\", "/")
                    core.save_log(source, pn, name, issue, conf * 100, 1, ref_path)
                    last_part = pn
            else:
                core.debug_print(f"WARNING: label '{label}' not in CSV")

    cv2.putText(annotated, "DETECTION ACTIVE", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    return annotated


# =====================
# REGISTER BLUEPRINTS
# =====================

from partial_manual.routes      import partial_manual_bp
from live_identification.routes import live_identification_bp

app.register_blueprint(partial_manual_bp)
app.register_blueprint(live_identification_bp)


# =====================
# DASHBOARD
# =====================

@app.route("/", methods=["GET", "POST"])
def dashboard():
    global detection_source, latest_detection

    uploaded_image = None
    result         = None

    if request.method == "POST":
        reset_detection()
        file = request.files["image"]

        if file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            uploaded_image   = url_for("static", filename="uploads/" + file.filename)
            img              = cv2.imread(path)
            detection_source = "UPLOAD"

            label, conf = detect_upload(img)

            if label is not None:
                row = core.metadata[core.metadata["yolo_class"] == label]
                if not row.empty:
                    pn    = row.iloc[0]["part_number"]
                    name  = row.iloc[0]["part_name"]
                    issue = row.iloc[0]["issue"]
                    latest_detection = {
                        "part_number": pn,
                        "part_name":   name,
                        "issue":       issue,
                        "confidence":  round(conf * 100, 2)
                    }
                    ref_path = core.get_reference_image(label).replace("\\", "/")
                    core.save_log("UPLOAD", pn, name, issue, conf * 100, 1, ref_path)
                else:
                    latest_detection = {
                        "part_number": "Unknown",
                        "part_name":   label,
                        "issue":       "—",
                        "confidence":  round(conf * 100, 2)
                    }
            else:
                # nothing passed the strict filter
                latest_detection = {
                    "part_number": "No confident detection",
                    "part_name":   "—",
                    "issue":       "—",
                    "confidence":  ""
                }

            result = latest_detection.copy()

    return render_template("dashboard.html",
                           uploaded_image=uploaded_image,
                           result=result)


# =====================
# VIDEO FEED (main)
# =====================

@app.route("/video_feed")
def video_feed():
    global detection_source
    detection_source = "LIVE"
    core.start_camera()

    def generate():
        while core.camera_active:
            if core.latest_frame is None:
                continue
            _, buf = cv2.imencode(".jpg", core.latest_frame)
            yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# =====================
# API ROUTES
# =====================

@app.route("/stop_camera")
def stop():
    global detection_source
    detection_source = "NONE"
    core.stop_camera()
    return jsonify({"ok": True})


@app.route("/reset_detection")
def reset():
    reset_detection()
    return jsonify({"ok": True})


@app.route("/detection_data")
def data():
    return jsonify(latest_detection)


@app.route("/detection_source")
def get_detection_source():
    return jsonify({"source": detection_source})


@app.route("/logs")
def logs():
    import pandas as pd
    df = pd.read_excel(core.LOG_FILE)
    records = df.iloc[::-1].to_dict("records")
    return render_template("logs.html", records=records)


@app.route("/serve_ref_image")
def serve_ref_image():
    from flask import send_file, abort

    rel_path = request.args.get("path", "").strip()
    if not rel_path:
        abort(400)

    rel_path     = rel_path.replace("\\", "/")
    project_root = os.path.dirname(os.path.abspath(__file__))
    abs_path     = os.path.normpath(os.path.join(project_root, rel_path))
    allowed      = os.path.normpath(os.path.join(project_root, core.RAW_IMAGES_DIR))

    if not abs_path.startswith(allowed):
        abort(403)
    if not os.path.isfile(abs_path):
        abort(404)

    return send_file(abs_path)


@app.route("/api/stats")
def api_stats():
    import pandas as pd
    try:
        df     = pd.read_excel(core.LOG_FILE)
        total  = len(df)
        live   = len(df[df["Source"] == "LIVE-IDENTIFICATION"])
        upload = len(df[df["Source"] == "UPLOAD"])
        parts  = df["Part Number"].nunique()
        return jsonify({"total": total, "live": live, "upload": upload, "parts": parts})
    except:
        return jsonify({"total": 0, "live": 0, "upload": 0, "parts": 0})


@app.route("/api/recent")
def api_recent():
    import pandas as pd
    try:
        df   = pd.read_excel(core.LOG_FILE)
        rows = df.iloc[::-1].head(5)[["Part Number", "Source", "Confidence"]].fillna("—")
        return jsonify([{
            "part_number": str(r["Part Number"]),
            "source":      str(r["Source"]),
            "confidence":  str(r["Confidence"])
        } for _, r in rows.iterrows()])
    except:
        return jsonify([])


if __name__ == "__main__":
    print("please open -: http://127.0.0.1:5000 ")
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)