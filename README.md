# HAL-Part-Identifier
### Offline Aircraft Part Recognition System — Hindustan Aeronautics Limited

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.1-lightgrey?logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Offline](https://img.shields.io/badge/Network-Fully%20Offline-blueviolet)

---
<img width="1917" height="1012" alt="image" src="https://github.com/user-attachments/assets/6ecb462a-457d-4131-9688-4bfd60ebc708" />
<img width="1917" height="997" alt="image" src="https://github.com/user-attachments/assets/1cdb4b71-7360-453f-b6e1-c37936e27478" />
<img width="1917" height="1010" alt="image" src="https://github.com/user-attachments/assets/ffca0e36-8daa-4b22-b4f2-7e52c31fe018" />



## What This Project Does

HAL-Part-Identifier is a **fully offline** computer vision web application that identifies aerospace components used in helicopter assembly at Hindustan Aeronautics Limited (HAL). A technician points a webcam at a part tray or uploads a photo, and the system instantly returns the Part Number, Part Name, and Issue Letter — the three identifiers HAL uses to track every component in its supply chain.

The system was built for and deployed within **HAL's air-gapped internal network**, where no cloud services or internet APIs are available. All inference runs locally on-device using a custom-trained YOLOv8 model.

---

## Demo

### Live Identification — 10-Second Auto-Scan
Demo videos for both provided within this project folder named "live tray identification (1).mp4" and "partial manual identification (1).mp4"
### Partial-manual Identification
Demo videos for both provided within this project folder named "live tray identification (1).mp4" and "partial manual identification (1).mp4"

> Demos recorded on the actual system against HAL helicopter part dataset (20 classes, live webcam).

---

## Industry Impact

### The problem this solves

In aerospace manufacturing, every component on an assembly line must be **correctly identified, counted, and logged** before it moves to the next stage. At HAL's Helicopter Division, this has traditionally been a manual process — a technician visually inspects a part, cross-references a physical catalogue or document, and records it by hand. This introduces:

- **Human error** in identification, especially for visually similar parts (e.g. multiple bracket variants)
- **Slow throughput** on the shop floor — each lookup can take minutes
- **Traceability gaps** — paper-based or informal logging makes audits difficult
- **No scalability** — the same bottleneck repeats for every part, every shift

### What this system changes

| Before | After |
|---|---|
| Manual visual inspection + catalogue lookup | Sub-second camera-based identification |
| Paper or informal logging | Timestamped Excel log with confidence scores, source, and count |
| No reference cross-check | Every log entry links to a reference photo of the part |
| Errors discovered downstream | User-editable results with Camera Detected vs User Confirmed audit trail |
| No count tracking | Per-part count with editable fields and total count display |

In a production environment, a system like this running at a part intake station could **reduce part identification time from minutes to seconds**, catch misidentified parts before they enter assembly, and provide a structured digital log for quality audits — directly supporting HAL's traceability requirements under aerospace manufacturing standards.

### Why offline matters

Aerospace manufacturing facilities — especially defence organisations like HAL — operate under strict network security policies. Cloud-based AI inference is not an option. This project demonstrates that **production-grade computer vision can be deployed entirely offline**, with no dependency on external APIs, internet connectivity, or cloud infrastructure. All model inference, logging, and serving happens locally.

---

## Technical Highlights

- **Custom YOLOv8 model** trained on a synthetic composite dataset — reference part photos background-removed with `rembg` and composited onto varied industrial backgrounds, forcing the model to learn part geometry rather than background context
- **Three detection modes** covering different operational scenarios (single upload, manual capture + count, automated tray scan)
- **Multi-layer false positive filtering** — confidence threshold, bounding box area ratio, aspect ratio constraints, and frame-consistency checks working in combination
- **Thread-safe architecture** — `model_lock` and `log_lock` prevent race conditions across concurrent Flask requests and camera threads
- **Modular Flask Blueprint design** — each detection mode is a self-contained Blueprint; `core.py` acts as a shared service layer (models, camera, metadata, logging)
- **True physical count detection** — live identification tracks the maximum number of boxes seen simultaneously in a single frame (not frame frequency), giving accurate per-class counts
- **Full audit trail** — every logged entry stores Original Detected Details and User Confirmed Details separately, preserving both machine output and human verification

---

## Features

- **Upload Detection** — Upload a part image; returns Part Name, Part Number, Issue, and Confidence. Auto-logged.
- **Partial-Manual Identification** — Live camera feed with Capture & Count; detects all parts in a single captured frame. Camera Detected vs User Edited two-column layout. Editable count, searchable dropdowns, delete noise detections.
- **Live Tray Identification** — 10-second scan window with Start/Re-Recognize controls. Fields locked during scanning, editable after. True count from simultaneous box detection. Confirm & Log on user confirmation only.
- **Searchable Recognition Log** — Newest-first detection history. Live search across all columns. Original Details vs User Edited Details columns. Count column. Reference image modal popup.
- **Thumb Detection** — Show thumb to camera to resume detection after 5-second inactivity timeout (Live Identification only).
- **Noise Filtering** — 5-layer false positive suppression: confidence, box area, aspect ratio, and frame consistency.
- **CSV-aware editing** — New part combinations entered by users are automatically appended to `parts_master.csv`.
- **Dark Mode** — Available from sidebar on all pages.

---

## Part Classes (20 Total)

| # | Class Name | Part Number Prefix | Issue |
|---|---|---|---|
| 1 | CONNECTOR_ASSEM | 201C_282H_7201_001 | A |
| 2 | BRACKET | 205P_285H_2400_201 | A |
| 3 | DUCT_ASSEMBLY_1 | 205P_285H_2610_001 | C |
| 4 | BRACKET_ASSEMBLY | 205P_285H_6200_001 | A |
| 5 | FRONT_PLATE_ASSEMBLY_RHS | 205P_311H_3016_001 | B |
| 6 | TOP_BRACKET_LH | 205P_321H_1720_201 | B |
| 7 | MIP_SUPPORT_BRACKET | 205P_531H_3020_002 | A |
| 8 | STIFFENER_ROTOR_BRAKE | 205P_531H_3155_001 | B |
| 9 | ANGLE_SUB_ASSEMBLY | 205P_535H_4289_001 | A |
| 10 | FAIRING_ASSEMBLY_IN_BOARD_FWD_RH | 205P_551H_1060_002 | A |
| 11 | RIB_ASSEMBLY_RH | 205P_551H_1230_006 | A |
| 12 | BUSH_TIE_ROD_ADAPTOR | 205P_636H_0000_295 | C |
| 13 | OUTLET_DUCT_BOTTOM | 205P_659H_2020_001 | C |
| 14 | BRACKET_BOTTOM_RH | 205P_711H_1000_206 | B |
| 15 | FLANGED_COVER_ASSEMBLY | 205P_711H_2006_001 | A |
| 16 | MESH_COVER_PLATE_RHS | 205P_711H_2040_001 | C |
| 17 | BRACKET_BASE_RH | 205P_711H_4241_502 | A |
| 18 | SCOOP_2 | 205P_711H_4550_005 | A |
| 19 | HAT_FAIRING_BRACKET_ASSEMBLY | 205P_720H_3000_001 | A |
| 20 | INLET_DUCT | 205P_795H_7300_005 | B |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (Flask UI)                  │
│   Upload Page │ Partial-Manual │ Live ID │ Logs         │
└──────────┬────────────┬────────────┬───────────────┬───┘
           │            │            │               │
    ┌──────▼──────┐ ┌───▼────────┐ ┌▼────────────┐  │
    │  app1.py    │ │partial_    │ │live_        │  │
    │  (Upload,   │ │manual/     │ │identification│  │
    │  Video Feed │ │routes.py   │ │/routes.py   │  │
    │  Logs)      │ │(Capture &  │ │(10s scan,   │  │
    └──────┬──────┘ │Count,      │ │Re-Recognize,│  │
           │        │Confirm)    │ │Confirm)     │  │
           └────────┴────────────┴─────┬─────────┘  │
                                       │             │
                    ┌──────────────────▼──────────┐  │
                    │           core.py            │  │
                    │  • YOLO model (model_lock)   │  │
                    │  • Camera manager            │  │
                    │  • parts_master.csv          │  │
                    │  • save_log() (log_lock)     │  │
                    │  • get_reference_image()     │  │
                    └──────────────────────────────┘  │
                                                      │
                    ┌─────────────────────────────────▼┐
                    │       detection_log.xlsx          │
                    │  (Timestamp, Source, Original,    │
                    │   Confirmed, Confidence, Count,   │
                    │   Reference Path)                 │
                    └──────────────────────────────────┘
```

---

## Project Structure

```
HAL-Part-Identifier/
│
├── app1.py                         ← Flask entry point; upload detection, video feed
├── core.py                         ← Shared models, camera, metadata, logging
│
├── partial_manual/
│   ├── __init__.py
│   ├── routes.py                   ← Capture & Count, Confirm & Log
│   └── templates/
│       └── partial_manual.html
│
├── live_identification/
│   ├── __init__.py
│   ├── routes.py                   ← 10s scan, Re-Recognize, Confirm & Log
│   └── templates/
│       └── live_identification.html
│
├── templates/
│   ├── dashboard.html              ← Upload page + mode selector
│   └── logs.html                  ← Searchable log table
│
├── static/css/style.css
├── trained_models/
│   ├── retrained.pt                ← YOLOv8 part detection model
│   └── thumb_detector.pt           ← Thumb gesture resume model
│
├── metadata/
│   └── parts_master.csv            ← Part number / name / issue / yolo_class
│
├── raw_images1/<PART_CLASS>/       ← Reference photos for log image links
├── logs/detection_log.xlsx         ← Auto-generated detection history
├── organize_yolo11.py              ← CVAT YOLO 1.1 export → project structure
├── requirements.txt
└── README.md
```

---

## Training Pipeline

Model was trained in a **separate `retrain/` project** using a synthetic composite dataset pipeline:

```
retrain/
├── synthetic_gen.py         ← SyntheticCompositor: rembg BG removal + compositing
├── generate_synthetic.py    ← Reads parts_master.csv → ~400 synthetic images/class
├── prepare_dataset.py       ← 85/15 train/val split, writes dataset.yaml
└── train.py                 ← YOLOv8n, 60 epochs, proven augmentation config
```

**Why synthetic data?** Collecting real annotated aerospace parts at scale is impractical in a secure facility. By removing backgrounds with `rembg` and compositing parts onto varied industrial backgrounds, the model learns part geometry independent of context — preventing the background-bias that causes models trained on real photos to overfit to their shooting environment.

**Key settings:**
- Base: `yolov8n.pt` | Epochs: 60 | Early stopping: patience=20 | Image size: 640
- Augmentation: mosaic 1.0, mixup 0.15, copy-paste 0.3, HSV, rotation ±10°, scale 0.5, shear 2°

---

## Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/HAL-Part-Identifier.git
cd HAL-Part-Identifier

# 2. Virtual environment (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app1.py
```

Open **http://127.0.0.1:5000** in your browser.

> If you get a PowerShell execution policy error on step 2:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

---

## False Positive Filtering

Five layers applied at every inference call to prevent random objects from being misidentified:

| Layer | Filter | Threshold |
|---|---|---|
| 1 | Confidence threshold | ≥ 0.75 |
| 2 | Minimum bounding box area | ≥ 1% of frame |
| 3 | Maximum bounding box area | ≤ 70% of frame |
| 4 | Aspect ratio constraint | 0.15 – 6.5 |
| 5 | Frame consistency (live only) | Must appear in ≥ 20% of frames |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.1.1 + Blueprint architecture |
| Object detection | YOLOv8 (Ultralytics 8.4.56) |
| Image processing | OpenCV 4.12, Pillow 11.3 |
| Data | Pandas 2.3, NumPy 2.2, openpyxl 3.1 |
| Synthetic data | rembg, scikit-image, numba |
| Concurrency | Python threading with model_lock + log_lock |
| Storage | CSV (metadata), Excel (logs) — no database required |
| Deployment | Fully offline, no internet dependency |

---

## Key Files

| File | Purpose |
|---|---|
| `app1.py` | Flask app, upload detection, `/serve_ref_image`, log route |
| `core.py` | Models, camera, `save_log()`, `get_reference_image()`, locks |
| `partial_manual/routes.py` | Capture & Count, Confirm & Log |
| `live_identification/routes.py` | 10s scan, Re-Recognize, Confirm & Log, true count logic |
| `metadata/parts_master.csv` | Ground truth: part_name, part_number, issue, yolo_class |
| `logs/detection_log.xlsx` | Auto-generated; safe to delete (recreates on restart) |
| `organize_yolo11.py` | Converts CVAT YOLO 1.1 export to project folder structure |

---

## Notes

- Runs fully offline — no internet required after initial `pip install`
- `detection_log.xlsx` recreates automatically if deleted
- Thumb detector is only active in Live Identification mode
- Dark mode available from sidebar on all pages
- New part combinations edited by users are automatically persisted to `parts_master.csv`

---
## Author

Developed during internship at **Hindustan Aeronautics Limited (HAL), Helicopter Division**
as part of the HAL-Part-Identifier (HAL-AIS) project.
The content presented here is intended solely to demonstrate the architecture, workflow, technologies, and engineering concepts explored during the internship while fully respecting the confidentiality obligations of Hindustan Aeronautics Limited (HAL).
