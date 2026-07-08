# HAL Aircraft Part Recognition System

## Project Overview

This project is an offline Aircraft Part Recognition System developed for Hindustan Aeronautics Limited (HAL).

The application uses a custom-trained YOLOv8 model to identify aircraft components from images and live webcam feeds. It provides three distinct recognition modes, a searchable recognition log with reference image viewing, and a modular Flask-based architecture.

---

## Features

- **Upload Detection** — Upload an image and identify the part instantly
- **Partial-Manual Identification** — Live camera feed with manual capture; detects all parts in a single frame, shows camera-detected vs user-confirmed details side by side, editable count per part
- **Live Tray Identification** — Continuous 10-second scanning window; detects all unique parts in frame, variable count boxes, editable details, confirm-to-log workflow
- **Searchable Recognition Logs** — Full detection history with search, count column, and reference image viewer (modal popup)
- **Thumb Detection** — Show thumb to camera to resume detection after inactivity timeout
- **Noise Filtering** — Strict confidence threshold (0.75), bounding box area/aspect ratio filters, and frame-consistency checks to reject false positives
- **CSV-aware editing** — Users can edit part details after detection; new part combinations are automatically added to `parts_master.csv`
- **Reference Image Linking** — Each log entry links to the reference photo of the detected part from `raw_images1/`

---

## Part Classes (20 Total)

| # | Class Name | Part Number Prefix |
|---|---|---|
| 1 | CONNECTOR_ASSEM | 201C_282H_7201_001 |
| 2 | BRACKET | 205P_285H_2400_201 |
| 3 | DUCT_ASSEMBLY_1 | 205P_285H_2610_001 |
| 4 | BRACKET_ASSEMBLY | 205P_285H_6200_001 |
| 5 | FRONT_PLATE_ASSEMBLY_RHS | 205P_311H_3016_001 |
| 6 | TOP_BRACKET_LH | 205P_321H_1720_201 |
| 7 | MIP_SUPPORT_BRACKET | 205P_531H_3020_002 |
| 8 | STIFFENER_ROTOR_BRAKE | 205P_531H_3155_001 |
| 9 | ANGLE_SUB_ASSEMBLY | 205P_535H_4289_001 |
| 10 | FAIRING_ASSEMBLY_IN_BOARD_FWD_RH | 205P_551H_1060_002 |
| 11 | RIB_ASSEMBLY_RH | 205P_551H_1230_006 |
| 12 | BUSH_TIE_ROD_ADAPTOR | 205P_636H_0000_295 |
| 13 | OUTLET_DUCT_BOTTOM | 205P_659H_2020_001 |
| 14 | BRACKET_BOTTOM_RH | 205P_711H_1000_206 |
| 15 | FLANGED_COVER_ASSEMBLY | 205P_711H_2006_001 |
| 16 | MESH_COVER_PLATE_RHS | 205P_711H_2040_001 |
| 17 | BRACKET_BASE_RH | 205P_711H_4241_502 |
| 18 | SCOOP_2 | 205P_711H_4550_005 |
| 19 | HAT_FAIRING_BRACKET_ASSEMBLY | 205P_720H_3000_001 |
| 20 | INLET_DUCT | 205P_795H_7300_005 |

---

## Project Structure

```
HAL-Part-Identifier/
│
├── app1.py                         ← Main Flask application entry point
├── core.py                         ← Shared models, camera, metadata, logging
│
├── partial_manual/
│   ├── __init__.py
│   ├── routes.py                   ← Capture & Count routes
│   └── templates/
│       └── partial_manual.html
│
├── live_identification/
│   ├── __init__.py
│   ├── routes.py                   ← Live detection routes with 10s scan window
│   └── templates/
│       └── live_identification.html
│
├── templates/
│   ├── dashboard.html              ← Upload detection + mode selector
│   └── logs.html                  ← Searchable recognition log table
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── modelimg.png                ← HAL logo shown in sidebar
│
├── trained_models/
│   └── retrained.pt                ← YOLOv8 part detection model
│   └── thumb_detector.pt           ← Thumb gesture model (resume detection)
│
├── metadata/
│   └── parts_master.csv            ← Part number, name, issue, yolo_class mapping
│
├── raw_images1/
│   └── <PART_CLASS_NAME>/          ← Reference photos used for log image links
│       └── *.jpg
│
├── uploads/
│   └── backgrounds/                ← Industrial background images for synthetic data
│
├── logs/
│   └── detection_log.xlsx          ← Auto-generated detection history (Excel)
│
├── organize_yolo11.py              ← Organises CVAT YOLO 1.1 export into project structure
│
├── requirements.txt
└── README.md
```

---

## Training Pipeline (separate `retrain/` project)

The model was retrained using a synthetic dataset pipeline:

```
retrain/
├── synthetic_gen.py         ← SyntheticCompositor: background removal + compositing
├── generate_synthetic.py    ← Reads parts_master.csv, generates ~400 images/class
├── prepare_dataset.py       ← 85/15 train/val split, writes dataset.yaml
└── train.py                 ← YOLOv8n training with proven augmentation hyperparameters
```

**Pipeline order:**
```bash
python generate_synthetic.py
python prepare_dataset.py
python train.py
```

After training, copy `trained_models/retrained_v2.pt` into `HAL-Part-Identifier/trained_models/` and update `PART_MODEL_PATH` in `core.py`.

**Key training settings:**
- Base model: `yolov8n.pt`
- Epochs: 60 (with `patience=20` early stopping)
- Image size: 640
- Augmentation: mosaic, mixup, copy-paste, HSV, rotation, scale, shear
- Background removal: `rembg` (AI-based) with GrabCut fallback

---

## Installation

### 1. Clone or copy the project folder

```bash
cd HAL-Part-Identifier
```

### 2. Create virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app1.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Detection Modes

### Upload Detection
- Navigate to Upload Detection from the sidebar
- Upload any image of an aircraft part
- System returns Part Name, Part Number, Issue, and Confidence
- Result is automatically logged

### Partial-Manual Identification
1. Click **Live Detection → Partial-Manual Identification**
2. Point camera at parts
3. Click **Capture & Count**
4. Review the side-by-side **Camera Detected / User Edited** boxes
5. Edit any incorrect details using the searchable dropdowns
6. Remove noise detections using the **✕ Remove** button
7. Click **Confirm & Log** to save to recognition log

### Live Tray Identification
1. Click **Live Detection → Live Identification**
2. Place parts under the camera
3. Click **▶ Start Recognition**
4. Wait 10 seconds for the scan to complete
5. Review detected parts — fields are locked during scanning, editable after
6. Use **🔄 Re-Recognize** to run another scan if needed
7. Click **Confirm & Log** to save

---

## Recognition Log

- Accessible from the sidebar under **Recognition Logs**
- Newest entries shown first
- Search bar filters across all columns in real time
- Columns: Timestamp, Source, Original Details, User Edited Details, Confidence, Count, Reference Image
- Click **🖼 View Image** to open the reference photo in a modal popup

---

## False Positive Filtering

To prevent random objects (paper, tools, hands) from being detected as parts, the system applies four layers of filtering:

1. **Confidence threshold** — minimum 0.75 (rejects weak detections)
2. **Minimum box area** — box must cover at least 1% of the frame
3. **Maximum box area** — box cannot cover more than 70% of the frame (rejects full-frame objects like paper sheets)
4. **Aspect ratio constraint** — rejects boxes with aspect ratio below 0.15 or above 6.5 (rejects extremely thin/wide non-part shapes)
5. **Frame consistency** (live identification only) — a detection must appear in at least 20% of frames during the 10-second window to be counted

---

## Key Files

| File | Purpose |
|---|---|
| `app1.py` | Flask app, upload detection, video feed, log serving |
| `core.py` | YOLO models, camera manager, `save_log()`, metadata |
| `partial_manual/routes.py` | Capture & Count, Confirm & Log for partial-manual mode |
| `live_identification/routes.py` | 10s scan, Re-Recognize, Confirm & Log for live mode |
| `metadata/parts_master.csv` | Ground truth: part_name, part_number, issue, yolo_class |
| `logs/detection_log.xlsx` | Auto-generated; recreated if deleted |
| `organize_yolo11.py` | Converts CVAT YOLO 1.1 export to project folder structure |

---

## Notes

- The application runs fully offline — no internet connection required after initial setup
- `detection_log.xlsx` is created automatically on first run
- If the log file becomes corrupted, delete it and restart — it will regenerate
- The thumb detector (`thumb_detector.pt`) is only active in Live Identification mode
- Dark mode is available from the sidebar on all pages