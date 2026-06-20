
# Traffic Violation CV

**Automated Photo Identification and Classification for Traffic Violations Using Computer Vision**

A scalable, modular pipeline that ingests traffic-camera stills, enhances them,
detects road users, recognises number plates, classifies violations, generates
annotated evidence, and surfaces everything through a searchable web dashboard
with analytics.

---

## 1. The idea (concept note)

Manual review of traffic-camera images is slow, inconsistent, and doesn't
scale. This project automates the full evidence lifecycle:

```
Image -> Enhance -> Detect road users -> Read plates -> Classify violations
      -> Annotate evidence -> Persist (metadata + timestamp) -> Analytics
```

### What makes it different
- **Pluggable violation engine.** Every violation is a self-contained detector
  registered via a decorator (Open/Closed principle). Add a new violation type
  by dropping in one file - the engine needs zero changes.
- **Graceful degradation / "demo mode".** The app boots and the *entire*
  workflow runs even before the heavy ML stack (YOLO + EasyOCR) is installed,
  using a deterministic mock detector. No "white screen" while you wrangle a
  multi-GB Torch install.
- **Honest confidence.** Violations that a single still cannot prove (seatbelt,
  wrong-side) are explicitly flagged as advisory / model-pluggable rather than
  faking certainty. Reviewers stay in the loop.
- **Camera-configurable geometry.** Stop-line position, no-parking polygons and
  carriageway dividers travel in per-image metadata - the same code serves
  many cameras.

---

## 2. Architecture

```
backend/
  config.py              # single source of truth: paths, thresholds, taxonomy
  app.py                 # FastAPI routes (transport + glue only)
  db.py                  # SQLite persistence (evidence + violation tables)
  pipeline/
    types.py             # Detection / Violation / Frame + geometry helpers
    preprocess.py        # low-light, shadow, rain, motion-blur handling
    detect.py            # YOLOv8 wrapper + mock fallback
    plate.py             # EasyOCR plate recognition (+ no-op fallback)
    annotate.py          # evidence image generation
    engine.py            # orchestrates the whole pipeline + metrics
    violations/          # pluggable detectors (one concern per file)
      base.py            # ABC + @register registry
      riders.py          # no_helmet, triple_riding
      traffic.py         # red_light, stop_line
      road.py            # illegal_parking, wrong_side, no_seatbelt
frontend/templates/      # Jinja + HTMX + Tailwind + Chart.js (WCAG 2.2 AA)
scripts/                 # fetch_samples, run_cli, evaluate
tests/                   # pytest smoke + unit tests
```

### Mapping to the brief's tasks
| Brief task | Where it lives |
|---|---|
| Image preprocessing (low-light/rain/shadow/blur) | `pipeline/preprocess.py` |
| Vehicle & road-user detection + classification | `pipeline/detect.py` |
| Violation detection (7 types) | `pipeline/violations/*` |
| Violation classification + confidence | each detector returns `Violation(confidence=...)` |
| License-plate recognition (OCR) | `pipeline/plate.py` |
| Evidence generation (annotated + metadata + timestamp) | `pipeline/annotate.py` + `db.py` |
| Analytics & reporting (stats, trends, searchable records) | `/analytics`, `/records`, `db.stats()` |
| Performance evaluation (Acc/P/R/F1/mAP) | `scripts/evaluate.py` |

### Violation coverage (honest status)
| Violation | Method | Robustness |
|---|---|---|
| Triple riding | geometric (3+ persons on one two-wheeler) | strong |
| Red-light jump | traffic-light HSV state + stop-line crossing | good |
| Stop-line crossing | configurable line geometry | good |
| No helmet | helmet model if provided, else head-region heuristic | strong w/ model, advisory without |
| Illegal parking | vehicle centre inside no-parking polygon | strong (needs zone config) |
| Wrong-side | carriageway-half geometry | advisory (true wrong-side needs motion) |
| No seatbelt | plug-in windscreen model | needs specialised model |

---

## 3. Quick start

```bash
# 1. Create the env (uv-managed Python)
uv venv
uv pip install -e .            # lightweight deps -> runs in demo mode

# 2. (optional) real inference: YOLOv8 + EasyOCR + Torch (multi-GB)
uv pip install -e ".[ml]"

# 3. Generate sample images (or drop your own JPEGs into data/samples/)
python -m scripts.fetch_samples

# 4a. Web dashboard
uvicorn backend.app:app --reload --port 8060
#     open http://127.0.0.1:8060
# 4b. or batch CLI
python -m scripts.run_cli data/samples
```

### Security
All web routes require HTTP Basic auth (except `/api/health`). Set credentials
via env, or a random password is generated and printed to the log at startup:
```bash
export TVCV_USERNAME=admin TVCV_PASSWORD=change-me
export TVCV_MAX_UPLOAD_MB=15   # upload size cap (DoS guard)
```
See [SECURITY.md](SECURITY.md) for the full data/PII notice. No company data,
no hardcoded secrets, and no real license plates are included in this repo.

### Model weights
YOLO and EasyOCR fetch their weights from GitHub automatically on first use.
If your network blocks the automatic download, fetch them manually:
```bash
# YOLOv8n detector weights -> project root
curl -sSL -o yolov8n.pt \
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"

# EasyOCR models -> ~/.EasyOCR/model/
mkdir -p ~/.EasyOCR/model && cd ~/.EasyOCR/model
curl -sSL -o c.zip "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip" && unzip -o c.zip && rm c.zip
curl -sSL -o e.zip "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip" && unzip -o e.zip && rm e.zip
```
`config.py` auto-detects `yolov8n.pt` in the project root and uses it directly.
### Enabling robust helmet detection
Drop a helmet-trained YOLO `.pt` at `models/helmet.pt`. The `NoHelmetDetector`
auto-detects it and switches from heuristic to model-based, high-confidence
output.

### Indian / South-Asian vehicle detector (auto-rickshaw aware)
COCO (default YOLOv8) has **no auto-rickshaw class**, so three-wheelers get
mislabelled as `car` or `motorcycle`. We use a South-Asian YOLOv8 model
(`arabinda91/yolov8-indian-vehicle`, ONNX, 17 classes incl. `CNG` auto-rickshaw,
`Rickshaw`, `Easybike`, `Leguna`, `Motorbike`, `Pedestrian`) as the **primary**
vehicle detector. Its labels are normalised via `config.INDIAN_CLASS_MAP`
(`CNG`/`Easybike` -> `auto_rickshaw`, `Rickshaw` -> `rickshaw`, etc.). Because it
has no traffic-light class, COCO is run as a thin supplement purely to harvest
traffic lights for the red-light detector (`config.SUPPLEMENT_TRAFFIC_LIGHTS`).
Download:
```bash
curl -sL -o models/indian_vehicle.onnx "https://huggingface.co/arabinda91/yolov8-indian-vehicle/resolve/main/yolov8_ods_new_100e_best.onnx"
curl -sL -o models/indian_vehicle_labels.txt "https://huggingface.co/arabinda91/yolov8-indian-vehicle/resolve/main/labels.txt"
```
Toggle with `config.PREFER_INDIAN`. Backend shows as `yolo-indian` in the UI.

### Pretrained models per violation (what's available)
Only 3 of the 7 violations have *standalone* models; the other 4 are rule-based
on top of the YOLOv8 detector (person/vehicle/traffic-light geometry).

| Need | Model | Source | Drop at |
|---|---|---|---|
| Helmet (helmet/head/person) | YOLOv8 motorcycle-helmet | `huggingface.co/JarvanLee/yolov8-helmet-violation-detection` | `models/helmet.pt` |
| Seatbelt (no_seatbelt/seat_belt) | YOLOv11s classifier | `huggingface.co/RISEF/yolov11s-seatbelt` | `models/seatbelt.pt` |
| License-plate detector | YOLOv5m ANPR | `huggingface.co/keremberke/yolov5m-license-plate` | `models/plate.pt` |
| Plate OCR | EasyOCR (craft + english_g2) | bundled on install | `~/.EasyOCR/model/` |
| Triple-riding / stop-line / red-light / wrong-side / illegal-parking | none needed | derived from YOLOv8 detections + geometry | n/a |

Download from HuggingFace:
```bash
curl -sSL -o models/helmet.pt "https://huggingface.co/JarvanLee/yolov8-helmet-violation-detection/resolve/main/weights/best.pt"
curl -sSL -o models/plate.pt  "https://huggingface.co/keremberke/yolov5m-license-plate/resolve/main/best.pt"
```
The pipeline auto-detects both at startup (`ocr` reports `easyocr+yolov5_detector`).

### Deep image restoration (Restormer) - heavy-weather robustness
Optional deep de-raining + motion-deblurring on top of the fast classical
preprocessing. Opt-in per request (slow on CPU). Download the official weights
from the Restormer GitHub release:
```bash
curl -sSL -o models/restormer_deraining.pth \
  "https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth"
curl -sSL -o models/restormer_motion_deblurring.pth \
  "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth"
```
The architecture (`backend/pipeline/restormer_arch.py`) is vendored from the
official repo. Enable per upload via the **De-rain / De-blur** checkboxes in the
dashboard, or `meta={"deep_derain": True, "deep_deblur": True}` in code. Restores
at `config.RESTORE_MAX_SIDE` (default 720px) to keep CPU inference tractable.

---

## 4. Performance evaluation

Provide ground truth as `data/ground_truth.json`:
```json
{ "sample_1.jpg": {"violations": ["triple_riding", "no_helmet"]},
  "sample_2.jpg": {"violations": []} }
```
Then:
```bash
python -m scripts.evaluate data/samples data/ground_truth.json
```
Outputs image-level exact-match accuracy plus per-class Precision, Recall, F1,
and a macro average. The engine also returns per-stage timings (preprocess /
detect / OCR / violations / total) on every request for efficiency profiling.

---

## 5. Scaling notes
- Stateless pipeline + SQLite for the prototype; swap `db.py` for Postgres and
  run multiple `uvicorn` workers behind a load balancer for production.
- Detection is the bottleneck - move to `yolov8s/m` on GPU, or batch frames.
- Plate numbers are PII: `data/` is git-ignored; add encryption-at-rest and
  access controls before any real deployment.

---

## 6. Testing
```bash
python -m pytest -q     # runs in demo mode, no GPU/ML stack required
```

Built as a prototype - classical-CV preprocessing + YOLOv8 detection +
EasyOCR + a clean, extensible violation-detector framework.
