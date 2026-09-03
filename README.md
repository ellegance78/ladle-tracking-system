🇬🇧 **English** · [🇹🇷 Türkçe](README.tr.md)

# Ladle Tracking System

Computer-vision system that detects steel ladles on plant CCTV, tracks their
identity across frames, and writes **when each ladle entered and left a
camera's field of view** into a database — replacing a manual, paper-based log.

Built during a summer internship in the process automation department of an
integrated iron and steel plant.

> **Note on data.** All plant footage, camera names, trained weights and zone
> calibration files are confidential and are **not** part of this repository.
> What is published is the code, the architecture and the reasoning. Camera
> identifiers in configuration examples are generic placeholders (`CAM-01`…).

---

## The problem

A ladle carries liquid steel between stations. Knowing when it arrived at and
left each station is operationally useful, but it was being written down by
hand — slow, and full of gaps.

Three things make this harder than a textbook detection task:

![Detection challenges](docs/detection-challenges.png)

Because of these, colour thresholding and background subtraction were ruled out
early in favour of a learned detector.

---

## Architecture

![Architecture](docs/architecture.png)

| Stage | Choice | Why |
|-------|--------|-----|
| Detection | **YOLO11l**, 960 px input | Accuracy over speed — a wrong entry/exit record costs more than a late frame. 640 px lost distant ladles on 2560×1440 sources. |
| Tracking | **ByteTrack** | A ladle shifts from orange to grey as it cools, so appearance-based re-identification (BoT-SORT ReID) is unreliable here. ByteTrack also keeps low-confidence detections, which helps through smoke. |
| Storage | **PostgreSQL** | Several cameras write concurrently; each is a separate service. |
| UI | **Flask** | Zone editor and live monitoring dashboard. |

---

## Two ideas that made it work

### 1 · Zones, and a direction-aware gate

A detection anywhere in the frame is not meaningful. Each camera gets four
hand-drawn zones, stored as JSON with the source resolution so coordinates can
be rescaled per camera:

![Zone types](docs/zone-types.png)

The critical question is what a disappearing track ID means. Get it wrong and
smoke produces a phantom exit every few minutes:

![Gate logic](docs/gate-logic.png)

Ending inside a gate is not enough — the object must also have travelled
outward by a fraction of the frame width (`GATE_DIR_MIN_DX_FRAC`). The system
runs **pass-through** by default: a ladle may enter left and leave right, and
one that genuinely left is never treated as having returned.

### 2 · Merging track fragments

Smoke hides a stationary ladle, the tracker drops it, and on reacquisition
assigns a fresh ID — the classic ID-switch problem. Since appearance can't be
trusted, fragments are merged on **position and time**:

- new fragment starts near where an older one ended, and
- the older fragment did not genuinely exit through a gate, and
- the gap is under `MERGE_TIME_GAP`

Continuous visibility of the **ladle transfer car** is used as extra evidence
that the ladle never left. A noise filter then discards anything shorter than
`MIN_TRACK_SEC`.

---

## The live trial that failed

The system behaved on recorded video, so it was connected to live RTSP streams
for a shift. It did not work. Rather than patch symptoms, every faulty record
was traced back to footage from the same minute:

![Live trial failures](docs/live-trial-failures.png)

All four were fixed in the back end, then the model was retrained with frames
from the cameras it had under-seen. Validated on a continuous 15-hour recording
before going live again:

![Before and after](docs/before-after.png)

Sixteen matched the count made by hand from the video. The second live run
behaved as expected.

---

## Model

![Training configuration](docs/training-config.png)

The dataset was built in two rounds. The first (~790 images) exposed cameras
the model handled poorly; rather than tuning hyperparameters, frames were
collected **specifically** from the failing conditions and the set grew to 835.
Training settings were held identical so the gain could only come from data.

| | Training curves | Confusion matrix | Label distribution |
|---|---|---|---|
| | ![](docs/training-curves.png) | ![](docs/confusion-matrix.png) | ![](docs/label-distribution.jpg) |

`ladle_car` shows more misses than `ladle` — it is usually partly occluded, is
a similar colour to the steelwork behind it, and has fewer examples. It only
plays a supporting role, so this was accepted.

---

## Repository layout

| File | Purpose |
|------|---------|
| `pota_takip_sistemi_v2.py` | Main pipeline — detection, tracking, zones, gates, fragment merging, DB writes |
| `canli_kaynak.py` | RTSP frame grabber; always yields the newest frame so frames never queue up |
| `bolge_editor.py` | Flask zone editor — draw processing zones, gates and exclusion areas on a sample frame |
| `bolge_tarama.py` | Sweeps long recordings to build a detection heatmap, used to place zones by measurement rather than guesswork |
| `pota_dashboard.py` | Flask monitoring dashboard (event table, per-camera filter, live refresh) |
| `db.py` / `report.py` | Database layer and CLI reporting |
| `prepare_dataset.py` | Label Studio export → YOLO format, camera-balanced train/val split |
| `train_pota.py` | Training entry point |
| `track_pota.py` | Single-camera prototype |
| `baslat_tum_kameralar.sh` | Starts one service per camera from `kameralar.json` |

## Running it

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
createdb ladle_db

# offline, on a recording
python pota_takip_sistemi_v2.py track --video kayit.mp4 --camera CAM-01

# live
python pota_takip_sistemi_v2.py track \
  --video "rtsp://<user>:<pass>@<camera-host>:554/stream1" --camera CAM-01

python bolge_editor.py     # zone editor      → :8001
python pota_dashboard.py   # monitoring panel → :8000
```

Camera list and RTSP endpoints go in `kameralar.json`. See `CANLI_KURULUM.md`
for the multi-camera and systemd setup.

## Notes

- Source comments and identifiers are in Turkish; documentation is in English.
- Device selection falls back CUDA → MPS → CPU, warning loudly on CPU since
  YOLO11l cannot keep up with a live stream there.
