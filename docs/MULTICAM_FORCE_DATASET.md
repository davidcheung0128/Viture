# 7-camera + pressure-pad dataset for YOLO-pose force

Collect **synced multi-view RGB + Tekscan (or mock) pressure**, calibrate the pad with ArUco, then export:

1. **YOLO-pose** labels (21 hand keypoints) + per-fingertip kPa sidecars  
2. **PressureVision++** sequences (`meta.json`, `camera_*`, `force/*.pkl`) for force-network finetune  

## Hardware

- 7 cameras (Orbbec Femto Bolt / UVC) on one host (Sync Hub if you have it)
- Tekscan 5330-class pressure pad on the desk
- ArUco markers (**IDs 0–3**) at pad corners: TL, TR, BR, BL — must not cover the sensing area

## Quick start (mock pad — no Tekscan required)

```bash
cd /path/to/Viture
source .venv/bin/activate   # or create venv per README

# 1) See which OpenCV indices work
python scripts/list_cameras.py

# 2) Edit config/multicam_force.yml → set cameras[].index for your 7 cams
#    Keep pad.backend: mock for a dry-run

# 3) Calibrate markers (optional for mock; required for real QC)
python scripts/calibrate_pad_markers.py
# press s when green pad quad looks correct on each camera, n for next, q to quit

# 4) Record a short take (mock force blobs)
python scripts/record_multicam_force.py \
  --participant p01 \
  --action one_finger_index_high \
  --seconds 3 \
  --camera-ids 0   # start with 1 cam; use all 7 when wired

# 5) Build YOLO-pose + PV2 sequences
python scripts/build_force_pose_dataset.py \
  --raw-root data/multicam_force/raw \
  --split train

# 6) QC overlay (after real markers + pad)
python scripts/qc_force_overlay.py \
  --seq data/multicam_force/train/p01/one_finger_index_high \
  --camera-id 0
```

## Real Tekscan

1. Install vendor SDK / Python bindings.  
2. Implement `open()` / `tare()` / `read_frame()` in `forcepad/__init__.py` → `TekscanPressurePad`.  
3. Set `pad.backend: tekscan` in `config/multicam_force.yml` (or `--backend tekscan`).  
4. Document `counts_per_newton` and `pixel_pitch_m` from the datasheet / your scale calibration.  
5. Tare with an empty pad before every session.

## Outputs

```text
data/multicam_force/
  raw/<session>/<participant>/<action>/
    camera_0/00000.jpg …
    camera_6/00000.jpg …
    force/00000.pkl …
    timestamps.json
  train|val/<participant>/<action>/
    meta.json
    camera_*/…
    force/…
  yolo_pose/
    data.yaml
    train|val/
      images/<participant>_<action>/<cam>_<frame>.jpg
      labels/<participant>_<action>/<cam>_<frame>.txt
      force/<participant>_<action>/<cam>_<frame>.json   # tip kPa + contact
calibration/pad_markers.json
data/pose_estimates/<participant>/<action>.pkl
```

### YOLO force sidecar (`*.json`)

```json
{
  "tip_force_kpa": {
    "Right": {"thumb": 0.1, "index": 4.2, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
  },
  "any_contact": true,
  "total_force_newton": 3.5
}
```

Use these as regression targets / auxiliary heads while YOLO-pose learns keypoints. Contact is correct when the warped pressure blob sits under the tip keypoints (check with `qc_force_overlay.py`).

## Suggested actions (same as Viture plan)

Per-finger low/high, multi-finger grip, palm, no_contact — hold 1.5–2 s. Split **train/val by sequence**, not by random frames inside one press. Aim ~10k–15k contact frames.

## Train YOLO-pose (example)

```bash
pip install ultralytics
yolo pose train data=data/multicam_force/yolo_pose/data.yaml model=yolo11n-pose.pt imgsz=640 epochs=50
```

For absolute Newtons in the live Viture tracker, also finetune PressureVision++ on the `train/` sequences starting from `weights/paper_29.pth`.

## Sync rules

- Shared `time.time_ns()` on one machine  
- Pair each camera frame to nearest force frame  
- Drop if `|dt| > sync_tolerance_ms` (default 20 ms)  
- Reject takes with match rate &lt; ~90% or markers missing &gt; 20% of frames  
