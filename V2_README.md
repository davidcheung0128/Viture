# V2 — Multi-camera + pressure-pad data collection

**Branch `V2`** links **7 cameras** and a **pressure pad**, records synced RGB + force, and exports training data for YOLO-pose (keypoints + tip force) and PressureVision++.

## What V2 adds

| Piece | Role |
|-------|------|
| `scripts/v2_collect.py` | One app: link cams + pad → live preview → record takes → optional build |
| `scripts/list_cameras.py` | Probe OpenCV indices / save previews |
| `scripts/calibrate_pad_markers.py` | ArUco pad corners for force↔image alignment |
| `scripts/record_multicam_force.py` | Headless / scripted recorder |
| `scripts/build_force_pose_dataset.py` | Sync + YOLO-pose labels + force JSON + PV2 sequences |
| `scripts/qc_force_overlay.py` | Check pressure blob under fingertips |
| `forcepad/` | `mock` (dry-run) or `tekscan` (wire your SDK) |
| `config/multicam_force.yml` | Camera indices, pad scale, sync tolerance |

## Setup

```bash
git fetch origin
git checkout V2
git submodule update --init --recursive
# venv + deps from README (opencv, mediapipe, numpy, pyyaml, torch…)
```

## Collect training data

### 1. Dry-run (no hardware)

```bash
python scripts/v2_collect.py --dry-run --participant p01 --seconds 2 --auto-build
```

### 2. Real 7 cameras + mock pad (pipeline test)

```bash
python scripts/list_cameras.py
# Edit config/multicam_force.yml → set cameras[0..6].index to your 7 feeds

python scripts/v2_collect.py --participant p01 --backend mock
```

### 3. Real Tekscan pad

1. Implement `TekscanPressurePad` in `forcepad/__init__.py` (`open` / `tare` / `read_frame`).
2. Set `pad.backend: tekscan` in config (or `--backend tekscan`).
3. Tare with an empty pad (`t` in the live UI).

```bash
python scripts/calibrate_pad_markers.py   # ArUco IDs 0–3 at pad corners
python scripts/v2_collect.py --participant p01 --backend tekscan --auto-build
```

### Live keys

- `1`–`9`, `a`–`e` — scripted press actions (thumb/index/…/grip/palm/no_contact)
- `r` — custom action name  
- `t` — tare pad  
- `b` — build dataset now  
- `q` — quit  
- During a take: `Space` stop early, `q` abort  

## Train

```bash
python scripts/build_force_pose_dataset.py --raw-root data/multicam_force/raw --split train
# optional val takes → --split val

pip install ultralytics
yolo pose train data=data/multicam_force/yolo_pose/data.yaml model=yolo11n-pose.pt imgsz=640 epochs=50
```

Force sidecars live next to labels:

`data/multicam_force/yolo_pose/train/force/<participant>_<action>/<cam>_<frame>.json`

Each JSON has `tip_force_kpa` for thumb/index/middle/ring/pinky — use as regression targets so the model learns pressure, not only pose.

## Force correctness checklist

1. Pad scale (`counts_per_newton`, `pixel_pitch_m`) matches your sheet  
2. ArUco markers visible; QC overlay puts the blob under the contacting fingertip  
3. Sync `|dt| ≤ 20 ms` (default); reject takes with match rate &lt; 90%  
4. Split train/val by **session/sequence**, not random frames inside one press  

Full detail: [`docs/MULTICAM_FORCE_DATASET.md`](docs/MULTICAM_FORCE_DATASET.md)
