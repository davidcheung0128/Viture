# planme.md — V2: 7 cameras + pressure pad → train force-aware pose

Personal plan to **link**, **calibrate**, **collect**, and **train** so the model knows both **hand pose** and **contact pressure**.

**Branch:** `V2`  
**Goal:** Each training sample pairs multi-view RGB (up to 7 cams) with a pressure-pad map, YOLO-pose keypoints, and per-fingertip force (kPa / Newtons).

---

## 0. What you will produce

```text
data/multicam_force/
  raw/<session>/<participant>/<action>/
    camera_0/… camera_6/
    force/*.pkl
    timestamps.json
  train|val/<participant>/<action>/
    meta.json
    camera_*/…
    force/…
  yolo_pose/
    data.yaml
    train|val/images|labels|force/
calibration/pad_markers.json
forcepad/tekscan_device.py          # your live pad driver
weights/…                           # YOLO-pose + optional PV2 finetune
```

---

## 1. Hardware checklist

| Item | Status |
|------|--------|
| 7 cameras (Orbbec Femto Bolt / UVC) on one host | |
| Sync Hub / stable USB power (no black feeds) | |
| Pressure pad (Tekscan 5330-class or equivalent) | |
| ArUco markers IDs **0–3** at pad corners (TL, TR, BR, BL) | |
| Same PC for cams + pad (shared clock for sync) | |
| Stable indoor lighting, pad not covered by markers | |
| Disk space (~50–100 GB for first sessions) | |

Measure once: pad active size (mm), grid rows×cols, counts→Newtons scale.

---

## 2. Software checklist

```bash
git fetch origin && git checkout V2
git submodule update --init --recursive
# venv + README deps: opencv, mediapipe, numpy, pyyaml, torch…
```

| Step | Command / file |
|------|----------------|
| Probe cams | `python scripts/list_cameras.py` |
| Set indices | `config/multicam_force.yml` → `cameras[].index` |
| Dry-run link | `python scripts/v2_collect.py --dry-run --participant p01` |
| Wire Tekscan | implement `forcepad/tekscan_device.py` |
| Set backend | `pad.backend: tekscan` (or `--backend tekscan`) |

---

## 3. Calibration (once per pad frame)

1. Place ArUco 0–3 at pad corners (not on sensing area).  
2. Run:

   ```bash
   python scripts/calibrate_pad_markers.py
   ```

3. For each camera: wait for green pad quad → press **`s`**, then **`n`**.  
4. Saves `calibration/pad_markers.json` (quad + homography + counts scale).  

Re-run only if the pad frame moves. During recording, prefer **per-frame** marker detection (head/camera motion); static calib is fallback only.

---

## 4. How much data to collect

### Frame budget (first useful train)

| Split | Contact frames | No-contact |
|-------|----------------|------------|
| Train | 8,000–12,000 | 1,500–2,500 |
| Val | 2,000–3,000 | 400–600 |
| **Total** | **~10k–15k** | **~2k–3k** |

~15–20 FPS after sync ≈ **12–20 minutes** of real contact time → plan **2–4 sessions**.

### Action list (record as separate takes)

| Action | Takes | Hold |
|--------|-------|------|
| `one_finger_*_low/high` (thumb→pinky) | 8–10 each | 1.5–2 s |
| `multi_finger_grip_low/high` | 10–12 | 2 s |
| `palm_press` | 6–8 | 2 s |
| `no_contact` | 8–10 | 2–3 s |

Vary angle, distance (~30–60 cm), lighting, and pad region. Left **and** Right if both will be used live.  
**Split rule:** train/val by **sequence/session**, never random frames inside one press.

---

## 5. Collect (day-of)

### Before each session

1. Warm up + **tare** pad (empty).  
2. `python scripts/list_cameras.py` — 7 non-black feeds.  
3. Markers visible from the cams you’ll use.  
4. Session id, e.g. `2026-08-19_p01_s01`.  

### Record

```bash
python scripts/v2_collect.py --participant p01 --backend tekscan --auto-build
```

Live keys: `1`–`9`/`a`–`e` actions · `t` tare · `b` build · `q` quit · Space stop take.

Or scripted:

```bash
python scripts/record_multicam_force.py \
  --participant p01 --action one_finger_index_high --seconds 3
```

### Spot-check every take

- RGB not black  
- Force blob under the intended finger  
- Markers visible most frames  
- Delete takes with markers lost &gt;20% or RGB/force mismatch  

---

## 6. Sync + build training set

```bash
python scripts/build_force_pose_dataset.py \
  --raw-root data/multicam_force/raw --split train
# later: --split val for held-out sessions
```

Sync: nearest force frame to each camera timestamp; drop if `|dt| > 20 ms`; reject sequence if match rate &lt; ~90%.

Exports:

1. **YOLO-pose** — 21 hand keypoints + `force/*.json` (`tip_force_kpa`)  
2. **PV2 sequences** — `meta.json` + warped force for PressureVision++  

QC:

```bash
python scripts/qc_force_overlay.py \
  --seq data/multicam_force/train/p01/one_finger_index_high --camera-id 0
```

Accept only if the pressure overlay sits under the contacting fingertip(s).

---

## 7. Train the model

### YOLO-pose (+ force sidecars)

```bash
pip install ultralytics
yolo pose train data=data/multicam_force/yolo_pose/data.yaml \
  model=yolo11n-pose.pt imgsz=640 epochs=50
```

Use `tip_force_kpa` JSON as regression / auxiliary targets so the model learns pressure, not only pose.

### Optional: PressureVision++ finetune

Start from `weights/paper_29.pth` on the `train/` sequences → e.g. `weights/viture_tekscan_ft.pth`.  
Pick checkpoint by contact IoU + force MAE. Point the live tracker at the new weights.

### Pass criteria

- [ ] Overlay QC: blob under correct fingertip on single-finger val takes  
- [ ] Offline: tip force tracks pad GT (not random)  
- [ ] Live: flat press moves the **correct** finger; air ≈ 0%  

---

## 8. End-to-end order (short)

1. Checkout `V2`; install deps.  
2. Link 7 cams (`list_cameras` → edit config).  
3. Dry-run `v2_collect.py --dry-run`.  
4. Wire Tekscan → calibrate ArUco.  
5. Record ~10k–15k contact frames + empties.  
6. Build + QC overlays.  
7. Train YOLO-pose (and optional PV2).  
8. Validate live presses.  

---

## 9. Files to touch when something breaks

| Problem | Where |
|---------|--------|
| Wrong camera indices | `config/multicam_force.yml` |
| Pad not reading | `forcepad/tekscan_device.py` |
| Bad alignment | `scripts/calibrate_pad_markers.py` / markers |
| Sync drops | `recording.sync_tolerance_ms` in config |
| Empty YOLO keypoints | MediaPipe / lighting / hand in view |
| Absolute Newtons wrong | `counts_per_newton`, `pixel_pitch_m` |

---

## 10. Out of scope (later)

- Hardware genlock / cross-machine NTP  
- EMG / muscle activation labels  
- Full Depth + RGB-D fusion beyond RGB force warp  

---

## Quick commands cheat sheet

```bash
git checkout V2
python scripts/list_cameras.py
python scripts/calibrate_pad_markers.py
python scripts/v2_collect.py --participant p01 --backend tekscan --auto-build
python scripts/qc_force_overlay.py --seq data/multicam_force/train/p01/<action> --camera-id 0
yolo pose train data=data/multicam_force/yolo_pose/data.yaml model=yolo11n-pose.pt
```

More detail: [`V2_README.md`](V2_README.md) · [`docs/MULTICAM_FORCE_DATASET.md`](docs/MULTICAM_FORCE_DATASET.md)
