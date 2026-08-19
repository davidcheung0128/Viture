# Viture Grip Tension Tracker

Live **per-fingertip press / grip** readout from a camera (Viture XR glasses or desk webcam), using MediaPipe hand tracking 

**What you get today (no custom Tekscan dataset required):**

- Tracks **both Left and Right** hands
- Per-finger bars (Thumb / Index / Middle / Ring / Pinky) for each hand
- PressureVision++ contact signal when it fires; otherwise a **pose-curl fallback** so bars still move when you grip
- Optional projection into Viture via **SpaceWalker**

**Limits:** This estimates **contact / grip effort from RGB**, not lab-grade Newtons and not EMG muscle activation. Absolute force gets accurate after you finetune on Orbbec + Tekscan data.

---

## Branch

All tracker code lives on:

```text
cursor/viture-tension-tracker-4b6d
```

`main` only has a stub README. Always use this branch.

---

## Fresh Mac setup (one-time)

Run **one line at a time** in Terminal (zsh does not treat `#` as comments by default).

### 1. Clone

```bash
cd ~
git clone -b cursor/viture-tension-tracker-4b6d https://github.com/davidcheung0128-ai/Viture.git
cd Viture
git submodule update --init --recursive
ls viture_tension_tracker.py
ls weights/hand_landmarker.task
```

### 2. Python venv + packages

```bash
cd ~/Viture
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e external/segmentation_models.pytorch
python -m pip install -r external/pressurevision2/requirements.txt
python -m pip install -r requirements.txt
```

### 3. Download PressureVision++ weights

```bash
cd ~/Viture
mkdir -p weights
curl -L -o weights/paper_29.pth "https://www.dropbox.com/scl/fi/0r2koefy7bhr66dffc8z7/paper_29.pth?rlkey=wshcxm8iy8l1qo60oo7khdqjp&dl=1"
ls -lh weights/paper_29.pth
```

You want ~148MB. Do **not** type `weights/paper_29.pth` alone in the shell — that is a file path, not a command.

`weights/hand_landmarker.task` is already in the repo (~7.5MB). If missing:

```bash
curl -L -o weights/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

### 4. macOS Camera permission

**System Settings → Privacy & Security → Camera → enable Terminal** (or iTerm). Quit and reopen Terminal after enabling.

---

## Already cloned? Update to latest

```bash
cd ~/Viture
source .venv/bin/activate
git fetch origin
git checkout cursor/viture-tension-tracker-4b6d
git pull origin cursor/viture-tension-tracker-4b6d
git submodule update --init --recursive
ls viture_tension_tracker.py
```

If checkout fails with “local changes would be overwritten”:

```bash
cd ~/Viture
cp weights/paper_29.pth /tmp/paper_29.pth
git fetch origin
git checkout -f cursor/viture-tension-tracker-4b6d
git reset --hard origin/cursor/viture-tension-tracker-4b6d
git submodule update --init --recursive
mkdir -p weights
cp /tmp/paper_29.pth weights/paper_29.pth
```

---

## Everyday launch

```bash
cd ~/Viture
source .venv/bin/activate
git fetch origin
git checkout cursor/viture-tension-tracker-4b6d
git pull origin cursor/viture-tension-tracker-4b6d
git submodule update --init --recursive
ls viture_tension_tracker.py
```

### A. Pick the correct camera (skip iPhone Continuity)

```bash
python viture_tension_tracker.py --camera-index 0 --device mps --project-glasses
```

Or:

```bash
python viture_tension_tracker.py --auto-camera --device mps --project-glasses
```

(`--auto-camera` skips Continuity/iPhone by name when ffmpeg can list devices.)

### B. Show the window in the Viture glasses

Viture is two devices:

1. **Camera in** — `--camera-index`
2. **Display out** — SpaceWalker / extended display

`--project-glasses` does **not** auto-inject into the lenses. Do this:

1. Open **SpaceWalker**
2. Start the tracker (window on the Mac)
3. Pin / capture **Viture Grip Tension Tracker** into XR
4. Focus that window and press **`f`** to fullscreen in the glasses
5. Press **`q`** to quit

### C. What you should see

- Green boxes around hands
- Color dots on fingertips
- **LEFT hand** panel (left side) and **RIGHT hand** panel (right side)
- Bottom bar: peak fingertip press across both hands
- Debug line with `src=pose-proxy` or `src=pv2+pose`

**Tip:** Squeeze a mouse or curl fingers — bars should move (pose proxy). Flat light presses may stay low until Tekscan finetuning. Air poses ≈ 0%.

---

## Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--camera-index N` | `0` | OpenCV camera index |
| `--list-cameras` | off | Probe cameras + save preview JPEGs |
| `--auto-camera` | off | Prefer non-Continuity, non-black camera |
| `--allow-continuity` | off | Allow iPhone Continuity Camera |
| `--device` | auto | `mps` / `cuda` / `cpu` |
| `--project-glasses` | off | SpaceWalker projection help; press `f` after pinning |
| `--gain` | `2.0` | Boost contact-probability map |
| `--smooth` | `0.5` | EMA smoothing for finger bars |
| `--fpv-adaptive` | on | Adaptive egocentric scaling |
| `--no-fpv-adaptive` | — | Use fixed `--max-force` only |
| `--max-force` | `16` | Scale for fixed-force mode |
| `--show-skeleton` | off | Draw MediaPipe skeleton |
| `--mirror` | off | Flip camera horizontally |
| `--weights` | `weights/paper_29.pth` | Model checkpoint |

---

## Expected layout

```text
Viture/
├── viture_tension_tracker.py
├── requirements.txt
├── README.md
├── weights/
│   ├── paper_29.pth              # download once
│   ├── hand_landmarker.task      # shipped in repo
│   └── camera_previews/          # from --list-cameras
├── .venv/
└── external/
    ├── pressurevision2/
    └── segmentation_models.pytorch/
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `can't open file '.../viture_tension_tracker.py'` | Wrong branch. `git checkout cursor/viture-tension-tracker-4b6d && git pull` |
| GitHub `/edit/...` clone URL fails | Use `https://github.com/davidcheung0128-ai/Viture.git` with `-b cursor/viture-tension-tracker-4b6d` |
| `command not found: pip` / `zsh: unknown file attribute` | Use venv + `python -m pip`; run commands one line at a time |
| Camera permission / can't open index | Enable Camera for Terminal; quit Zoom/FaceTime; replug USB |
| Black window / forever Searching for hands | Wrong camera — `--list-cameras`, avoid iPhone Continuity |
| Window on Mac but not in glasses | SpaceWalker → pin tracker window → press `f` |
| All fingers stuck at 0% | Pull latest. Grip/curl should move bars (`src=pose-proxy`). Flat press needs Tekscan finetune for real Newtons |
| Only one hand’s bars | Pull latest — LEFT and RIGHT panels are separate |
| `No module named 'pretrainedmodels'` | `python -m pip install -r requirements.txt` |
| `weights_only` UnpicklingError | Pull latest (`weights_only=False` is set) |
| `mediapipe has no attribute solutions` | Pull latest (Tasks HandLandmarker) |
| SSL error downloading hand model | Use shipped `weights/hand_landmarker.task` or `curl -L -o ...` |

---

## How it works (short)

1. Capture frames from the selected camera  
2. MediaPipe detects up to two hands + fingertips (Left / Right)  
3. Each hand is cropped and run through PressureVision++  
4. Contact probability is mapped to each fingertip; if PV2 is dead, finger-curl pose proxy fills in  
5. UI shows dual-hand panels + peak tension bar  

---

## V2 — 7 cameras + pressure pad → train data

On branch **`V2`**, link seven cameras and a pressure pad, record synced takes, and export YOLO-pose + force labels:

```bash
git checkout V2
python scripts/v2_collect.py --dry-run --participant p01 --auto-build   # no hardware
python scripts/list_cameras.py                                         # map your 7 cams
python scripts/v2_collect.py --participant p01 --backend mock          # real cams
# after wiring Tekscan SDK:
python scripts/calibrate_pad_markers.py
python scripts/v2_collect.py --participant p01 --backend tekscan --auto-build
```

Start here: [`V2_README.md`](V2_README.md). Details: [`docs/MULTICAM_FORCE_DATASET.md`](docs/MULTICAM_FORCE_DATASET.md).

## Later: custom Orbbec + Tekscan training

When your 7× Femto Bolt + Sync Hub Pro + Tekscan 5330 rig is ready, collect synced multi-view RGB-D + pressure ground truth (scripts above) and finetune `paper_29.pth` / YOLO-pose for accurate absolute per-finger force. Until then, use this live tracker for dual-hand relative grip feedback.
