# Viture Grip Tension Tracker

Estimate **grip / muscle tension** from the first-person camera on your Viture XR glasses using MediaPipe hand tracking and [PressureVision++](https://github.com/pgrady3/pressurevision2).

The glasses expose a standard UVC webcam. This repo runs the tracker on a host PC (Windows / Linux / macOS) while you wear the glasses.

## What you need

| Item | Notes |
|------|--------|
| Viture XR glasses | Connected over USB so the onboard camera shows up as a webcam |
| Host PC with display | Runs Python + OpenCV window (GPU recommended) |
| Model weights | `weights/paper_29.pth` (PressureVision++ paper checkpoint) |
| Submodules | `external/pressurevision2`, `external/segmentation_models.pytorch` |

## 1. Connect the glasses

1. Power on the Viture glasses and plug them into the PC over USB.
2. Confirm the OS sees an extra webcam (in addition to any built-in laptop camera).
3. Optionally open the OS camera app and switch devices until you see the **egocentric / glasses** view — that is the feed this script will use.

> Tip: On many machines the built-in webcam is index `0` and the Viture camera is index `1` (the script default). If that is wrong, use the camera-index steps below.

## 2. One-time setup

```bash
# Clone (if needed) and enter the repo
git clone <your-repo-url> Viture
cd Viture

# Pull PressureVision++ + segmentation_models.pytorch
git submodule update --init --recursive

# Create / activate a Python 3.10+ environment, then install deps
pip install torch torchvision  # follow https://pytorch.org for CUDA builds
pip install -e external/segmentation_models.pytorch
pip install -r external/pressurevision2/requirements.txt
pip install opencv-python mediapipe numpy pyyaml
```

### Download model weights

1. Download `paper_29.pth` from the [PressureVision2 Dropbox link](https://www.dropbox.com/scl/fi/0r2koefy7bhr66dffc8z7/paper_29.pth?rlkey=wshcxm8iy8l1qo60oo7khdqjp&dl=0).
2. Place it at:

```text
weights/paper_29.pth
```

## 3. Find the Viture camera index

With the glasses plugged in:

```bash
python - <<'PY'
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    print(f"index {i}: {'OPEN' if ok else 'closed'}")
    if ok:
        ok, frame = cap.read()
        print(f"  frame: {None if frame is None else frame.shape}")
    cap.release()
PY
```

Or launch once and flip `--camera-index` until the OpenCV window shows your hands from the glasses POV (not the laptop webcam).

## 4. Launch on the AR glasses camera

```bash
# Default: camera index 1 (typical Viture UVC device)
python viture_tension_tracker.py

# Explicit camera + GPU/CPU
python viture_tension_tracker.py --camera-index 1 --device cuda

# CPU-only machine
python viture_tension_tracker.py --camera-index 1 --device cpu
```

Put the glasses on, look at your hands, and grip something (or press fingertips against a surface). When a hand is detected you should see:

- MediaPipe hand landmarks + a green crop box
- A pressure heatmap overlay on the hand
- A **Muscle/Grip Tension** progress bar and percentage

If no hands are in view, inference is skipped and the window shows **Searching for hands...**

Press **`q`** (or Esc) to quit.

## Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--camera-index` | `1` | OpenCV index for the Viture UVC camera |
| `--weights` | `weights/paper_29.pth` | PressureVision++ checkpoint path |
| `--device` | auto (`cuda` / `mps` / `cpu`) | Torch device |
| `--tension-mode` | `peak` | `peak` or `average` pressure → tension % |
| `--max-force` | `64.0` | Pressure value mapped to 100% |
| `--width` / `--height` | `1280` / `720` | Requested capture resolution |

Example with average tension and a custom max:

```bash
python viture_tension_tracker.py \
  --camera-index 1 \
  --tension-mode average \
  --max-force 32
```

## Expected layout

```text
Viture/
├── viture_tension_tracker.py
├── weights/
│   └── paper_29.pth
└── external/
    ├── pressurevision2/              # submodule
    └── segmentation_models.pytorch/  # submodule
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Wrong camera / laptop selfie view | Try `--camera-index 0`, `2`, … until you see the glasses POV |
| `Unable to open camera index N` | Re-seat USB; close other apps using the camera; try another index |
| `Model weights not found` | Put `paper_29.pth` under `weights/` |
| Missing submodule import errors | Run `git submodule update --init --recursive` |
| Slow / laggy | Use `--device cuda` (or `mps` on Apple Silicon); lower `--width`/`--height` |
| Hands not detected | Improve lighting; keep hands in the lower/center FOV; lower `--min-detection-confidence 0.3` |

## How it works (short)

1. Grab frames from the Viture UVC camera.
2. Detect hands with MediaPipe.
3. Crop each hand with a **20%** padding margin, resize to **448×448**.
4. Run PressureVision++ to get a fingertip pressure heatmap.
5. Convert peak (or average) pressure into a tension percentage and draw the UI bar.
