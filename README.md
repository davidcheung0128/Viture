# Viture Grip Tension Tracker

Estimate **grip / muscle tension** from the first-person camera on your Viture XR glasses using MediaPipe hand tracking and [PressureVision++](https://github.com/pgrady3/pressurevision2).

The glasses expose a standard UVC webcam. Run this tracker on a host Mac/PC while you wear the glasses.

## Quick start

Run **one command per line** in Terminal (do not paste `#` comment lines into zsh):

```bash
cd ~
git clone -b cursor/viture-tension-tracker-4b6d https://github.com/davidcheung0128-ai/Viture.git
cd Viture
git submodule update --init --recursive
ls viture_tension_tracker.py
```

If `ls` says **No such file or directory**, you are on the wrong branch (often `main`, which only has the README). Fix with:

```bash
cd ~/Viture
git fetch origin
git checkout cursor/viture-tension-tracker-4b6d
git pull origin cursor/viture-tension-tracker-4b6d
git submodule update --init --recursive
ls viture_tension_tracker.py
```

Then create the venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e external/segmentation_models.pytorch
python -m pip install -r external/pressurevision2/requirements.txt
python -m pip install opencv-python mediapipe numpy pyyaml
```

Download `paper_29.pth` into `weights/`:

```bash
mkdir -p weights
curl -L -o weights/paper_29.pth "https://www.dropbox.com/scl/fi/0r2koefy7bhr66dffc8z7/paper_29.pth?rlkey=wshcxm8iy8l1qo60oo7khdqjp&dl=1"
ls -lh weights/paper_29.pth
```

Plug in the glasses, then:

```bash
source .venv/bin/activate
cd ~/Viture
python viture_tension_tracker.py --camera-index 1
```

On Apple Silicon Macs, prefer:

```bash
python viture_tension_tracker.py --camera-index 1 --device mps
```

## What you need

| Item | Notes |
|------|--------|
| Viture XR glasses | Connected over USB so the onboard camera shows up as a webcam |
| Host PC with display | Runs Python + OpenCV window (Apple Silicon / NVIDIA GPU recommended) |
| Model weights | `weights/paper_29.pth` |
| Submodules | `external/pressurevision2`, `external/segmentation_models.pytorch` |

## 1. Connect the glasses

1. Power on the Viture glasses and plug them into the PC over USB.
2. Confirm the OS sees an extra webcam (not only the laptop camera).
3. Optionally open the OS camera app and switch devices until you see the **egocentric / glasses** view.

On many machines the built-in webcam is index `0` and the Viture camera is index `1` (the script default).

## 2. Clone this branch

Use the **`.git` clone URL**, not a GitHub `/edit/...` page URL.

Run these **one line at a time** (macOS zsh does not treat `#` as a comment by default, so pasting a whole README block can fail).

```bash
cd ~
git clone -b cursor/viture-tension-tracker-4b6d https://github.com/davidcheung0128-ai/Viture.git
cd Viture
git submodule update --init --recursive
```

If you already cloned the repo and are inside it:

```bash
git fetch origin
git checkout cursor/viture-tension-tracker-4b6d
git submodule update --init --recursive
```

## 3. Create a Python environment and install deps

macOS usually has no `pip` on PATH. Use `python3 -m pip` inside a venv:

```bash
cd ~/Viture
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch (pick the command that matches your machine):

```bash
python -m pip install torch torchvision
```

For NVIDIA CUDA builds, use the install command from https://pytorch.org instead.

Then install the project dependencies:

```bash
python -m pip install -e external/segmentation_models.pytorch
python -m pip install -r external/pressurevision2/requirements.txt
python -m pip install opencv-python mediapipe numpy pyyaml
```

## 4. Download model weights

Create the folder and download the PressureVision++ checkpoint (do **not** type the path alone into the shell — that is not a command):

```bash
cd ~/Viture
mkdir -p weights
curl -L -o weights/paper_29.pth "https://www.dropbox.com/scl/fi/0r2koefy7bhr66dffc8z7/paper_29.pth?rlkey=wshcxm8iy8l1qo60oo7khdqjp&dl=1"
ls -lh weights/paper_29.pth
```

You should see a large file (hundreds of MB). If `curl` is unavailable:

1. Open the [Dropbox download link](https://www.dropbox.com/scl/fi/0r2koefy7bhr66dffc8z7/paper_29.pth?rlkey=wshcxm8iy8l1qo60oo7khdqjp&dl=1) in a browser.
2. Move the downloaded file to `~/Viture/weights/paper_29.pth`.

## 5. Find the Viture camera index

With the glasses plugged in and the venv activated:

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

Use the index that corresponds to the glasses feed.

## 6. Launch

```bash
source .venv/bin/activate
cd ~/Viture

python viture_tension_tracker.py --camera-index 1
```

On Apple Silicon:

```bash
python viture_tension_tracker.py --camera-index 1 --device mps
```

On CPU only:

```bash
python viture_tension_tracker.py --camera-index 1 --device cpu
```

Put the glasses on, look at your hands, and grip something. You should see:

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

## Expected layout

```text
Viture/
├── viture_tension_tracker.py
├── weights/
│   └── paper_29.pth
├── .venv/
└── external/
    ├── pressurevision2/
    └── segmentation_models.pytorch/
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `can't open file '.../viture_tension_tracker.py'` | You are on `main` or an incomplete clone. Run `git checkout cursor/viture-tension-tracker-4b6d` then `git pull` and confirm with `ls viture_tension_tracker.py` |
| `fatal: repository '.../edit/...' not found` | You copied a GitHub **web edit** URL. Use `https://github.com/davidcheung0128-ai/Viture.git` with `-b cursor/viture-tension-tracker-4b6d` |
| `zsh: unknown file attribute: i` / `command not found: #` | Don't paste README comment lines into zsh. Run real commands **one line at a time** |
| `zsh: command not found: pip` | Activate the venv, then use `python -m pip ...` |
| Wrong camera / laptop selfie view | Try `--camera-index 0`, `2`, … until you see the glasses POV |
| `Unable to open camera index N` | Re-seat USB; quit Zoom/FaceTime/other camera apps; try another index |
| `zsh: no such file or directory: weights/paper_29.pth` | That path is not a command. Download the file first with the `curl -L -o weights/paper_29.pth "..."` step above |
| `Model weights not found` | Run the `curl` download into `weights/paper_29.pth`, then `ls -lh weights/paper_29.pth` |
| Missing `external/...` imports | Run `git submodule update --init --recursive` |
| Slow / laggy | Prefer `--device mps` (Mac) or `--device cuda`; lower `--width`/`--height` |
| Hands not detected | Improve lighting; keep hands in view; try `--min-detection-confidence 0.3` |

## How it works (short)

1. Grab frames from the Viture UVC camera.
2. Detect hands with MediaPipe.
3. Crop each hand with a **20%** padding margin, resize to **448×448**.
4. Run PressureVision++ to get a fingertip pressure heatmap.
5. Convert peak (or average) pressure into a tension percentage and draw the UI bar.
