#!/usr/bin/env python3
"""
Viture XR grip-tension tracker.

Captures frames from a Viture glasses UVC camera, tracks hands with MediaPipe,
crops detected hands for PressureVision++, overlays a Muscle/Grip Tension
progress bar, and can fullscreen the result onto the glasses display.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pathing: PressureVision2 + segmentation_models.pytorch (editable clone)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external"
PV2_ROOT = EXTERNAL / "pressurevision2"
SMP_ROOT = EXTERNAL / "segmentation_models.pytorch"
HAND_TASK_PATH = ROOT / "weights" / "hand_landmarker.task"
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

for path in (PV2_ROOT, SMP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# PressureVision2 model architecture / helpers
from prediction.model.fpn_dann_logits_model import FPN_DANN_Logits  # noqa: E402
from prediction.pred_util import (  # noqa: E402
    classes_to_scalar,
    resnet_preprocessor,
)

# Default PressureVision++ paper config values (config/paper.yml)
DEFAULT_FORCE_THRESHOLDS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
NETWORK_IMAGE_SIZE = (448, 448)  # (W, H) for cv2.resize
DEFAULT_WEIGHTS = ROOT / "weights" / "paper_29.pth"
HAND_PADDING = 0.20  # 20% margin around landmark AABB

# MediaPipe hand skeleton edges (landmark index pairs)
HAND_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

NormPoint = Tuple[float, float]


def build_default_config() -> SimpleNamespace:
    """Minimal config namespace matching PressureVision++ paper settings."""
    return SimpleNamespace(
        NETWORK_TYPE="fpn_dann_logits",
        NETWORK_IMAGE_SIZE_X=448,
        NETWORK_IMAGE_SIZE_Y=448,
        NETWORK_INPUT_CHANNELS=3,
        NUM_FORCE_CLASSES=9,
        FORCE_THRESHOLDS=list(DEFAULT_FORCE_THRESHOLDS),
        WEAK_LABEL_HIGH_LOW=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate grip tension from a Viture XR UVC camera and project "
            "the live overlay onto the glasses display."
        )
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index for the Viture glasses (default: 1).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to PressureVision++ checkpoint (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda / cpu / mps (default: auto-detect).",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=64.0,
        help="Pressure value mapped to 100%% tension (default: 64.0).",
    )
    parser.add_argument(
        "--tension-mode",
        choices=("peak", "average"),
        default="peak",
        help="How to reduce the pressure heatmap to a tension score.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe Hands min detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe Hands min tracking confidence.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--project-glasses",
        action="store_true",
        help=(
            "Prepare live projection onto Viture glasses via SpaceWalker / "
            "extended display. Starts windowed so you can drag onto the glasses, "
            "then press 'f' to fullscreen there."
        ),
    )
    parser.add_argument(
        "--window-x",
        type=int,
        default=None,
        help="Optional window X position (use to place on the Viture monitor).",
    )
    parser.add_argument(
        "--window-y",
        type=int,
        default=None,
        help="Optional window Y position (use to place on the Viture monitor).",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indexes, print brightness, save preview JPEGs, then exit.",
    )
    parser.add_argument(
        "--auto-camera",
        action="store_true",
        help="Pick the first camera index that returns a non-black live frame.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Horizontally flip the camera feed (sometimes needed for egocentric view).",
    )
    return parser.parse_args()


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: SimpleNamespace, device: torch.device) -> FPN_DANN_Logits:
    """Instantiate the PressureVision++ FPN_DANN_Logits architecture."""
    num_weak_logits = 7 if config.WEAK_LABEL_HIGH_LOW else 6
    model = FPN_DANN_Logits(
        encoder_name="se_resnext50_32x4d",
        encoder_weights=None,  # weights come from the checkpoint
        classes=config.NUM_FORCE_CLASSES,
        activation=None,
        in_channels=config.NETWORK_INPUT_CHANNELS,
        num_out_logits=num_weak_logits,
    )
    return model.to(device)


def load_pressurevision_model(
    weights_path: Path,
    config: SimpleNamespace,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load paper_29.pth.

    Official demos pickle the full module with torch.save(model). Newer or
    manually exported checkpoints may store a state_dict instead — both are
    supported here.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Download paper_29.pth into weights/ (see PressureVision2 README)."
        )

    print(f"Loading PressureVision++ weights from {weights_path} on {device}...")
    # paper_29.pth is a full pickled nn.Module (official PressureVision++ demo format).
    # PyTorch >= 2.6 defaults weights_only=True, which rejects that pickle.
    try:
        try:
            checkpoint = torch.load(
                str(weights_path), map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(str(weights_path), map_location=device)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ModuleNotFoundError(
            f"Missing dependency {missing!r} while unpickling {weights_path}. "
            "The official PressureVision++ checkpoint needs the legacy encoder "
            "stack. Install with:\n"
            "  python -m pip install pretrainedmodels efficientnet-pytorch timm\n"
            "Or:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.to(device)
    elif isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        cleaned = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
        }
        model = build_model(config, device)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"Warning: missing keys when loading state_dict: {missing[:8]}...")
        if unexpected:
            print(f"Warning: unexpected keys when loading state_dict: {unexpected[:8]}...")
    else:
        raise TypeError(
            f"Unsupported checkpoint type {type(checkpoint)!r} in {weights_path}"
        )

    model.eval()
    return model


def ensure_hand_landmarker_model(task_path: Path = HAND_TASK_PATH) -> Path:
    """
    Ensure MediaPipe HandLandmarker .task model exists.

    Prefers a vendored copy under weights/. If missing, downloads via curl
    (more reliable on macOS Python.org builds than urllib SSL).
    """
    if task_path.is_file() and task_path.stat().st_size > 1_000_000:
        return task_path

    task_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand landmarker model to {task_path}...")

    # 1) curl — uses system certs on macOS; avoids Python SSL issues
    import shutil
    import subprocess

    curl = shutil.which("curl")
    if curl:
        try:
            subprocess.run(
                [curl, "-L", "--fail", "-o", str(task_path), HAND_TASK_URL],
                check=True,
            )
            if task_path.is_file() and task_path.stat().st_size > 1_000_000:
                return task_path
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"curl download failed ({exc}); trying Python urllib...")

    # 2) urllib with certifi CA bundle if available
    try:
        import ssl

        context = None
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            # Last resort for broken macOS Python.org cert installs.
            print(
                "Warning: using unverified SSL context to download hand model "
                "(macOS Python certificate store often broken)."
            )
            context = ssl._create_unverified_context()

        with urllib.request.urlopen(HAND_TASK_URL, context=context) as resp:
            task_path.write_bytes(resp.read())
        if task_path.is_file() and task_path.stat().st_size > 1_000_000:
            return task_path
    except Exception as exc:
        if task_path.exists():
            task_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Failed to download hand_landmarker.task.\n"
            "Run this manually, then rerun the tracker:\n"
            f'  curl -L -o "{task_path}" "{HAND_TASK_URL}"\n'
            f"Original error: {exc}"
        ) from exc

    raise RuntimeError(
        f"Downloaded file looks invalid: {task_path}. "
        f'Re-download with: curl -L -o "{task_path}" "{HAND_TASK_URL}"'
    )


class HandTracker:
    """
    Hands tracker compatible with MediaPipe 1.0 Tasks and legacy solutions API.

    Returns a list of hands; each hand is a list of normalized (x, y) points.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ) -> None:
        self._mode = "tasks"
        self._legacy_hands = None
        self._landmarker = None
        self._frame_ts_ms = 0

        has_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if has_solutions:
            self._mode = "solutions"
            self._legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            print("Using MediaPipe legacy solutions Hands API.")
            return

        # MediaPipe >= 1.0: Tasks HandLandmarker
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        task_path = ensure_hand_landmarker_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._mode = "tasks"
        print("Using MediaPipe Tasks HandLandmarker API (mediapipe 1.x).")

    def process(self, rgb_frame: np.ndarray) -> List[List[NormPoint]]:
        if self._mode == "solutions":
            assert self._legacy_hands is not None
            results = self._legacy_hands.process(rgb_frame)
            hands: List[List[NormPoint]] = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hands.append([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            return hands

        assert self._landmarker is not None
        # Monotonic timestamps required for VIDEO mode.
        self._frame_ts_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        hands = []
        for hand_landmarks in result.hand_landmarks:
            hands.append([(lm.x, lm.y) for lm in hand_landmarks])
        return hands

    def close(self) -> None:
        if self._legacy_hands is not None:
            self._legacy_hands.close()
        if self._landmarker is not None:
            self._landmarker.close()


def landmarks_to_bbox(
    points: Sequence[NormPoint],
    frame_w: int,
    frame_h: int,
    padding: float = HAND_PADDING,
) -> Optional[Tuple[int, int, int, int]]:
    """Convert normalized hand landmarks to a padded pixel AABB (x1,y1,x2,y2)."""
    if not points:
        return None
    xs = [p[0] * frame_w for p in points]
    ys = [p[1] * frame_h for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw = x_max - x_min
    bh = y_max - y_min

    pad_x = bw * padding
    pad_y = bh * padding

    x1 = int(max(0, np.floor(x_min - pad_x)))
    y1 = int(max(0, np.floor(y_min - pad_y)))
    x2 = int(min(frame_w, np.ceil(x_max + pad_x)))
    y2 = int(min(frame_h, np.ceil(y_max + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_hand_skeleton(
    frame: np.ndarray,
    points: Sequence[NormPoint],
) -> None:
    """Draw simple hand landmarks/connections without mediapipe.solutions."""
    h, w = frame.shape[:2]
    pix = [(int(x * w), int(y * h)) for x, y in points]
    for a, b in HAND_CONNECTIONS:
        if a < len(pix) and b < len(pix):
            cv2.line(frame, pix[a], pix[b], (0, 255, 0), 2, cv2.LINE_AA)
    for p in pix:
        cv2.circle(frame, p, 3, (0, 200, 255), -1, cv2.LINE_AA)


def preprocess_hand_crop(bgr_crop: np.ndarray) -> torch.Tensor:
    """BGR crop -> NCHW float tensor with ImageNet / ResNet normalization."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype("float32") / 255.0
    rgb = resnet_preprocessor(rgb)
    chw = rgb.transpose(2, 0, 1).astype("float32")
    return torch.from_numpy(chw).unsqueeze(0)


@torch.inference_mode()
def run_pressure_inference(
    model: torch.nn.Module,
    bgr_crop: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> np.ndarray:
    """
    Run PressureVision++ on a hand crop.

    Returns a HxW float pressure heatmap in the crop's resized space.
    """
    resized = cv2.resize(
        bgr_crop,
        (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y),
        interpolation=cv2.INTER_LINEAR,
    )
    batch = preprocess_hand_crop(resized).to(device)
    outputs = model(batch)
    force_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    force_class = torch.argmax(force_logits, dim=1)
    force_scalar = classes_to_scalar(force_class, config.FORCE_THRESHOLDS)
    return force_scalar.detach().cpu().squeeze().numpy()


def tension_from_heatmap(
    heatmap: np.ndarray,
    max_force: float,
    mode: str = "peak",
) -> Tuple[float, float]:
    """Reduce a pressure heatmap to (tension_01, raw_force)."""
    if heatmap.size == 0:
        return 0.0, 0.0

    contact = heatmap[heatmap > 0]
    if contact.size == 0:
        raw = float(np.max(heatmap))
    elif mode == "average":
        raw = float(np.mean(contact))
    else:
        raw = float(np.max(heatmap))

    tension = float(np.clip(raw / max(max_force, 1e-6), 0.0, 1.0))
    return tension, raw


def draw_tension_bar(
    frame: np.ndarray,
    tension: float,
    label: str = "Muscle/Grip Tension",
) -> None:
    """Draw a horizontal progress bar + percentage onto the OpenCV frame."""
    h, w = frame.shape[:2]
    bar_w = min(420, max(220, w // 3))
    bar_h = 28
    margin = 24
    x1 = margin
    y1 = h - margin - bar_h - 28
    x2 = x1 + bar_w
    y2 = y1 + bar_h

    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)

    fill_w = int(bar_w * np.clip(tension, 0.0, 1.0))
    if fill_w > 0:
        t = float(np.clip(tension, 0.0, 1.0))
        if t < 0.5:
            g = 1.0
            r = t * 2.0
        else:
            r = 1.0
            g = 1.0 - (t - 0.5) * 2.0
        color = (0, int(255 * g), int(255 * r))  # BGR
        cv2.rectangle(frame, (x1, y1), (x1 + fill_w, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)

    pct = int(round(tension * 100))
    cv2.putText(
        frame,
        f"{pct}%",
        (x2 + 12, y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_searching_overlay(frame: np.ndarray) -> None:
    text = "Searching for hands..."
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (0, 200, 255), thickness, cv2.LINE_AA)


def overlay_heatmap_on_bbox(
    frame: np.ndarray,
    heatmap: np.ndarray,
    bbox: Tuple[int, int, int, int],
    alpha: float = 0.45,
) -> None:
    """Optional contact visualization inside the hand crop."""
    x1, y1, x2, y2 = bbox
    region_w = x2 - x1
    region_h = y2 - y1
    if region_w <= 0 or region_h <= 0:
        return

    hm = heatmap.astype(np.float32)
    peak = float(hm.max()) if hm.size else 0.0
    if peak <= 1e-6:
        return
    norm = np.clip(hm / peak, 0.0, 1.0)
    color_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(color_u8, cv2.COLORMAP_JET)
    color = cv2.resize(color, (region_w, region_h), interpolation=cv2.INTER_LINEAR)

    roi = frame[y1:y2, x1:x2]
    mask = cv2.resize(
        (hm > 0).astype(np.uint8) * 255,
        (region_w, region_h),
        interpolation=cv2.INTER_NEAREST,
    )
    blended = cv2.addWeighted(roi, 1.0 - alpha, color, alpha, 0.0)
    roi[mask > 0] = blended[mask > 0]
    frame[y1:y2, x1:x2] = roi


def frame_mean_brightness(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    return float(np.mean(frame))


def is_black_frame(frame: np.ndarray, threshold: float = 8.0) -> bool:
    """True when the frame is essentially black / no real camera image."""
    return frame_mean_brightness(frame) < threshold


def preferred_camera_backends() -> List[int]:
    backends: List[int] = []
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    if hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    backends.append(cv2.CAP_ANY)
    return backends


def try_open_camera_index(index: int) -> Tuple[Optional[cv2.VideoCapture], Optional[np.ndarray], Optional[int]]:
    """Try backends for one index; return (cap, first_frame, backend) or (None, None, None)."""
    for backend in preferred_camera_backends():
        candidate = cv2.VideoCapture(index, backend)
        if not candidate.isOpened():
            candidate.release()
            continue
        ok, frame = candidate.read()
        if ok and frame is not None:
            return candidate, frame, backend
        candidate.release()
    return None, None, None


def list_cameras(max_index: int = 8, preview_dir: Optional[Path] = None) -> int:
    """
    Probe camera indexes and write preview JPEGs so the user can find the Viture feed.
    """
    preview_dir = preview_dir or (ROOT / "weights" / "camera_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    print("Probing cameras (plug in Viture first; grant Terminal Camera permission):\n")
    found = 0
    for index in range(max_index + 1):
        cap, frame, backend = try_open_camera_index(index)
        if cap is None or frame is None:
            print(f"  index {index}: closed")
            continue
        brightness = frame_mean_brightness(frame)
        black = is_black_frame(frame)
        preview_path = preview_dir / f"camera_{index}.jpg"
        cv2.imwrite(str(preview_path), frame)
        status = "BLACK/empty?" if black else "OK (has image)"
        print(
            f"  index {index}: OPEN  shape={frame.shape}  "
            f"brightness={brightness:.1f}  {status}  preview={preview_path}"
        )
        cap.release()
        found += 1

    print(
        "\nOpen the preview JPEGs and pick the index that shows the glasses POV "
        "(your hands from your eyes), then run:\n"
        "  python viture_tension_tracker.py --camera-index N --device mps --project-glasses"
    )
    if found == 0:
        print(
            "\nNo cameras opened. Check: USB cable, Camera permission for Terminal, "
            "and quit Zoom/FaceTime/SpaceWalker camera preview if they hold the device."
        )
        return 1
    return 0


def find_first_non_black_camera(max_index: int = 8) -> Optional[int]:
    for index in range(max_index + 1):
        cap, frame, _backend = try_open_camera_index(index)
        if cap is None or frame is None:
            continue
        bright = frame_mean_brightness(frame)
        cap.release()
        if bright >= 8.0:
            print(f"Auto-selected camera index {index} (brightness={bright:.1f})")
            return index
    return None


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap, frame, backend = try_open_camera_index(index)
    if cap is None or frame is None:
        raise RuntimeError(
            f"Unable to open camera index {index}. "
            "Grant Camera permission to Terminal, plug in Viture, then run "
            "`python viture_tension_tracker.py --list-cameras`."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if hasattr(cv2, "VideoWriter_fourcc"):
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Re-read after format negotiation.
    ok, frame2 = cap.read()
    sample = frame2 if ok and frame2 is not None else frame
    brightness = frame_mean_brightness(sample)
    print(
        f"Opened camera index={index} backend={backend} "
        f"shape={sample.shape} brightness={brightness:.1f}"
    )
    if is_black_frame(sample):
        print(
            "WARNING: camera feed looks black. This is probably NOT the Viture glasses.\n"
            "  1) Wear/uncover the glasses camera\n"
            "  2) Run: python viture_tension_tracker.py --list-cameras\n"
            "  3) Rerun with the index whose preview shows your egocentric view\n"
            "  4) Or try: --auto-camera",
            file=sys.stderr,
        )
    return cap


def print_glasses_projection_help() -> None:
    print(
        "\n=== Project onto Viture glasses ===\n"
        "The tracker window opens on your Mac. Viture does NOT auto-mirror it.\n"
        "Do this:\n"
        "  1. Open Viture SpaceWalker (or enable the glasses as a display)\n"
        "  2. Drag the 'Viture Grip Tension Tracker' window into the glasses view\n"
        "     (SpaceWalker: pin/capture that window into XR)\n"
        "  3. With that window focused in the glasses, press 'f' to fullscreen\n"
        "  4. Press 'q' to quit\n"
    )


def setup_projection_window(
    window: str,
    project_glasses: bool,
    window_x: Optional[int],
    window_y: Optional[int],
) -> bool:
    """
    Create the OpenCV window.

    Important: do NOT immediately fullscreen on the laptop. Fullscreen on the
    primary display hides the window from SpaceWalker. Start windowed, let the
    user drag it into the glasses, then press 'f'.
    """
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    if window_x is not None and window_y is not None:
        cv2.moveWindow(window, window_x, window_y)
        print(f"Moved window to ({window_x}, {window_y})")
    if project_glasses:
        print_glasses_projection_help()
    return False


def draw_camera_status(
    frame: np.ndarray,
    camera_index: int,
    brightness: float,
    black: bool,
) -> None:
    msg = f"Camera #{camera_index}  brightness={brightness:.0f}"
    color = (0, 0, 255) if black else (0, 255, 0)
    cv2.putText(
        frame,
        msg,
        (16, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    if black:
        lines = [
            "BLACK FEED - not using glasses camera",
            "Run: python viture_tension_tracker.py --list-cameras",
            "Then: --camera-index N   or   --auto-camera",
        ]
        y = frame.shape[0] // 2 + 40
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = (frame.shape[1] - tw) // 2
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            y += th + 12


def main() -> int:
    args = parse_args()

    if args.list_cameras:
        return list_cameras()

    config = build_default_config()
    device = resolve_device(args.device)

    missing_roots = [p for p in (PV2_ROOT, SMP_ROOT) if not p.is_dir()]
    if missing_roots:
        print(
            "Warning: expected submodule paths are missing:\n  "
            + "\n  ".join(str(p) for p in missing_roots)
            + "\nInitialize with:\n"
            "  git submodule update --init --recursive\n"
            "or clone pressurevision2 / segmentation_models.pytorch into external/.",
            file=sys.stderr,
        )

    camera_index = args.camera_index
    if args.auto_camera:
        auto_idx = find_first_non_black_camera()
        if auto_idx is None:
            print(
                "Auto-camera failed: every probed index was closed or black. "
                "Run --list-cameras after plugging in the Viture glasses.",
                file=sys.stderr,
            )
            return 1
        camera_index = auto_idx

    try:
        model = load_pressurevision_model(args.weights, config, device)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        cap = open_camera(camera_index, args.width, args.height)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    tracker = HandTracker(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    window = "Viture Grip Tension Tracker"
    fullscreen = setup_projection_window(
        window,
        project_glasses=args.project_glasses,
        window_x=args.window_x,
        window_y=args.window_y,
    )

    fps_ema = 0.0
    last_t = time.perf_counter()

    print(
        f"Streaming camera {camera_index} | device={device} | "
        f"tension_mode={args.tension_mode}."
    )
    print("Keys: q=quit | f=toggle fullscreen AFTER dragging window into glasses")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            brightness = frame_mean_brightness(frame)
            black = is_black_frame(frame)

            display = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            # Skip expensive inference on black/invalid camera feeds.
            if black:
                draw_searching_overlay(display)
                draw_camera_status(display, camera_index, brightness, black=True)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = tracker.process(rgb)
                tensions: List[float] = []

                if hands:
                    for points in hands:
                        draw_hand_skeleton(display, points)
                        bbox = landmarks_to_bbox(
                            points, frame_w, frame_h, padding=HAND_PADDING
                        )
                        if bbox is None:
                            continue

                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 128), 2)

                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        try:
                            heatmap = run_pressure_inference(
                                model, crop, config, device
                            )
                        except Exception as exc:  # keep UI alive on a bad frame
                            print(f"Inference error: {exc}", file=sys.stderr)
                            continue

                        tension, _raw = tension_from_heatmap(
                            heatmap, args.max_force, mode=args.tension_mode
                        )
                        tensions.append(tension)
                        overlay_heatmap_on_bbox(display, heatmap, bbox)

                    if tensions:
                        draw_tension_bar(display, max(tensions))
                    else:
                        draw_searching_overlay(display)
                else:
                    draw_searching_overlay(display)

                draw_camera_status(display, camera_index, brightness, black=False)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = inst if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * inst)
            cv2.putText(
                display,
                f"FPS: {fps_ema:4.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if args.project_glasses:
                hint = (
                    "FULLSCREEN IN GLASSES"
                    if fullscreen
                    else "Drag window into SpaceWalker, then press f"
                )
                cv2.putText(
                    display,
                    hint,
                    (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, prop)
                print(f"Fullscreen projection: {'ON' if fullscreen else 'OFF'}")
                if fullscreen:
                    print(
                        "If fullscreen took over the laptop, press f again, "
                        "drag the window into SpaceWalker/glasses, then press f."
                    )
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
