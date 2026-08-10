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
        help="Probe camera indexes, print names/brightness, save preview JPEGs, then exit.",
    )
    parser.add_argument(
        "--auto-camera",
        action="store_true",
        help=(
            "Auto-pick a non-black camera, skipping iPhone/Continuity Camera "
            "and preferring names that look like Viture/UVC."
        ),
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Horizontally flip the camera feed (sometimes needed for egocentric view).",
    )
    parser.add_argument(
        "--show-skeleton",
        action="store_true",
        help="Also draw MediaPipe hand skeleton (off by default; fingertip pressure is shown instead).",
    )
    parser.add_argument(
        "--allow-continuity",
        action="store_true",
        help="Allow Continuity Camera / iPhone as a source (skipped by default).",
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


FINGERTIP_IDS = (4, 8, 12, 16, 20)
FINGERTIP_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
CONTINUITY_NAME_HINTS = (
    "continuity",
    "iphone",
    "ipad",
    "desk view",
    "apple vision",
)
PREFERRED_NAME_HINTS = ("viture", "uvc", "usb", "webcam", "hd camera", "camera")


def macos_avfoundation_camera_names() -> dict[int, str]:
    """Best-effort map of AVFoundation index -> device name via ffmpeg."""
    import re
    import shutil
    import subprocess

    names: dict[int, str] = {}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return names
    try:
        proc = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (proc.stderr or "") + (proc.stdout or "")
    except OSError:
        return names

    in_video = False
    for line in text.splitlines():
        lower = line.lower()
        if "avfoundation video devices" in lower:
            in_video = True
            continue
        if in_video and "avfoundation audio devices" in lower:
            break
        if not in_video:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    return names


def camera_name_is_continuity(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in CONTINUITY_NAME_HINTS)


def camera_name_preference_score(name: str) -> int:
    """Higher is better when auto-selecting (Viture-like names win)."""
    lower = name.lower()
    if camera_name_is_continuity(lower):
        return -100
    score = 0
    if "viture" in lower:
        score += 50
    for hint in PREFERRED_NAME_HINTS:
        if hint in lower:
            score += 5
    if "facetime" in lower or "macbook" in lower:
        score -= 5
    return score


def overlay_heatmap_on_bbox(
    frame: np.ndarray,
    heatmap: np.ndarray,
    bbox: Tuple[int, int, int, int],
    alpha: float = 0.65,
) -> None:
    """Blend PressureVision++ pressure colormap onto the hand crop (fingertip contact)."""
    x1, y1, x2, y2 = bbox
    region_w = x2 - x1
    region_h = y2 - y1
    if region_w <= 0 or region_h <= 0:
        return

    hm = heatmap.astype(np.float32)
    peak = float(hm.max()) if hm.size else 0.0
    if peak <= 1e-6:
        return

    # Log-ish stretch so light fingertip presses stay visible.
    norm = np.clip(hm / peak, 0.0, 1.0)
    norm = np.sqrt(norm)
    color_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(color_u8, cv2.COLORMAP_JET)
    color = cv2.resize(color, (region_w, region_h), interpolation=cv2.INTER_LINEAR)

    roi = frame[y1:y2, x1:x2]
    mask = cv2.resize(
        (hm > (0.02 * peak)).astype(np.uint8) * 255,
        (region_w, region_h),
        interpolation=cv2.INTER_NEAREST,
    )
    blended = cv2.addWeighted(roi, 1.0 - alpha, color, alpha, 0.0)
    roi[mask > 0] = blended[mask > 0]
    frame[y1:y2, x1:x2] = roi


def sample_fingertip_pressures(
    heatmap: np.ndarray,
    points: Sequence[NormPoint],
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    max_force: float,
    radius: int = 4,
) -> List[Tuple[str, Tuple[int, int], float, float]]:
    """
    Sample PressureVision++ heatmap around each fingertip landmark.

    Returns list of (name, pixel_xy, tension_01, raw_force).
    """
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    hh, hw = heatmap.shape[:2]
    out: List[Tuple[str, Tuple[int, int], float, float]] = []

    for name, tip_id in zip(FINGERTIP_NAMES, FINGERTIP_IDS):
        if tip_id >= len(points):
            continue
        nx, ny = points[tip_id]
        px = int(round(nx * frame_w))
        py = int(round(ny * frame_h))

        rx = (nx * frame_w - x1) / bw
        ry = (ny * frame_h - y1) / bh
        hx = int(np.clip(rx * (hw - 1), 0, hw - 1))
        hy = int(np.clip(ry * (hh - 1), 0, hh - 1))

        y0 = max(0, hy - radius)
        y1h = min(hh, hy + radius + 1)
        x0 = max(0, hx - radius)
        x1h = min(hw, hx + radius + 1)
        patch = heatmap[y0:y1h, x0:x1h]
        raw = float(np.max(patch)) if patch.size else 0.0
        tension = float(np.clip(raw / max(max_force, 1e-6), 0.0, 1.0))
        out.append((name, (px, py), tension, raw))
    return out


def draw_fingertip_pressures(
    frame: np.ndarray,
    tip_pressures: Sequence[Tuple[str, Tuple[int, int], float, float]],
) -> None:
    """Draw per-fingertip press strength (how hard each tip is pressing)."""
    for name, (px, py), tension, _raw in tip_pressures:
        # Radius grows with press strength.
        radius = int(8 + 18 * tension)
        if tension < 0.5:
            g = 1.0
            r = tension * 2.0
        else:
            r = 1.0
            g = 1.0 - (tension - 0.5) * 2.0
        color = (0, int(255 * g), int(255 * r))  # BGR green->yellow->red
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 2, cv2.LINE_AA)
        label = f"{name} {int(round(tension * 100))}%"
        cv2.putText(
            frame,
            label,
            (px + radius + 4, py + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (px + radius + 4, py + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_finger_pressure_panel(
    frame: np.ndarray,
    tip_pressures: Sequence[Tuple[str, Tuple[int, int], float, float]],
) -> None:
    """Right-side panel of per-finger press bars."""
    if not tip_pressures:
        return
    h, w = frame.shape[:2]
    panel_w = 220
    x0 = w - panel_w - 16
    y0 = 90
    cv2.rectangle(frame, (x0 - 8, y0 - 36), (w - 8, y0 + 28 * len(tip_pressures) + 8), (20, 20, 20), -1)
    cv2.putText(
        frame,
        "Fingertip press",
        (x0, y0 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for i, (name, _xy, tension, _raw) in enumerate(tip_pressures):
        y = y0 + i * 28
        cv2.putText(
            frame,
            name[:5],
            (x0, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        bar_x = x0 + 70
        bar_w = 110
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 16), (60, 60, 60), -1)
        fill = int(bar_w * tension)
        color = (0, int(255 * (1.0 - tension)), int(255 * tension))
        if fill > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + fill, y + 16), color, -1)
        cv2.putText(
            frame,
            f"{int(round(tension * 100))}%",
            (bar_x + bar_w + 6, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


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


def list_cameras(
    max_index: int = 8,
    preview_dir: Optional[Path] = None,
    allow_continuity: bool = False,
) -> int:
    """
    Probe camera indexes and write preview JPEGs so the user can find the Viture feed.
    """
    preview_dir = preview_dir or (ROOT / "weights" / "camera_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)
    names = macos_avfoundation_camera_names() if sys.platform == "darwin" else {}
    print("Probing cameras (plug in Viture; disable Continuity Camera / iPhone):\n")
    if names:
        print("AVFoundation device names:")
        for idx, name in sorted(names.items()):
            tag = "  [SKIP: iPhone/Continuity]" if camera_name_is_continuity(name) else ""
            print(f"  [{idx}] {name}{tag}")
        print()

    found = 0
    for index in range(max_index + 1):
        name = names.get(index, "")
        continuity = camera_name_is_continuity(name) if name else False
        cap, frame, backend = try_open_camera_index(index)
        if cap is None or frame is None:
            label = f" ({name})" if name else ""
            print(f"  index {index}{label}: closed")
            continue
        brightness = frame_mean_brightness(frame)
        black = is_black_frame(frame)
        preview_path = preview_dir / f"camera_{index}.jpg"
        cv2.imwrite(str(preview_path), frame)
        status = "BLACK/empty?" if black else "OK (has image)"
        if continuity and not allow_continuity:
            status += "  [iPhone/Continuity — skip for Viture]"
        name_bit = f"  name={name!r}" if name else ""
        print(
            f"  index {index}: OPEN  shape={frame.shape}  "
            f"brightness={brightness:.1f}  {status}{name_bit}  preview={preview_path}"
        )
        cap.release()
        found += 1

    print(
        "\nPick the index that is NOT your iPhone and shows the glasses POV.\n"
        "Typical: skip Continuity Camera / iPhone, avoid FaceTime HD (laptop).\n"
        "Then run:\n"
        "  python viture_tension_tracker.py --camera-index N --device mps --project-glasses\n"
        "Or:\n"
        "  python viture_tension_tracker.py --auto-camera --device mps --project-glasses"
    )
    if found == 0:
        print(
            "\nNo cameras opened. Check USB, Camera permission, and quit apps "
            "holding the camera. On Mac: System Settings → Continuity Camera off "
            "if your iPhone keeps stealing the index."
        )
        return 1
    return 0


def find_best_camera(
    max_index: int = 8,
    allow_continuity: bool = False,
) -> Optional[int]:
    """Prefer Viture/UVC-like names; skip Continuity/iPhone unless allowed."""
    names = macos_avfoundation_camera_names() if sys.platform == "darwin" else {}
    candidates: List[Tuple[int, int, float, str]] = []  # score, index, brightness, name

    for index in range(max_index + 1):
        name = names.get(index, f"camera-{index}")
        if camera_name_is_continuity(name) and not allow_continuity:
            print(f"  skipping index {index} ({name}) — Continuity/iPhone")
            continue
        cap, frame, _backend = try_open_camera_index(index)
        if cap is None or frame is None:
            continue
        bright = frame_mean_brightness(frame)
        cap.release()
        if bright < 8.0:
            print(f"  skipping index {index} ({name}) — black frame")
            continue
        score = camera_name_preference_score(name) + int(bright / 25.0)
        candidates.append((score, index, bright, name))
        print(f"  candidate index {index} ({name}) score={score} brightness={bright:.1f}")

    if not candidates:
        return None
    candidates.sort(reverse=True)
    score, index, bright, name = candidates[0]
    print(f"Auto-selected camera index {index} ({name}) score={score} brightness={bright:.1f}")
    return index


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    names = macos_avfoundation_camera_names() if sys.platform == "darwin" else {}
    name = names.get(index, "")
    if name and camera_name_is_continuity(name):
        print(
            f"WARNING: camera index {index} looks like Continuity/iPhone ({name}).\n"
            "  Run --list-cameras and choose the Viture device, or use --auto-camera.\n"
            "  To silence this and force Continuity: pass --allow-continuity.",
            file=sys.stderr,
        )

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

    ok, frame2 = cap.read()
    sample = frame2 if ok and frame2 is not None else frame
    brightness = frame_mean_brightness(sample)
    name_bit = f" name={name!r}" if name else ""
    print(
        f"Opened camera index={index} backend={backend} "
        f"shape={sample.shape} brightness={brightness:.1f}{name_bit}"
    )
    if is_black_frame(sample):
        print(
            "WARNING: camera feed looks black. This is probably NOT the Viture glasses.\n"
            "  Run: python viture_tension_tracker.py --list-cameras\n"
            "  Then: --camera-index N   or   --auto-camera",
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
        return list_cameras(allow_continuity=args.allow_continuity)

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
        auto_idx = find_best_camera(allow_continuity=args.allow_continuity)
        if auto_idx is None:
            print(
                "Auto-camera failed: no suitable non-Continuity camera found. "
                "Run --list-cameras after plugging in the Viture glasses. "
                "Turn off Continuity Camera if your iPhone is selected.",
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
        # Refuse Continuity/iPhone unless explicitly allowed.
        if sys.platform == "darwin" and not args.allow_continuity:
            names = macos_avfoundation_camera_names()
            cname = names.get(camera_index, "")
            if cname and camera_name_is_continuity(cname):
                print(
                    f"Camera index {camera_index} is {cname!r} (iPhone/Continuity), "
                    "not the Viture glasses.\n"
                    "Re-run with --auto-camera, or --list-cameras then --camera-index N.\n"
                    "Pass --allow-continuity only if you really want the iPhone.",
                    file=sys.stderr,
                )
                return 1
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
                    all_tip_pressures: List[
                        Tuple[str, Tuple[int, int], float, float]
                    ] = []
                    for points in hands:
                        if args.show_skeleton:
                            draw_hand_skeleton(display, points)
                        bbox = landmarks_to_bbox(
                            points, frame_w, frame_h, padding=HAND_PADDING
                        )
                        if bbox is None:
                            continue

                        x1, y1, x2, y2 = bbox
                        # Light crop guide only (no skeleton by default).
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 180, 80), 1)

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

                        # PressureVision-style fingertip contact overlay.
                        overlay_heatmap_on_bbox(display, heatmap, bbox, alpha=0.7)
                        tip_pressures = sample_fingertip_pressures(
                            heatmap,
                            points,
                            bbox,
                            frame_w,
                            frame_h,
                            args.max_force,
                        )
                        draw_fingertip_pressures(display, tip_pressures)
                        all_tip_pressures.extend(tip_pressures)

                        tension, _raw = tension_from_heatmap(
                            heatmap, args.max_force, mode=args.tension_mode
                        )
                        tensions.append(tension)

                    if all_tip_pressures:
                        draw_finger_pressure_panel(display, all_tip_pressures)
                        # Overall = hardest fingertip press (more intuitive than area avg).
                        peak_tip = max(t for _n, _xy, t, _r in all_tip_pressures)
                        draw_tension_bar(
                            display,
                            peak_tip,
                            label="Fingertip Press / Grip Tension",
                        )
                    elif tensions:
                        draw_tension_bar(
                            display,
                            max(tensions),
                            label="Fingertip Press / Grip Tension",
                        )
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
