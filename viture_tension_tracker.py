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
HAND_PADDING = 0.20  # kept for CLI compatibility; inference uses square PV2-style crop
HAND_CROP_SCALE = 1.5  # PressureVision++ paper crop scale around hand center
FINGERTIP_IDS = (4, 8, 12, 16, 20)
FINGERTIP_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
# Distinct BGR colors so each tip is obvious even at 0% force
FINGERTIP_COLORS = (
    (255, 100, 50),   # Thumb
    (0, 165, 255),    # Index
    (0, 255, 255),    # Middle
    (0, 255, 0),      # Ring
    (255, 0, 255),    # Pinky
)

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
        default=0,
        help="OpenCV camera index for the Viture glasses (default: 0).",
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
        default=16.0,
        help="Pressure value mapped to 100%% fingertip force UI (default: 16.0).",
    )
    parser.add_argument(
        "--fpv-adaptive",
        action="store_true",
        default=True,
        help="Adaptive egocentric scaling from recent peak tip forces (default on).",
    )
    parser.add_argument(
        "--no-fpv-adaptive",
        action="store_false",
        dest="fpv_adaptive",
        help="Disable adaptive FPV scaling; use fixed --max-force only.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="EMA smoothing for per-finger force in [0,1) (default: 0.5).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=2.0,
        help="Multiply contact-probability map before tip assignment (default: 2.0).",
    )
    parser.add_argument(
        "--hard-argmax",
        action="store_true",
        help="Use hard class argmax (often stuck at 0 on FPV). Default is soft expected force.",
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

    Returns a list of (hand_label, points) where hand_label is "Left" or "Right".
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

    @staticmethod
    def _normalize_hand_label(label: str) -> str:
        lower = (label or "").strip().lower()
        if "left" in lower:
            return "Left"
        if "right" in lower:
            return "Right"
        return "Hand"

    def process(
        self, rgb_frame: np.ndarray
    ) -> List[Tuple[str, List[NormPoint]]]:
        if self._mode == "solutions":
            assert self._legacy_hands is not None
            results = self._legacy_hands.process(rgb_frame)
            hands: List[Tuple[str, List[NormPoint]]] = []
            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = "Hand"
                    if i < len(handedness) and handedness[i].classification:
                        label = self._normalize_hand_label(
                            handedness[i].classification[0].label
                        )
                    # MediaPipe solutions assumes mirrored selfie view; for a
                    # normal webcam/desk view the Left/Right labels are swapped.
                    if label == "Left":
                        label = "Right"
                    elif label == "Right":
                        label = "Left"
                    points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    hands.append((label, points))
            return self._dedupe_hand_labels(hands)

        assert self._landmarker is not None
        self._frame_ts_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        hands = []
        handedness_list = result.handedness or []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            label = "Hand"
            if i < len(handedness_list) and handedness_list[i]:
                # Tasks API: categories[0].category_name is "Left"/"Right"
                cat = handedness_list[i][0]
                name = getattr(cat, "category_name", None) or getattr(
                    cat, "display_name", ""
                )
                label = self._normalize_hand_label(str(name))
            points = [(lm.x, lm.y) for lm in hand_landmarks]
            hands.append((label, points))
        return self._dedupe_hand_labels(hands)

    @staticmethod
    def _dedupe_hand_labels(
        hands: List[Tuple[str, List[NormPoint]]],
    ) -> List[Tuple[str, List[NormPoint]]]:
        """Ensure unique Left/Right labels; fall back to wrist x-order if needed."""
        if len(hands) <= 1:
            return hands
        labels = [h[0] for h in hands]
        if labels.count("Left") <= 1 and labels.count("Right") <= 1 and "Hand" not in labels:
            return hands
        # Sort by wrist x (landmark 0): leftmost in image -> Left
        ordered = sorted(
            hands,
            key=lambda hp: hp[1][0][0] if hp[1] else 0.5,
        )
        out: List[Tuple[str, List[NormPoint]]] = []
        if len(ordered) == 1:
            return [("Right", ordered[0][1])]
        out.append(("Left", ordered[0][1]))
        out.append(("Right", ordered[1][1]))
        for extra in ordered[2:]:
            out.append((f"Hand{len(out)+1}", extra[1]))
        return out

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
    scale: float = HAND_CROP_SCALE,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Square hand crop matching PressureVision++ (center + radius * scale).

    `padding` is accepted for compatibility; square scale controls the crop.
    """
    if not points:
        return None
    xs = [p[0] * frame_w for p in points]
    ys = [p[1] * frame_h for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    radius = max(x_max - cx, y_max - cy, 1.0) * scale
    # Keep a little extra room from the legacy 20% pad request.
    radius *= 1.0 + max(0.0, padding) * 0.25

    x1 = int(max(0, np.floor(cx - radius)))
    y1 = int(max(0, np.floor(cy - radius)))
    x2 = int(min(frame_w, np.ceil(cx + radius)))
    y2 = int(min(frame_h, np.ceil(cy + radius)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def fingertip_pixels(
    points: Sequence[NormPoint],
    frame_w: int,
    frame_h: int,
) -> List[Tuple[str, Tuple[int, int], int]]:
    """Return (name, pixel_xy, tip_landmark_id) for each fingertip."""
    out: List[Tuple[str, Tuple[int, int], int]] = []
    for name, tip_id in zip(FINGERTIP_NAMES, FINGERTIP_IDS):
        if tip_id >= len(points):
            continue
        nx, ny = points[tip_id]
        px = int(round(nx * frame_w))
        py = int(round(ny * frame_h))
        out.append((name, (px, py), tip_id))
    return out


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
    soft: bool = True,
    temperature: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run PressureVision++ on a hand crop.

    Returns (force_heatmap, contact_prob_map, stats).

    On out-of-domain RGB, hard argmax is almost always class 0. We decode a
    temperature-scaled contact-probability map and use that as the primary
    per-pixel press signal for fingertip readout.
    """
    resized = cv2.resize(
        bgr_crop,
        (config.NETWORK_IMAGE_SIZE_X, config.NETWORK_IMAGE_SIZE_Y),
        interpolation=cv2.INTER_LINEAR,
    )
    batch = preprocess_hand_crop(resized).to(device=device, dtype=torch.float32)
    outputs = model(batch)
    force_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    force_logits = force_logits.float()

    thresholds = list(config.FORCE_THRESHOLDS)
    class_values = []
    for idx, threshold in enumerate(thresholds):
        if idx == 0:
            class_values.append(float(thresholds[0]))
        elif idx == len(thresholds) - 1:
            class_values.append(
                float(thresholds[-1] + (thresholds[-1] - thresholds[-2]) / 2.0)
            )
        else:
            class_values.append(float((thresholds[idx] + thresholds[idx + 1]) / 2.0))
    value_t = torch.tensor(class_values, device=force_logits.device, dtype=torch.float32)
    value_t = value_t.view(1, -1, 1, 1)

    logit_span = float((force_logits.max() - force_logits.min()).item())

    if soft:
        temp = max(float(temperature), 1e-3)
        probs = torch.softmax(force_logits / temp, dim=1)
        p_contact = (1.0 - probs[:, 0]).clamp(0.0, 1.0)
        expected = (probs * value_t).sum(dim=1)
        contact_map = p_contact
        force_map = expected * (0.25 + 0.75 * p_contact)
        p_contact_mean = float(p_contact.mean().item())
        p_contact_max = float(p_contact.max().item())
    else:
        force_class = torch.argmax(force_logits, dim=1)
        force_map = classes_to_scalar(force_class, thresholds)
        contact_map = (force_class > 0).float()
        p_contact_mean = float(contact_map.mean().item())
        p_contact_max = float(contact_map.max().item())

    force_heatmap = force_map.detach().cpu().squeeze().numpy().astype(np.float32)
    contact_heatmap = contact_map.detach().cpu().squeeze().numpy().astype(np.float32)
    force_heatmap = np.nan_to_num(force_heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    contact_heatmap = np.nan_to_num(contact_heatmap, nan=0.0, posinf=0.0, neginf=0.0)

    stats = {
        "peak": float(force_heatmap.max()) if force_heatmap.size else 0.0,
        "contact_peak": float(contact_heatmap.max()) if contact_heatmap.size else 0.0,
        "mean": float(force_heatmap.mean()) if force_heatmap.size else 0.0,
        "nonzero": int(np.count_nonzero(contact_heatmap > 0.02)),
        "p_contact_mean": p_contact_mean,
        "p_contact_max": p_contact_max,
        "logit_span": logit_span,
        "soft": soft,
    }
    return force_heatmap, contact_heatmap, stats


def _angle_deg(a: NormPoint, b: NormPoint, c: NormPoint) -> float:
    """Angle ABC in degrees at point b."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return 180.0
    cos = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def pose_finger_press_proxy(
    points: Sequence[NormPoint],
    frame_w: int,
    frame_h: int,
) -> List[Tuple[str, Tuple[int, int], float, float]]:
    """
    Pose-based per-finger effort proxy when PV2 contact is dead.

    Uses finger curl (PIP flexion). NOT true contact Newtons — responds to
    gripping/pressing postures so the UI is not stuck at 0% without Tekscan data.
    """
    chains = (
        ("Thumb", 1, 2, 3, 4),
        ("Index", 5, 6, 7, 8),
        ("Middle", 9, 10, 11, 12),
        ("Ring", 13, 14, 15, 16),
        ("Pinky", 17, 18, 19, 20),
    )
    out: List[Tuple[str, Tuple[int, int], float, float]] = []
    if len(points) < 21:
        return out
    for name, mcp, pip, dip, tip in chains:
        ang = _angle_deg(points[mcp], points[pip], points[dip])
        curl = float(np.clip((165.0 - ang) / 100.0, 0.0, 1.0))
        tip_y = points[tip][1]
        wrist_y = points[0][1]
        planted = float(np.clip((tip_y - wrist_y) * 2.5 + 0.25, 0.0, 0.35))
        tension = float(np.clip(0.75 * curl + 0.25 * planted, 0.0, 1.0))
        px = int(round(points[tip][0] * frame_w))
        py = int(round(points[tip][1] * frame_h))
        out.append((name, (px, py), tension, tension))
    return out



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
) -> List[Tuple[str, Tuple[int, int], float, float]]:
    """
    Estimate per-fingertip force from a PressureVision++ heatmap.

    Uses distance-weighted soft assignment of contact pixels to the five
    fingertips (better than a tiny tip patch alone), which is the best we can
    do from FPV RGB without a custom Tekscan dataset.
    """
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    hh, hw = heatmap.shape[:2]
    hm = heatmap.astype(np.float32)

    tip_px: List[Tuple[str, Tuple[int, int], Tuple[float, float]]] = []
    palm_x = points[0][0] * frame_w if points else 0.0
    palm_y = points[0][1] * frame_h if points else 0.0

    for name, tip_id in zip(FINGERTIP_NAMES, FINGERTIP_IDS):
        if tip_id >= len(points):
            continue
        nx, ny = points[tip_id]
        px = int(round(nx * frame_w))
        py = int(round(ny * frame_h))
        # Finger pad: slightly toward palm from the tip landmark.
        pad_x = px + 0.18 * (palm_x - px)
        pad_y = py + 0.18 * (palm_y - py)
        # Heatmap coordinates of the pad center.
        hx = ((pad_x - x1) / bw) * (hw - 1)
        hy = ((pad_y - y1) / bh) * (hh - 1)
        tip_px.append((name, (px, py), (hx, hy)))

    if not tip_px:
        return []

    # Soft-assign contact mass to nearest fingertips.
    peak = float(hm.max()) if hm.size else 0.0
    thr = max(peak * 0.01, 1e-6) if peak > 0 else 1e9
    raw_forces = np.zeros(len(tip_px), dtype=np.float64)
    # Influence radius in heatmap pixels (~12% of crop).
    sigma = max(10.0, 0.12 * min(hh, hw))
    ys, xs = np.where(hm >= thr)
    if ys.size:
        for y, x in zip(ys.tolist(), xs.tolist()):
            val = float(hm[y, x])
            dists = []
            for _name, _pix, (hx, hy) in tip_px:
                d2 = (x - hx) ** 2 + (y - hy) ** 2
                dists.append(d2)
            d2 = np.asarray(dists, dtype=np.float64)
            w = np.exp(-d2 / (2.0 * sigma * sigma))
            w_sum = float(w.sum())
            if w_sum <= 1e-8:
                continue
            w /= w_sum
            raw_forces += w * val

    # Also take a local max near each tip as a floor (catches small contacts).
    radius = max(12, int(0.07 * min(hh, hw)))
    out: List[Tuple[str, Tuple[int, int], float, float]] = []
    for i, (name, pix, (hx, hy)) in enumerate(tip_px):
        cx, cy = int(round(hx)), int(round(hy))
        y0 = max(0, cy - radius)
        y1h = min(hh, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1h = min(hw, cx + radius + 1)
        patch = hm[y0:y1h, x0:x1h]
        local = float(np.max(patch)) if patch.size else 0.0
        raw = float(max(raw_forces[i], local))
        tension = float(np.clip(raw / max(max_force, 1e-6), 0.0, 1.0))
        out.append((name, pix, tension, raw))
    return out


class FingerForceSmoother:
    """EMA + adaptive FPV scale so relative tip forces are readable egocentrically."""

    def __init__(self, smooth: float = 0.65, adaptive: bool = True, max_force: float = 16.0):
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.adaptive = adaptive
        self.max_force = max_force
        self._ema_raw: dict[str, float] = {n: 0.0 for n in FINGERTIP_NAMES}
        self._peak = max(max_force * 0.08, 0.5)

    def update(
        self,
        tip_pressures: Sequence[Tuple[str, Tuple[int, int], float, float]],
    ) -> List[Tuple[str, Tuple[int, int], float, float]]:
        # Update EMA on raw forces.
        seen = set()
        for name, pix, _t, raw in tip_pressures:
            seen.add(name)
            prev = self._ema_raw.get(name, 0.0)
            self._ema_raw[name] = self.smooth * prev + (1.0 - self.smooth) * float(raw)

        for name in FINGERTIP_NAMES:
            if name not in seen:
                self._ema_raw[name] = self.smooth * self._ema_raw.get(name, 0.0)

        # Adaptive ceiling from recent peaks (FPV absolute scale is unreliable).
        cur_peak = max(self._ema_raw.values()) if self._ema_raw else 0.0
        if self.adaptive:
            self._peak = max(self._peak * 0.998, cur_peak, self.max_force * 0.05)
            scale = max(self._peak, 1e-3)
        else:
            scale = max(self.max_force, 1e-3)

        out: List[Tuple[str, Tuple[int, int], float, float]] = []
        pix_by_name = {n: (0, 0) for n in FINGERTIP_NAMES}
        for name, pix, _t, _r in tip_pressures:
            pix_by_name[name] = pix
        for name in FINGERTIP_NAMES:
            raw = self._ema_raw.get(name, 0.0)
            tension = float(np.clip(raw / scale, 0.0, 1.0))
            out.append((name, pix_by_name.get(name, (0, 0)), tension, raw))
        return out


def draw_fingertip_pressures(
    frame: np.ndarray,
    tip_pressures: Sequence[Tuple[str, Tuple[int, int], float, float]],
) -> None:
    """Draw always-visible fingertip markers + per-tip force %."""
    for i, (name, (px, py), tension, raw) in enumerate(tip_pressures):
        base = FINGERTIP_COLORS[i % len(FINGERTIP_COLORS)]
        # Mix toward red as force rises.
        color = (
            int(base[0] * (1.0 - tension) + 0 * tension),
            int(base[1] * (1.0 - tension) + 0 * tension),
            int(base[2] * (1.0 - tension) + 255 * tension),
        )
        radius = int(12 + 22 * tension)
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3, (0, 0, 0), -1, cv2.LINE_AA)

        label = f"{name} {int(round(tension * 100))}% ({raw:.1f})"
        tx, ty = px - 40, py - radius - 10
        tx = int(np.clip(tx, 4, frame.shape[1] - 180))
        ty = int(np.clip(ty, 18, frame.shape[0] - 4))
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_finger_pressure_panel(
    frame: np.ndarray,
    tip_pressures: Sequence[Tuple[str, Tuple[int, int], float, float]],
    title: str = "Fingertip press",
    side: str = "right",
) -> None:
    """Per-hand fingertip press bars. side: 'right' or 'left' of the frame."""
    if not tip_pressures:
        return
    h, w = frame.shape[:2]
    panel_w = 210
    margin = 16
    if side == "left":
        x0 = margin + 8
    else:
        x0 = w - panel_w - margin
    y0 = 90
    box_h = 28 * len(tip_pressures) + 44
    cv2.rectangle(
        frame,
        (x0 - 8, y0 - 36),
        (x0 + panel_w, y0 - 36 + box_h),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        title,
        (x0, y0 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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
        bar_x = x0 + 60
        bar_w = 100
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 16), (60, 60, 60), -1)
        fill = int(bar_w * np.clip(tension, 0.0, 1.0))
        color = (0, int(255 * (1.0 - tension)), int(255 * tension))
        if fill > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + fill, y + 16), color, -1)
        cv2.putText(
            frame,
            f"{int(round(tension * 100))}%",
            (bar_x + bar_w + 4, y + 13),
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
    # Separate smoothers so Left and Right hands don't overwrite each other.
    finger_smoothers: dict[str, FingerForceSmoother] = {
        "Left": FingerForceSmoother(
            smooth=args.smooth, adaptive=args.fpv_adaptive, max_force=args.max_force
        ),
        "Right": FingerForceSmoother(
            smooth=args.smooth, adaptive=args.fpv_adaptive, max_force=args.max_force
        ),
    }

    def smoother_for(label: str) -> FingerForceSmoother:
        if label not in finger_smoothers:
            finger_smoothers[label] = FingerForceSmoother(
                smooth=args.smooth,
                adaptive=args.fpv_adaptive,
                max_force=args.max_force,
            )
        return finger_smoothers[label]

    print(
        f"Streaming camera {camera_index} | device={device} | "
        f"gain={args.gain} | fpv_adaptive={args.fpv_adaptive}."
    )
    print(
        "Per-finger readout: PressureVision++ contact prob + pose-curl fallback. "
        "If on-screen src=pose-proxy, pretrained PV2 sees no contact — bars still "
        "move from finger curl (grip). True press Newtons need Tekscan finetune."
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
                    tips_by_hand: dict[str, List[Tuple[str, Tuple[int, int], float, float]]] = {}
                    last_source = "—"
                    last_stats = None
                    for hand_label, points in hands:
                        if args.show_skeleton:
                            draw_hand_skeleton(display, points)

                        tips = fingertip_pixels(points, frame_w, frame_h)
                        # Label which hand near the wrist.
                        if points:
                            wx = int(round(points[0][0] * frame_w))
                            wy = int(round(points[0][1] * frame_h))
                            cv2.putText(
                                display,
                                hand_label,
                                (wx - 20, max(20, wy - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        for i, (name, (px, py), _tid) in enumerate(tips):
                            color = FINGERTIP_COLORS[i % len(FINGERTIP_COLORS)]
                            cv2.circle(display, (px, py), 10, color, 2, cv2.LINE_AA)

                        bbox = landmarks_to_bbox(
                            points, frame_w, frame_h, padding=HAND_PADDING
                        )
                        sm = smoother_for(hand_label)
                        if bbox is None:
                            zero_tips = [
                                (name, xy, 0.0, 0.0) for name, xy, _tid in tips
                            ]
                            zero_tips = sm.update(zero_tips)
                            draw_fingertip_pressures(display, zero_tips)
                            tips_by_hand[hand_label] = zero_tips
                            continue

                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 180, 80), 1)

                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        try:
                            force_hm, contact_hm, heat_stats = run_pressure_inference(
                                model,
                                crop,
                                config,
                                device,
                                soft=not args.hard_argmax,
                                temperature=3.0,
                            )
                        except Exception as exc:
                            print(f"Inference error ({hand_label}): {exc}", file=sys.stderr)
                            tip_pressures = pose_finger_press_proxy(
                                points, frame_w, frame_h
                            )
                            tip_pressures = sm.update(tip_pressures)
                            draw_fingertip_pressures(display, tip_pressures)
                            tips_by_hand[hand_label] = tip_pressures
                            continue

                        overlay_heatmap_on_bbox(
                            display, contact_hm, bbox, alpha=0.55
                        )
                        pv2_tips = sample_fingertip_pressures(
                            contact_hm * float(args.gain),
                            points,
                            bbox,
                            frame_w,
                            frame_h,
                            max_force=1.0,
                        )
                        pose_tips = pose_finger_press_proxy(
                            points, frame_w, frame_h
                        )
                        pv2_alive = heat_stats.get("contact_peak", 0.0) > 0.03
                        last_source = "pv2+pose" if pv2_alive else "pose-proxy"
                        last_stats = heat_stats
                        merged: List[Tuple[str, Tuple[int, int], float, float]] = []
                        pose_by = {n: t for n, _xy, t, _r in pose_tips}
                        pix_by = {n: xy for n, xy, _t, _r in pose_tips}
                        for name, pix, tension, raw in pv2_tips:
                            pix_by[name] = pix
                            pose_t = pose_by.get(name, 0.0)
                            if pv2_alive:
                                t = max(float(tension), 0.65 * float(pose_t))
                                r = max(float(raw), 0.65 * float(pose_t))
                            else:
                                t = float(pose_t)
                                r = float(pose_t)
                            merged.append((name, pix_by[name], t, r))
                        if not merged:
                            merged = pose_tips
                        tip_pressures = sm.update(merged)
                        draw_fingertip_pressures(display, tip_pressures)
                        tips_by_hand[hand_label] = tip_pressures

                        if tip_pressures:
                            tensions.append(
                                max(t for _n, _xy, t, _r in tip_pressures)
                            )

                    if last_stats is not None:
                        cv2.putText(
                            display,
                            (
                                f"contact={last_stats.get('contact_peak', 0):.2f}  "
                                f"p_max={last_stats['p_contact_max']:.2f}  "
                                f"src={last_source}  hands={len(tips_by_hand)}"
                            ),
                            (16, 128),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255)
                            if last_source.startswith("pv2")
                            else (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    if tips_by_hand:
                        # Left hand panel on the left, Right on the right.
                        if "Left" in tips_by_hand:
                            ordered_l = [
                                next(
                                    (
                                        it
                                        for it in tips_by_hand["Left"]
                                        if it[0] == n
                                    ),
                                    (n, (0, 0), 0.0, 0.0),
                                )
                                for n in FINGERTIP_NAMES
                            ]
                            draw_finger_pressure_panel(
                                display,
                                ordered_l,
                                title="LEFT hand",
                                side="left",
                            )
                        if "Right" in tips_by_hand:
                            ordered_r = [
                                next(
                                    (
                                        it
                                        for it in tips_by_hand["Right"]
                                        if it[0] == n
                                    ),
                                    (n, (0, 0), 0.0, 0.0),
                                )
                                for n in FINGERTIP_NAMES
                            ]
                            draw_finger_pressure_panel(
                                display,
                                ordered_r,
                                title="RIGHT hand",
                                side="right",
                            )
                        # Any extra unlabeled hands: stack under right panel title.
                        for label, tips in tips_by_hand.items():
                            if label in ("Left", "Right"):
                                continue
                            draw_finger_pressure_panel(
                                display,
                                tips,
                                title=f"{label} hand",
                                side="right",
                            )

                        peak_tip = max(
                            t
                            for tips in tips_by_hand.values()
                            for _n, _xy, t, _r in tips
                        )
                        draw_tension_bar(
                            display,
                            peak_tip,
                            label="Both hands — peak fingertip press",
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
