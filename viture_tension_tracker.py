#!/usr/bin/env python3
"""
Viture XR grip-tension tracker.

Captures frames from a Viture glasses UVC camera, tracks hands with MediaPipe,
crops detected hands for PressureVision++, and overlays a Muscle/Grip Tension
progress bar on the live OpenCV window.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
        description="Estimate grip tension from a Viture XR UVC camera feed."
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
        help="MediaPipe Hands min_detection_confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe Hands min_tracking_confidence.",
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
        checkpoint = torch.load(
            str(weights_path), map_location=device, weights_only=False
        )
    except TypeError:
        # Older torch without weights_only=
        checkpoint = torch.load(str(weights_path), map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.to(device)
    elif isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        # Strip common DataParallel prefixes if present.
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


def landmarks_to_bbox(
    landmarks,
    frame_w: int,
    frame_h: int,
    padding: float = HAND_PADDING,
) -> Optional[Tuple[int, int, int, int]]:
    """Convert MediaPipe hand landmarks to a padded pixel AABB (x1,y1,x2,y2)."""
    xs = [lm.x * frame_w for lm in landmarks.landmark]
    ys = [lm.y * frame_h for lm in landmarks.landmark]
    if not xs or not ys:
        return None

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
    # FPN_DANN_Logits returns (seg_logits, ...) — seg_logits: N x C x H x W
    force_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    force_class = torch.argmax(force_logits, dim=1)
    force_scalar = classes_to_scalar(force_class, config.FORCE_THRESHOLDS)
    return force_scalar.detach().cpu().squeeze().numpy()


def tension_from_heatmap(
    heatmap: np.ndarray,
    max_force: float,
    mode: str = "peak",
) -> Tuple[float, float]:
    """
    Reduce a pressure heatmap to (tension_01, raw_force).

    tension_01 is clamped to [0, 1] for the UI bar.
    """
    if heatmap.size == 0:
        return 0.0, 0.0

    # Focus on contact pixels; zeros dominate when averaging the full crop.
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

    # Label
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

    # Track
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)

    # Fill — green -> yellow -> red as tension rises
    fill_w = int(bar_w * np.clip(tension, 0.0, 1.0))
    if fill_w > 0:
        t = float(np.clip(tension, 0.0, 1.0))
        if t < 0.5:
            # green -> yellow
            g = 1.0
            r = t * 2.0
        else:
            # yellow -> red
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

    # Normalize heatmap for color mapping (avoid divide-by-zero)
    hm = heatmap.astype(np.float32)
    peak = float(hm.max()) if hm.size else 0.0
    if peak <= 1e-6:
        return
    norm = np.clip(hm / peak, 0.0, 1.0)
    color_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(color_u8, cv2.COLORMAP_JET)
    color = cv2.resize(color, (region_w, region_h), interpolation=cv2.INTER_LINEAR)

    roi = frame[y1:y2, x1:x2]
    # Only blend where pressure is non-trivial
    mask = cv2.resize(
        (hm > 0).astype(np.uint8) * 255,
        (region_w, region_h),
        interpolation=cv2.INTER_NEAREST,
    )
    blended = cv2.addWeighted(roi, 1.0 - alpha, color, alpha, 0.0)
    roi[mask > 0] = blended[mask > 0]
    frame[y1:y2, x1:x2] = roi


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # Prefer V4L2 on Linux for UVC devices (Viture glasses).
    backends: Sequence[int] = []
    if hasattr(cv2, "CAP_V4L2"):
        backends = (cv2.CAP_V4L2, cv2.CAP_ANY)
    else:
        backends = (cv2.CAP_ANY,)

    cap = None
    for backend in backends:
        candidate = cv2.VideoCapture(index, backend)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Unable to open camera index {index}. "
            "Try --camera-index 0 or verify the Viture UVC device is connected."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if hasattr(cv2, "VideoWriter_fourcc"):
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main() -> int:
    args = parse_args()
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

    try:
        model = load_pressurevision_model(args.weights, config, device)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        cap = open_camera(args.camera_index, args.width, args.height)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    window = "Viture Grip Tension Tracker"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps_ema = 0.0
    last_t = time.perf_counter()
    last_tension = 0.0

    print(
        f"Streaming camera {args.camera_index} | device={device} | "
        f"tension_mode={args.tension_mode}. Press 'q' to quit."
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Brief stall (USB / UVC hiccup) — keep the loop alive.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            display = frame.copy()
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_detected = bool(results.multi_hand_landmarks)
            tensions: List[float] = []

            if hand_detected:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        display,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                    bbox = landmarks_to_bbox(
                        hand_landmarks, frame_w, frame_h, padding=HAND_PADDING
                    )
                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 128), 2)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    try:
                        heatmap = run_pressure_inference(model, crop, config, device)
                    except Exception as exc:  # keep UI alive on a bad frame
                        print(f"Inference error: {exc}", file=sys.stderr)
                        continue

                    tension, _raw = tension_from_heatmap(
                        heatmap, args.max_force, mode=args.tension_mode
                    )
                    tensions.append(tension)
                    overlay_heatmap_on_bbox(display, heatmap, bbox)

                if tensions:
                    # Multi-hand: report the strongest grip.
                    last_tension = max(tensions)
                    draw_tension_bar(display, last_tension)
                else:
                    draw_searching_overlay(display)
            else:
                # No hands — skip PressureVision++ to save compute.
                last_tension = 0.0
                draw_searching_overlay(display)

            # FPS readout
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

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
