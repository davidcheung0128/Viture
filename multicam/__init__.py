"""Multi-camera helpers: discovery, capture, ArUco pad alignment."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class CameraSpec:
    id: str
    index: int
    name: str = ""


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cameras_from_config(cfg: dict) -> List[CameraSpec]:
    out: List[CameraSpec] = []
    for c in cfg.get("cameras", []):
        out.append(CameraSpec(id=str(c["id"]), index=int(c["index"]), name=str(c.get("name", ""))))
    return out


def list_opencv_cameras(max_index: int = 12, settle_s: float = 0.2) -> List[dict]:
    """Probe OpenCV indices; return metadata + mean brightness."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        time.sleep(settle_s)
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        mean = float(frame.mean()) if ok and frame is not None else -1.0
        black = bool(ok and mean < 5.0)
        found.append(
            {
                "index": idx,
                "opened": True,
                "ok_frame": bool(ok),
                "width": w,
                "height": h,
                "fps": fps,
                "mean_brightness": mean,
                "likely_black": black,
            }
        )
        cap.release()
    return found


def open_cameras(specs: Sequence[CameraSpec]) -> Dict[str, cv2.VideoCapture]:
    caps: Dict[str, cv2.VideoCapture] = {}
    try:
        for spec in specs:
            cap = cv2.VideoCapture(spec.index)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera id={spec.id} index={spec.index}")
            caps[spec.id] = cap
    except Exception:
        for c in caps.values():
            c.release()
        raise
    return caps


def grab_frames(caps: Dict[str, cv2.VideoCapture]) -> Tuple[Dict[str, np.ndarray], int]:
    """Grab one frame from each open camera; shared host timestamp."""
    t_ns = time.time_ns()
    frames: Dict[str, np.ndarray] = {}
    for cam_id, cap in caps.items():
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Camera {cam_id} failed to grab a frame")
        frames[cam_id] = frame
    return frames, t_ns


def aruco_dictionary(name: str):
    name = name.upper()
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_aruco(frame_bgr: np.ndarray, dictionary_name: str = "DICT_4X4_50"):
    dictionary = aruco_dictionary(dictionary_name)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, rejected = detector.detectMarkers(frame_bgr)
    return corners, ids, rejected


def marker_centers(corners, ids) -> Dict[int, np.ndarray]:
    """Map marker id → center pixel (x, y)."""
    out: Dict[int, np.ndarray] = {}
    if ids is None:
        return out
    for i, mid in enumerate(ids.flatten()):
        c = corners[i][0]  # 4x2
        out[int(mid)] = c.mean(axis=0)
    return out


def pad_quad_from_markers(
    centers: Dict[int, np.ndarray],
    corner_marker_ids: Sequence[int],
) -> Optional[np.ndarray]:
    """
    Build pad quad (TL, TR, BR, BL) from four marker centers.
    Returns float32 (4, 2) or None if any marker missing.
    """
    if len(corner_marker_ids) != 4:
        raise ValueError("corner_marker_ids must have 4 IDs (TL, TR, BR, BL)")
    pts = []
    for mid in corner_marker_ids:
        if int(mid) not in centers:
            return None
        pts.append(centers[int(mid)])
    return np.asarray(pts, dtype=np.float32)


def homography_pad_to_image(pad_wh: Tuple[int, int], img_quad: np.ndarray) -> np.ndarray:
    """
    Homography mapping pad grid coords (col, row) in [0..W, 0..H] → image pixels.
    img_quad: TL, TR, BR, BL in image space.
    """
    w, h = pad_wh  # (cols, rows) as width/height in grid units
    src = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = np.asarray(img_quad, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def warp_force_to_image(
    force_hw: np.ndarray,
    H: np.ndarray,
    image_shape_hw: Tuple[int, int],
) -> np.ndarray:
    """Warp HxW force map into full image using pad→image homography."""
    h_img, w_img = image_shape_hw
    return cv2.warpPerspective(
        force_hw.astype(np.float32),
        H,
        (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def save_calibration(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_calibration(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
