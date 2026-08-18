#!/usr/bin/env python3
"""
Build synced multi-camera force datasets for:
  1) YOLO-pose hand training with per-fingertip pressure targets
  2) PressureVision++-style sequences (meta.json + force pkl + camera_*)

Usage:
  python scripts/build_force_pose_dataset.py --raw data/multicam_force/raw/<session>/<p>/<action>
  python scripts/build_force_pose_dataset.py --raw-root data/multicam_force/raw --split train
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from forcepad import PadSpec
from multicam import (
    detect_aruco,
    homography_pad_to_image,
    load_calibration,
    load_config,
    marker_centers,
    pad_quad_from_markers,
    warp_force_to_image,
)

# MediaPipe Hands landmark names (21)
HAND_LANDMARKS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def nearest_force_idx(t_cam: int, force_ts: List[dict], tol_ns: int) -> Optional[int]:
    if not force_ts:
        return None
    best = None
    best_dt = None
    for row in force_ts:
        dt = abs(int(row["t_ns"]) - int(t_cam))
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = int(row["idx"])
    if best_dt is None or best_dt > tol_ns:
        return None
    return best


def load_force(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr, dtype=np.float32)


def init_mediapipe(allow_empty: bool = True):
    """Prefer MediaPipe Tasks HandLandmarker; fall back to solutions; else empty detector."""
    model = Path("weights/hand_landmarker.task")
    if model.exists():
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base = mp_python.BaseOptions(model_asset_path=str(model))
            options = vision.HandLandmarkerOptions(
                base_options=base,
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)

            def detect(frame_bgr):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                hands = []
                h, w = frame_bgr.shape[:2]
                for i, lms in enumerate(result.hand_landmarks):
                    label = "Right"
                    if result.handedness and i < len(result.handedness):
                        cats = result.handedness[i]
                        if cats:
                            label = cats[0].category_name
                    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
                    hands.append({"label": label, "xy": pts})
                return hands

            return detect, landmarker
        except Exception as exc:
            print(f"Tasks HandLandmarker unavailable ({exc}); trying solutions…")

    try:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise AttributeError("mediapipe.solutions missing (Tasks-only install)")

        hands_mod = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )

        def detect(frame_bgr):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = hands_mod.process(rgb)
            out = []
            h, w = frame_bgr.shape[:2]
            if not result.multi_hand_landmarks:
                return out
            handedness = result.multi_handedness or []
            for i, lms in enumerate(result.multi_hand_landmarks):
                label = "Right"
                if i < len(handedness):
                    label = handedness[i].classification[0].label
                pts = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark], dtype=np.float32)
                out.append({"label": label, "xy": pts})
            return out

        return detect, hands_mod
    except Exception as exc:
        if not allow_empty:
            raise
        print(f"MediaPipe unavailable ({exc}); building force/geometry only (empty YOLO keypoints).")

        def detect(_frame_bgr):
            return []

        return detect, None


def yolo_pose_line(cls: int, xy: np.ndarray, img_w: int, img_h: int) -> str:
    """One YOLO-pose label line: cls cx cy w h (kx ky v)*21  (normalized)."""
    xs, ys = xy[:, 0], xy[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    # pad bbox slightly
    pad = 0.05 * max(x1 - x0, y1 - y0, 1.0)
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(img_w - 1.0, x1 + pad), min(img_h - 1.0, y1 + pad)
    cx = ((x0 + x1) / 2.0) / img_w
    cy = ((y0 + y1) / 2.0) / img_h
    bw = (x1 - x0) / img_w
    bh = (y1 - y0) / img_h
    parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for x, y in xy:
        parts.append(f"{x / img_w:.6f}")
        parts.append(f"{y / img_h:.6f}")
        parts.append("2")  # visible
    return " ".join(parts)


def sample_tip_forces(
    force_img: np.ndarray,
    xy: np.ndarray,
    tip_indices: Dict[str, int],
    radius: int,
    pad_spec: PadSpec,
) -> Dict[str, float]:
    """Mean kPa under each fingertip in the warped force image."""
    h, w = force_img.shape[:2]
    out = {}
    for name, idx in tip_indices.items():
        x, y = xy[idx]
        x_i, y_i = int(round(x)), int(round(y))
        x0, x1 = max(0, x_i - radius), min(w, x_i + radius + 1)
        y0, y1 = max(0, y_i - radius), min(h, y_i + radius + 1)
        patch = force_img[y0:y1, x0:x1]
        if patch.size == 0:
            out[name] = 0.0
            continue
        # force_img is already in counts if we warp counts; convert mean counts → kPa
        mean_counts = float(patch.mean())
        out[name] = float(pad_spec.counts_to_kpa(np.array([mean_counts]))[0])
    return out


def process_take(
    take_dir: Path,
    cfg: dict,
    split: str,
    detect_hands,
    calib: Optional[dict],
) -> Tuple[int, int]:
    ts_path = take_dir / "timestamps.json"
    if not ts_path.exists():
        print(f"SKIP {take_dir} (no timestamps.json)")
        return 0, 0

    with open(ts_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    pad_meta = meta.get("pad") or cfg["pad"]
    pad_spec = PadSpec(
        rows=int(pad_meta["rows"]),
        cols=int(pad_meta["cols"]),
        counts_per_newton=float(pad_meta.get("counts_per_newton", cfg["pad"]["counts_per_newton"])),
        pixel_pitch_m=float(pad_meta.get("pixel_pitch_m", cfg["pad"]["pixel_pitch_m"])),
    )
    corner_ids = list(cfg["aruco"]["corner_marker_ids"])
    tol_ns = int(float(cfg["recording"]["sync_tolerance_ms"]) * 1e6)
    tip_idx = {k: int(v) for k, v in cfg["yolo_pose"]["fingertips"].items()}
    tip_r = int(cfg["yolo_pose"]["tip_sample_radius_px"])
    contact_thr = float(cfg["yolo_pose"]["contact_thresh_kpa"])

    participant = meta["participant"]
    action = meta["action"]
    cameras = [str(c["id"]) for c in meta["cameras"]]

    # Index force timestamps (unique by force idx)
    force_ts = meta["force_timestamps"]
    # Camera timestamps grouped by frame idx (multi-cam share idx)
    cam_by_idx: Dict[int, Dict[str, int]] = {}
    for row in meta["camera_timestamps"]:
        cam_by_idx.setdefault(int(row["idx"]), {})[str(row["cam"])] = int(row["t_ns"])

    split_root = Path(cfg["paths"][f"{split}_root"])
    seq_dir = split_root / participant / action
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    for cid in cameras:
        (seq_dir / f"camera_{cid}").mkdir(parents=True, exist_ok=True)
    (seq_dir / "force").mkdir(parents=True, exist_ok=True)

    yolo_root = Path(cfg["paths"]["yolo_root"]) / split
    img_out = yolo_root / "images" / f"{participant}_{action}"
    lbl_out = yolo_root / "labels" / f"{participant}_{action}"
    force_out = yolo_root / "force" / f"{participant}_{action}"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    force_out.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    calibrations: Dict[str, Dict[str, dict]] = {cid: {} for cid in cameras}
    pose_dump: Dict[str, Dict[int, dict]] = {cid: {} for cid in cameras}

    num_src = int(meta["num_frames"])
    for idx in range(num_src):
        if idx not in cam_by_idx:
            dropped += 1
            continue
        # Use first camera timestamp as sync key
        t_ref = next(iter(cam_by_idx[idx].values()))
        f_idx = nearest_force_idx(t_ref, force_ts, tol_ns)
        if f_idx is None:
            dropped += 1
            continue

        force_src = take_dir / "force" / f"{f_idx:05d}.pkl"
        if not force_src.exists():
            # recorder wrote force with same idx as frames
            force_src = take_dir / "force" / f"{idx:05d}.pkl"
        if not force_src.exists():
            dropped += 1
            continue
        force = load_force(force_src)

        # Per-camera process
        any_cam = False
        for cid in cameras:
            img_path = take_dir / f"camera_{cid}" / f"{idx:05d}.jpg"
            if not img_path.exists():
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]

            corners, ids, _ = detect_aruco(frame, cfg["aruco"]["dictionary"])
            centers = marker_centers(corners, ids)
            quad = pad_quad_from_markers(centers, corner_ids)
            if quad is None and calib and cid in calib.get("cameras", {}):
                # fallback to static calib
                quad = np.asarray(calib["cameras"][cid]["imgpts"], dtype=np.float32)
            if quad is None:
                continue

            H = homography_pad_to_image((pad_spec.cols, pad_spec.rows), quad)
            calibrations[cid][str(kept)] = {
                "homography": H.tolist(),
                "imgpts": quad.tolist(),
            }

            # Save PV2 frame (reindexed)
            dst_img = seq_dir / f"camera_{cid}" / f"{kept:05d}.jpg"
            shutil.copy2(img_path, dst_img)
            if cid == cameras[0]:
                dst_force = seq_dir / "force" / f"{kept:05d}.pkl"
                with open(dst_force, "wb") as f:
                    pickle.dump(force, f, protocol=pickle.HIGHEST_PROTOCOL)

            force_img = warp_force_to_image(force, H, (h, w))
            hands = detect_hands(frame)

            # Pose dump (normalized)
            pose_entry = {"right_points": None, "left_points": None}
            label_lines = []
            tip_forces_all = {}
            for hand in hands:
                label = hand["label"]
                xy = hand["xy"]
                key = "right_points" if label.lower().startswith("r") else "left_points"
                pose_entry[key] = (xy / np.array([w, h], dtype=np.float32)).tolist()
                tip_f = sample_tip_forces(force_img, xy, tip_idx, tip_r, pad_spec)
                tip_forces_all[label] = tip_f
                # class 0 = left, 1 = right for YOLO
                cls = 1 if label.lower().startswith("r") else 0
                label_lines.append(yolo_pose_line(cls, xy, w, h))

            pose_dump[cid][kept] = pose_entry

            # YOLO image named with cam + frame
            stem = f"{cid}_{kept:05d}"
            cv2.imwrite(str(img_out / f"{stem}.jpg"), frame)
            with open(lbl_out / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

            force_sidecar = {
                "participant": participant,
                "action": action,
                "camera_id": cid,
                "frame": kept,
                "src_frame": idx,
                "force_src_idx": f_idx,
                "tip_force_kpa": tip_forces_all,
                "contact_thresh_kpa": contact_thr,
                "any_contact": bool(
                    any(
                        v >= contact_thr
                        for hand_f in tip_forces_all.values()
                        for v in hand_f.values()
                    )
                ),
                "total_force_newton": float(pad_spec.counts_to_newtons(force).sum()),
                "landmarks": HAND_LANDMARKS,
            }
            with open(force_out / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(force_sidecar, f, indent=2)

            any_cam = True

        if any_cam:
            kept += 1
        else:
            dropped += 1

    meta_out = {
        "camera_ids": cameras,
        "num_frames": kept,
        "timesteps": list(range(kept)),
        "is_weak": False,
        "camera_calibrations": calibrations,
        "participant": participant,
        "action": action,
        "source_raw": str(take_dir),
        "sync_tolerance_ms": cfg["recording"]["sync_tolerance_ms"],
        "dropped_frames": dropped,
        "created_unix": time.time(),
    }
    with open(seq_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    pose_root = Path("data/pose_estimates") / participant
    pose_root.mkdir(parents=True, exist_ok=True)
    with open(pose_root / f"{action}.pkl", "wb") as f:
        pickle.dump(pose_dump, f, protocol=pickle.HIGHEST_PROTOCOL)

    match_rate = kept / max(kept + dropped, 1)
    print(
        f"{take_dir}: kept={kept} dropped={dropped} match_rate={match_rate:.1%} "
        f"→ {seq_dir} + YOLO {img_out}"
    )
    if match_rate < 0.9:
        print("  WARNING: match rate < 90% — check sync / marker visibility")
    return kept, dropped


def find_takes(raw_root: Path) -> List[Path]:
    takes = []
    for ts in raw_root.rglob("timestamps.json"):
        takes.append(ts.parent)
    return sorted(takes)


def write_yolo_data_yaml(cfg: dict, split_names=("train", "val")) -> Path:
    root = Path(cfg["paths"]["yolo_root"]).resolve()
    names = {0: "left_hand", 1: "right_hand"}
    # Ultralytics pose: kpt_shape [21, 3]
    payload = {
        "path": str(root),
        "train": "images/train" if False else "train/images",
        # We store images under yolo_pose/{split}/images/...
        "names": names,
        "kpt_shape": [21, 3],
        "flip_idx": list(range(21)),  # hands are not left-right flipped symmetrically here
    }
    # Simpler explicit paths:
    lines = [
        f"path: {root}",
        f"train: train/images",
        f"val: val/images",
        "names:",
        "  0: left_hand",
        "  1: right_hand",
        "kpt_shape: [21, 3]",
    ]
    out = root / "data.yaml"
    root.mkdir(parents=True, exist_ok=True)
    # Ensure train/val image roots exist as unions — Ultralytics expects flat or nested;
    # we use nested participant folders which Ultralytics accepts recursively.
    for sp in split_names:
        (root / sp / "images").mkdir(parents=True, exist_ok=True)
        (root / sp / "labels").mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/multicam_force.yml")
    ap.add_argument("--raw", type=Path, default=None, help="Single take directory")
    ap.add_argument("--raw-root", type=Path, default=None, help="Scan all takes under root")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--calibration", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    calib_path = Path(args.calibration or cfg["paths"]["calibration"])
    calib = load_calibration(calib_path) if calib_path.exists() else None
    if calib is None:
        print("NOTE: no calibration JSON yet — relying on per-frame ArUco only")

    detect_hands, _engine = init_mediapipe()

    takes: List[Path] = []
    if args.raw:
        takes = [args.raw]
    else:
        root = args.raw_root or Path(cfg["paths"]["raw_root"])
        takes = find_takes(Path(root))
    if not takes:
        raise SystemExit("No takes found. Record first with scripts/record_multicam_force.py")

    total_k = total_d = 0
    for take in takes:
        k, d = process_take(take, cfg, args.split, detect_hands, calib)
        total_k += k
        total_d += d

    yml = write_yolo_data_yaml(cfg)
    # Symlink/copy nested images into split/images for Ultralytics flat discovery
    yolo_root = Path(cfg["paths"]["yolo_root"])
    for sp in ("train", "val"):
        src_images = yolo_root / sp / "images"
        # already nested under participant_action — OK
        src_images.mkdir(parents=True, exist_ok=True)

    print(f"Done. kept={total_k} dropped={total_d}")
    print(f"YOLO data yaml → {yml}")
    print("Train YOLO-pose (example):")
    print(f"  yolo pose train data={yml} model=yolo11n-pose.pt imgsz=640 epochs=50")


if __name__ == "__main__":
    main()
