#!/usr/bin/env python3
"""
Calibrate pressure-pad ArUco markers for one or all cameras.

Live view: markers highlighted; when all 4 corner IDs are visible, press
  s  — save per-camera pad quad + template calibration JSON
  q  — quit

Saves calibration/pad_markers.json used by recording and dataset build.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from multicam import (
    cameras_from_config,
    detect_aruco,
    homography_pad_to_image,
    load_config,
    marker_centers,
    pad_quad_from_markers,
    save_calibration,
)


def draw_overlay(frame, corners, ids, quad, cam_label: str):
    vis = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    if quad is not None:
        pts = quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(quad):
            cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(vis, f"{i}", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    status = "READY (press s)" if quad is not None else "Need 4 corner markers"
    cv2.putText(vis, f"{cam_label}  {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(vis, f"{cam_label}  {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/multicam_force.yml")
    ap.add_argument("--camera-id", default=None, help="Calibrate one camera id (default: cycle all)")
    ap.add_argument("--out", default=None, help="Override calibration JSON path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cams = cameras_from_config(cfg)
    if args.camera_id is not None:
        cams = [c for c in cams if c.id == str(args.camera_id)]
        if not cams:
            raise SystemExit(f"No camera id={args.camera_id} in config")

    pad_cfg = cfg["pad"]
    aruco_cfg = cfg["aruco"]
    corner_ids = list(aruco_cfg["corner_marker_ids"])
    out_path = Path(args.out or cfg["paths"]["calibration"])

    payload = {
        "created_unix": time.time(),
        "aruco": aruco_cfg,
        "pad": pad_cfg,
        "cameras": {},
    }

    print("Corner marker IDs (TL, TR, BR, BL):", corner_ids)
    print("Keys: s=save this camera, n=next camera, q=quit")

    for spec in cams:
        cap = cv2.VideoCapture(spec.index)
        if not cap.isOpened():
            print(f"SKIP cam {spec.id} index={spec.index} — cannot open")
            continue
        win = f"calibrate cam {spec.id}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        saved = False
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"cam {spec.id}: grab failed")
                break
            corners, ids, _ = detect_aruco(frame, aruco_cfg["dictionary"])
            centers = marker_centers(corners, ids)
            quad = pad_quad_from_markers(centers, corner_ids)
            vis = draw_overlay(frame, corners, ids, quad, f"cam{spec.id}/idx{spec.index}")
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                save_calibration(out_path, payload)
                print(f"Wrote {out_path}")
                return
            if key == ord("n"):
                break
            if key == ord("s"):
                if quad is None:
                    print("Cannot save — missing corner markers")
                    continue
                rows, cols = int(pad_cfg["rows"]), int(pad_cfg["cols"])
                H = homography_pad_to_image((cols, rows), quad)
                payload["cameras"][spec.id] = {
                    "index": spec.index,
                    "name": spec.name,
                    "image_size": [int(frame.shape[1]), int(frame.shape[0])],
                    "imgpts": quad.tolist(),
                    "homography": H.tolist(),
                    "marker_centers": {str(k): v.tolist() for k, v in centers.items()},
                    "saved_unix": time.time(),
                }
                print(f"Saved calibration for camera {spec.id}")
                saved = True
                break
        cap.release()
        cv2.destroyWindow(win)
        if not saved:
            print(f"No save for camera {spec.id}")

    save_calibration(out_path, payload)
    print(f"Wrote {out_path} with {len(payload['cameras'])} cameras")
    print("Re-run if you move the pad frame. During recording we still detect markers per frame.")


if __name__ == "__main__":
    main()
