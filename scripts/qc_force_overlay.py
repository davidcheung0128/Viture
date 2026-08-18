#!/usr/bin/env python3
"""
QC: overlay warped pressure on RGB and verify blob sits under contacting fingertips.

  python scripts/qc_force_overlay.py --seq data/multicam_force/train/p01/one_finger_index_high
  python scripts/qc_force_overlay.py --seq ... --camera-id 0 --save-dir /tmp/qc
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
from pathlib import Path

import cv2
import numpy as np


def pressure_colormap(force: np.ndarray) -> np.ndarray:
    vis = force.astype(np.float32)
    if vis.max() > 0:
        vis = vis / vis.max()
    vis = (vis * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--camera-id", default="0")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    meta = json.loads((args.seq / "meta.json").read_text(encoding="utf-8"))
    cam = str(args.camera_id)
    if cam not in meta["camera_ids"]:
        raise SystemExit(f"camera {cam} not in {meta['camera_ids']}")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    n = min(int(meta["num_frames"]), args.max_frames)
    for i in range(n):
        img_path = args.seq / f"camera_{cam}" / f"{i:05d}.jpg"
        force_path = args.seq / "force" / f"{i:05d}.pkl"
        if not img_path.exists() or not force_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        with open(force_path, "rb") as f:
            force = np.asarray(pickle.load(f), dtype=np.float32)

        calib = meta["camera_calibrations"].get(cam, {}).get(str(i))
        if calib is None:
            print(f"frame {i}: no homography")
            continue
        H = np.asarray(calib["homography"], dtype=np.float32)
        h, w = frame.shape[:2]
        warped = cv2.warpPerspective(force, H, (w, h))
        color = pressure_colormap(warped)
        overlay = cv2.addWeighted(frame, 0.65, color, 0.35, 0)
        # draw pad quad
        pts = np.asarray(calib["imgpts"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.putText(overlay, f"{args.seq.name} cam{cam} f={i}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if args.save_dir:
            cv2.imwrite(str(args.save_dir / f"qc_{cam}_{i:05d}.jpg"), overlay)
        if not args.no_show:
            cv2.imshow("qc overlay", overlay)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

    if not args.no_show:
        cv2.destroyAllWindows()
    print("QC done — accept only if pressure blob sits under the contacting fingertip(s).")


if __name__ == "__main__":
    main()
