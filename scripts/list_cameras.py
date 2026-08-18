#!/usr/bin/env python3
"""Probe OpenCV cameras and save preview JPEGs for picking your 7-cam layout."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path

import cv2

from multicam import list_opencv_cameras


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-index", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("weights/camera_previews"))
    ap.add_argument("--json-out", type=Path, default=Path("calibration/camera_probe.json"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    found = list_opencv_cameras(max_index=args.max_index)
    rows = []
    print(f"{'idx':>3}  {'ok':>3}  {'WxH':>11}  {'fps':>6}  {'mean':>7}  note")
    for info in found:
        note = "BLACK?" if info["likely_black"] else ""
        print(
            f"{info['index']:3d}  {str(info['ok_frame']):>3}  "
            f"{info['width']}x{info['height']:>4}  {info['fps']:6.1f}  "
            f"{info['mean_brightness']:7.1f}  {note}"
        )
        cap = cv2.VideoCapture(info["index"])
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            path = args.out / f"camera_{info['index']:02d}.jpg"
            cv2.imwrite(str(path), frame)
            info["preview"] = str(path)
        rows.append(info)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print()
    print(f"Previews → {args.out}/")
    print(f"JSON     → {args.json_out}")
    print("Edit config/multicam_force.yml cameras[].index to match your 7 working feeds.")
    if len([r for r in rows if r["ok_frame"] and not r["likely_black"]]) < 7:
        print("WARNING: fewer than 7 non-black cameras found — check Sync Hub / USB / power.")


if __name__ == "__main__":
    main()
