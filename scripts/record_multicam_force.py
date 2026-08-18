#!/usr/bin/env python3
"""
Record N cameras (default 7) + pressure pad with shared host timestamps.

Dry-run:
  python scripts/record_multicam_force.py --participant p01 --action one_finger_index_high --seconds 3

Real pad (after wiring Tekscan SDK):
  set pad.backend: tekscan in config, or pass --backend tekscan
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
import time
from pathlib import Path

import cv2
import numpy as np

from forcepad import PadSpec, make_pad
from multicam import cameras_from_config, grab_frames, load_config, open_cameras


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/multicam_force.yml")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--session", default=None, help="Session id (default: timestamp)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--countdown", type=float, default=1.0)
    ap.add_argument("--backend", default=None, help="mock|tekscan (overrides config)")
    ap.add_argument("--camera-ids", default=None, help="Comma list of camera ids to use")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cams = cameras_from_config(cfg)
    if args.camera_ids:
        want = {x.strip() for x in args.camera_ids.split(",") if x.strip()}
        cams = [c for c in cams if c.id in want]
    if not cams:
        raise SystemExit("No cameras selected")

    pad_cfg = cfg["pad"]
    backend = (args.backend or pad_cfg.get("backend") or "mock").lower()
    spec = PadSpec(
        rows=int(pad_cfg["rows"]),
        cols=int(pad_cfg["cols"]),
        counts_per_newton=float(pad_cfg["counts_per_newton"]),
        pixel_pitch_m=float(pad_cfg["pixel_pitch_m"]),
        width_mm=float(pad_cfg["width_mm"]),
        height_mm=float(pad_cfg["height_mm"]),
    )
    pad = make_pad(backend, spec)
    pad.open()
    pad.tare()

    session = args.session or time.strftime("%Y-%m-%d_%H%M%S")
    out_root = Path(args.out_root or cfg["paths"]["raw_root"])
    take_dir = out_root / session / args.participant / args.action
    for c in cams:
        (take_dir / f"camera_{c.id}").mkdir(parents=True, exist_ok=True)
    (take_dir / "force").mkdir(parents=True, exist_ok=True)

    print(f"Opening {len(cams)} cameras…")
    caps = open_cameras(cams)
    fps = float(cfg["recording"]["target_fps"])
    period = 1.0 / max(fps, 1e-3)
    jpeg_q = int(cfg["recording"].get("jpeg_quality", 90))

    cam_ts = []
    force_ts = []
    n_frames = 0

    if args.countdown > 0:
        print(f"Countdown {args.countdown:.1f}s — approach the pad")
        t_end = time.time() + args.countdown
        while time.time() < t_end:
            try:
                frames, _ = grab_frames(caps)
            except RuntimeError:
                frames = {}
            if not args.no_preview and frames:
                # mosaic first up to 4 cams
                tiles = []
                for i, (cid, fr) in enumerate(list(frames.items())[:4]):
                    small = cv2.resize(fr, (320, 240))
                    cv2.putText(small, f"cam{cid}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    tiles.append(small)
                while len(tiles) < 4:
                    tiles.append(np.zeros_like(tiles[0]))
                mosaic = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
                cv2.imshow("record preview", mosaic)
                cv2.waitKey(1)
            time.sleep(0.03)

    print(f"Recording {args.seconds:.1f}s → {take_dir}")
    t0 = time.time()
    next_t = t0
    try:
        while time.time() - t0 < args.seconds:
            now = time.time()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue
            next_t += period

            frames, t_cam = grab_frames(caps)
            force, t_force = pad.read_frame()

            for cid, fr in frames.items():
                path = take_dir / f"camera_{cid}" / f"{n_frames:05d}.jpg"
                cv2.imwrite(str(path), fr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                cam_ts.append({"cam": cid, "idx": n_frames, "t_ns": int(t_cam)})

            force_path = take_dir / "force" / f"{n_frames:05d}.pkl"
            with open(force_path, "wb") as f:
                pickle.dump(force.astype(np.float32), f, protocol=pickle.HIGHEST_PROTOCOL)
            force_ts.append({"idx": n_frames, "t_ns": int(t_force)})

            if not args.no_preview:
                # force colormap tile
                vis_f = force.copy()
                if vis_f.max() > 0:
                    vis_f = (255.0 * vis_f / vis_f.max()).astype(np.uint8)
                else:
                    vis_f = vis_f.astype(np.uint8)
                vis_f = cv2.applyColorMap(vis_f, cv2.COLORMAP_INFERNO)
                vis_f = cv2.resize(vis_f, (320, 120))
                tiles = []
                for cid, fr in list(frames.items())[:3]:
                    small = cv2.resize(fr, (320, 240))
                    cv2.putText(small, f"cam{cid}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    tiles.append(small)
                while len(tiles) < 3:
                    tiles.append(np.zeros((240, 320, 3), np.uint8))
                row = np.hstack(tiles)
                # pad force under
                pad_row = np.zeros((120, row.shape[1], 3), np.uint8)
                pad_row[:, : vis_f.shape[1]] = vis_f
                cv2.putText(pad_row, f"force f={n_frames}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                mosaic = np.vstack([row, pad_row])
                cv2.imshow("record preview", mosaic)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user")
                    break

            n_frames += 1
    finally:
        for c in caps.values():
            c.release()
        pad.close()
        cv2.destroyAllWindows()

    meta = {
        "participant": args.participant,
        "action": args.action,
        "session": session,
        "backend": backend,
        "num_frames": n_frames,
        "cameras": [{"id": c.id, "index": c.index, "name": c.name} for c in cams],
        "pad": {
            "rows": spec.rows,
            "cols": spec.cols,
            "counts_per_newton": spec.counts_per_newton,
            "pixel_pitch_m": spec.pixel_pitch_m,
        },
        "target_fps": fps,
        "camera_timestamps": cam_ts,
        "force_timestamps": force_ts,
        "created_unix": time.time(),
    }
    with open(take_dir / "timestamps.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {n_frames} frames @ ~{fps} FPS target → {take_dir}")
    print("Next: python scripts/build_force_pose_dataset.py --raw", take_dir)


if __name__ == "__main__":
    main()
