#!/usr/bin/env python3
"""
V2 interactive linker + collector: 7 cameras + pressure pad → training data.

One command to:
  1) Discover / link cameras
  2) Open + tare the pressure pad (mock or Tekscan)
  3) Live preview mosaic (all cams + force map)
  4) Record synced takes for model training
  5) Optionally build YOLO-pose + force exports when you quit

Examples:
  python scripts/v2_collect.py --dry-run
  python scripts/v2_collect.py --participant p01 --backend mock
  python scripts/v2_collect.py --participant p01 --backend tekscan --auto-build
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import pickle
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from forcepad import PadSpec, PressurePad, make_pad
from multicam import (
    CameraSpec,
    cameras_from_config,
    grab_frames,
    list_opencv_cameras,
    load_config,
    open_cameras,
)


ACTIONS = [
    "one_finger_thumb_low",
    "one_finger_thumb_high",
    "one_finger_index_low",
    "one_finger_index_high",
    "one_finger_middle_low",
    "one_finger_middle_high",
    "one_finger_ring_low",
    "one_finger_ring_high",
    "one_finger_pinky_low",
    "one_finger_pinky_high",
    "multi_finger_grip_low",
    "multi_finger_grip_high",
    "palm_press",
    "no_contact",
]


def mosaic_cameras(frames: Dict[str, np.ndarray], force: Optional[np.ndarray], cols: int = 4) -> np.ndarray:
    """Tile camera frames + optional force colormap into one preview image."""
    tiles: List[np.ndarray] = []
    ids = sorted(frames.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    for cid in ids:
        fr = frames[cid]
        small = cv2.resize(fr, (320, 240))
        cv2.putText(small, f"cam {cid}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(small, f"cam {cid}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        tiles.append(small)

    if force is not None:
        vis = force.astype(np.float32)
        if vis.max() > 0:
            vis = (255.0 * vis / vis.max()).astype(np.uint8)
        else:
            vis = np.zeros_like(vis, dtype=np.uint8)
        color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        color = cv2.resize(color, (320, 240))
        cv2.putText(color, "FORCE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        peak = float(force.max()) if force.size else 0.0
        cv2.putText(color, f"peak={peak:.0f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(color)

    if not tiles:
        return np.zeros((240, 320, 3), np.uint8)

    # pad to rectangle
    while len(tiles) % cols != 0:
        tiles.append(np.zeros_like(tiles[0]))
    rows = []
    for r in range(0, len(tiles), cols):
        rows.append(np.hstack(tiles[r : r + cols]))
    return np.vstack(rows)


def link_system(
    cfg: dict,
    backend: str,
    require_n: int,
    dry_run: bool,
) -> Tuple[List[CameraSpec], Dict[str, cv2.VideoCapture], PressurePad, PadSpec]:
    """Open cameras + pad; print a link status table."""
    print("=" * 60)
    print("V2 LINK: cameras + pressure pad")
    print("=" * 60)

    probed = list_opencv_cameras(max_index=max(12, require_n + 4))
    live = [p for p in probed if p["ok_frame"] and not p["likely_black"]]
    print(f"OpenCV probes: {len(probed)} opened, {len(live)} non-black")
    for p in probed:
        flag = "OK" if p["ok_frame"] and not p["likely_black"] else "bad"
        print(
            f"  idx={p['index']:2d}  {p['width']}x{p['height']}  "
            f"mean={p['mean_brightness']:6.1f}  [{flag}]"
        )

    cams = cameras_from_config(cfg)
    if dry_run and len(live) < require_n:
        # Keep configured cams; open may fail — caller uses synthetic path
        print(f"DRY-RUN: need {require_n} cams, found {len(live)} — will use synthetic frames if open fails")
    elif len(live) < require_n:
        print(
            f"WARNING: only {len(live)} usable cameras (want {require_n}). "
            "Fix USB/Sync Hub or edit config/multicam_force.yml indices."
        )

    pad_cfg = cfg["pad"]
    spec = PadSpec(
        rows=int(pad_cfg["rows"]),
        cols=int(pad_cfg["cols"]),
        counts_per_newton=float(pad_cfg["counts_per_newton"]),
        pixel_pitch_m=float(pad_cfg["pixel_pitch_m"]),
        width_mm=float(pad_cfg["width_mm"]),
        height_mm=float(pad_cfg["height_mm"]),
    )
    pad = make_pad(backend, spec)
    print(f"Opening pressure pad backend={backend!r} …")
    pad.open()
    pad.tare()
    force, _ = pad.read_frame()
    print(f"  pad OK  shape={force.shape}  peak={float(force.max()):.1f}")

    print(f"Opening {len(cams)} configured cameras …")
    caps: Dict[str, cv2.VideoCapture] = {}
    linked = []
    for spec_c in cams:
        cap = cv2.VideoCapture(spec_c.index)
        if not cap.isOpened():
            print(f"  cam id={spec_c.id} index={spec_c.index}  FAIL")
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  cam id={spec_c.id} index={spec_c.index}  FAIL (no frame)")
            cap.release()
            continue
        mean = float(frame.mean())
        if mean < 5:
            print(f"  cam id={spec_c.id} index={spec_c.index}  FAIL (black)")
            cap.release()
            continue
        print(f"  cam id={spec_c.id} index={spec_c.index}  OK  {frame.shape[1]}x{frame.shape[0]} mean={mean:.1f}")
        caps[spec_c.id] = cap
        linked.append(spec_c)

    print("-" * 60)
    print(f"LINKED: {len(linked)} cameras + pad({backend})")
    if not linked and not dry_run:
        pad.close()
        raise SystemExit("No cameras linked. Run: python scripts/list_cameras.py")
    return linked, caps, pad, spec


def record_take(
    caps: Dict[str, cv2.VideoCapture],
    pad: PressurePad,
    pad_spec: PadSpec,
    linked: List[CameraSpec],
    out_dir: Path,
    seconds: float,
    fps: float,
    jpeg_q: int,
    backend: str,
    participant: str,
    action: str,
    session: str,
    preview: bool,
) -> int:
    for c in linked:
        (out_dir / f"camera_{c.id}").mkdir(parents=True, exist_ok=True)
    (out_dir / "force").mkdir(parents=True, exist_ok=True)

    cam_ts: List[dict] = []
    force_ts: List[dict] = []
    n = 0
    period = 1.0 / max(fps, 1e-3)
    print(f"Recording {seconds:.1f}s @ ~{fps} FPS → {out_dir}")
    print("  SPACE=stop early   q=abort take")

    t0 = time.time()
    next_t = t0
    aborted = False
    while time.time() - t0 < seconds:
        now = time.time()
        if now < next_t:
            time.sleep(min(0.002, next_t - now))
            continue
        next_t += period

        if caps:
            frames, t_cam = grab_frames(caps)
        else:
            # synthetic RGB for dry-run without hardware
            frames = {
                c.id: np.full((480, 640, 3), 30 + (n % 5) * 5, np.uint8) for c in linked
            }
            t_cam = time.time_ns()
        force, t_force = pad.read_frame()

        for cid, fr in frames.items():
            path = out_dir / f"camera_{cid}" / f"{n:05d}.jpg"
            cv2.imwrite(str(path), fr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
            cam_ts.append({"cam": cid, "idx": n, "t_ns": int(t_cam)})

        with open(out_dir / "force" / f"{n:05d}.pkl", "wb") as f:
            pickle.dump(force.astype(np.float32), f, protocol=pickle.HIGHEST_PROTOCOL)
        force_ts.append({"idx": n, "t_ns": int(t_force)})

        if preview:
            mosaic = mosaic_cameras(frames, force)
            cv2.putText(
                mosaic,
                f"{action}  f={n}  {time.time()-t0:.1f}/{seconds:.0f}s",
                (12, mosaic.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("V2 collect", mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                aborted = True
                break
            if key == ord(" "):
                break
        n += 1

    meta = {
        "participant": participant,
        "action": action,
        "session": session,
        "backend": backend,
        "num_frames": n,
        "cameras": [{"id": c.id, "index": c.index, "name": c.name} for c in linked],
        "pad": {
            "rows": pad_spec.rows,
            "cols": pad_spec.cols,
            "counts_per_newton": pad_spec.counts_per_newton,
            "pixel_pitch_m": pad_spec.pixel_pitch_m,
        },
        "target_fps": fps,
        "camera_timestamps": cam_ts,
        "force_timestamps": force_ts,
        "aborted": aborted,
        "created_unix": time.time(),
        "v2": True,
    }
    with open(out_dir / "timestamps.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {n} frames" + (" (aborted)" if aborted else ""))
    return n


def live_preview_loop(
    caps: Dict[str, cv2.VideoCapture],
    pad: PressurePad,
    linked: List[CameraSpec],
) -> str:
    """Idle preview until user picks a key. Returns key char."""
    print()
    print("LIVE LINK PREVIEW")
    print("  [1-9 / a-e] pick action from list   r = custom action name")
    print("  t = tare pad   s = status   b = build dataset now   q = quit")
    for i, name in enumerate(ACTIONS):
        key = str(i + 1) if i < 9 else chr(ord("a") + (i - 9))
        print(f"    {key}: {name}")

    while True:
        if caps:
            try:
                frames, _ = grab_frames(caps)
            except RuntimeError as exc:
                print(f"grab error: {exc}")
                frames = {c.id: np.zeros((240, 320, 3), np.uint8) for c in linked}
        else:
            frames = {c.id: np.zeros((240, 320, 3), np.uint8) for c in linked}
            for cid, fr in frames.items():
                cv2.putText(fr, f"SYNTH {cid}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        force, _ = pad.read_frame()
        mosaic = mosaic_cameras(frames, force)
        cv2.putText(mosaic, "V2 linked — pick action key", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("V2 collect", mosaic)
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        return chr(key) if key < 128 else ""


def build_dataset(split: str = "train") -> None:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "build_force_pose_dataset.py"),
        "--raw-root",
        "data/multicam_force/raw",
        "--split",
        split,
    ]
    print("Building training export:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(_ROOT), check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/multicam_force.yml")
    ap.add_argument("--participant", default="p01")
    ap.add_argument("--backend", default=None, help="mock|tekscan (default: config)")
    ap.add_argument("--seconds", type=float, default=2.5, help="Hold duration per take")
    ap.add_argument("--require-cameras", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="Allow missing cams; use mock pad + synth frames")
    ap.add_argument("--auto-build", action="store_true", help="Run dataset build on quit")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--session", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = (args.backend or cfg["pad"].get("backend") or "mock").lower()
    if args.dry_run:
        backend = "mock"
        cfg["pad"]["backend"] = "mock"

    session = args.session or time.strftime("%Y-%m-%d_%H%M%S")
    fps = float(cfg["recording"]["target_fps"])
    jpeg_q = int(cfg["recording"].get("jpeg_quality", 90))
    raw_root = Path(cfg["paths"]["raw_root"])

    linked, caps, pad, pad_spec = link_system(
        cfg, backend, require_n=args.require_cameras, dry_run=args.dry_run
    )
    if args.dry_run and not caps:
        # fabricate linked cam list from config for synthetic recording
        linked = cameras_from_config(cfg)[: max(1, min(args.require_cameras, 7))]
        print(f"DRY-RUN synthetic cameras: {[c.id for c in linked]}")

    recorded = 0
    try:
        while True:
            if args.no_preview:
                action = "one_finger_index_high"
                key = None
            else:
                key = live_preview_loop(caps, pad, linked)

            if key == "q":
                break
            if key == "t":
                pad.tare()
                print("Pad tared")
                continue
            if key == "s":
                print(f"Linked cams={[c.id for c in linked]} backend={backend} recorded_takes={recorded}")
                continue
            if key == "b":
                build_dataset("train")
                continue
            if key is None:
                pass  # action already set for --no-preview
            elif key == "r":
                action = input("Action name: ").strip() or "custom_action"
            elif key.isdigit() and 1 <= int(key) <= 9:
                action = ACTIONS[int(key) - 1]
            elif key in "abcde":
                action = ACTIONS[9 + (ord(key) - ord("a"))]
            else:
                print(f"Unknown key {key!r}")
                continue

            out_dir = raw_root / session / args.participant / action
            # unique take folder if re-recording same action
            if (out_dir / "timestamps.json").exists():
                stamp = time.strftime("%H%M%S")
                out_dir = raw_root / session / args.participant / f"{action}_{stamp}"

            n = record_take(
                caps=caps,
                pad=pad,
                pad_spec=pad_spec,
                linked=linked,
                out_dir=out_dir,
                seconds=args.seconds,
                fps=fps,
                jpeg_q=jpeg_q,
                backend=backend,
                participant=args.participant,
                action=action,
                session=session,
                preview=not args.no_preview,
            )
            recorded += 1 if n > 0 else 0
            if args.no_preview:
                break
    finally:
        for c in caps.values():
            c.release()
        pad.close()
        cv2.destroyAllWindows()

    print(f"Session done. takes≈{recorded}  raw → {raw_root / session}")
    if args.auto_build or (recorded and not args.no_preview and input("Build YOLO/PV2 dataset now? [y/N] ").strip().lower() == "y"):
        build_dataset("train")
    print("Next:")
    print("  python scripts/build_force_pose_dataset.py --raw-root data/multicam_force/raw --split train")
    print("  python scripts/qc_force_overlay.py --seq data/multicam_force/train/<p>/<action> --camera-id 0")
    print("  yolo pose train data=data/multicam_force/yolo_pose/data.yaml model=yolo11n-pose.pt")


if __name__ == "__main__":
    main()
