"""Pressure-pad backends: mock (dry-run) and Tekscan stub."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PadSpec:
    rows: int = 64
    cols: int = 256
    counts_per_newton: float = 1736.0
    pixel_pitch_m: float = 0.00125
    width_mm: float = 203.2
    height_mm: float = 76.2

    def counts_to_newtons(self, counts: np.ndarray) -> np.ndarray:
        return counts.astype(np.float64) / float(self.counts_per_newton)

    def counts_to_kpa(self, counts: np.ndarray) -> np.ndarray:
        force_n = self.counts_to_newtons(counts)
        pa = force_n / (self.pixel_pitch_m ** 2)
        return pa / 1000.0


class PressurePad(ABC):
    """Common interface for all pad backends."""

    def __init__(self, spec: PadSpec):
        self.spec = spec

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def tare(self) -> None:
        """Zero / tare with nothing on the pad."""

    @abstractmethod
    def read_frame(self) -> Tuple[np.ndarray, int]:
        """Return (pressure_map HxW counts, timestamp_ns)."""


class MockPressurePad(PressurePad):
    """Synthetic Gaussian contact blobs for pipeline dry-runs."""

    def __init__(self, spec: PadSpec, seed: int = 0):
        super().__init__(spec)
        self._rng = np.random.default_rng(seed)
        self._t0 = time.time_ns()
        self._frame = 0
        self._baseline = np.zeros((spec.rows, spec.cols), dtype=np.float32)
        self._open = False
        self.pressing = True
        self.contact_xy = (spec.cols // 2, spec.rows // 2)
        self.peak_counts = 4000.0

    def open(self) -> None:
        self._open = True
        self._t0 = time.time_ns()

    def close(self) -> None:
        self._open = False

    def tare(self) -> None:
        self._baseline = np.zeros((self.spec.rows, self.spec.cols), dtype=np.float32)

    def set_contact(self, x: float, y: float, peak_counts: float = 4000.0, pressing: bool = True) -> None:
        self.contact_xy = (float(x), float(y))
        self.peak_counts = float(peak_counts)
        self.pressing = bool(pressing)

    def read_frame(self) -> Tuple[np.ndarray, int]:
        if not self._open:
            raise RuntimeError("MockPressurePad not open")
        h, w = self.spec.rows, self.spec.cols
        frame = self._baseline.copy()
        if self.pressing:
            cx, cy = self.contact_xy
            # Slow drift so consecutive frames are not identical
            cx += 3.0 * np.sin(self._frame / 15.0)
            cy += 2.0 * np.cos(self._frame / 11.0)
            ys, xs = np.mgrid[0:h, 0:w]
            sigma = 4.5
            blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
            frame = frame + (blob * self.peak_counts).astype(np.float32)
            frame += self._rng.normal(0, 15, size=frame.shape).astype(np.float32)
            frame = np.clip(frame, 0, None)
        t_ns = time.time_ns()
        self._frame += 1
        return frame.astype(np.float32), int(t_ns)


class TekscanPressurePad(PressurePad):
    """
    Stub for a real Tekscan SDK / CSV / Evolution handle.

    Wire your vendor SDK here so `read_frame()` returns the live pressure grid.
    Until then, raise on open() with install instructions.
    """

    def __init__(self, spec: PadSpec, device: Optional[str] = None):
        super().__init__(spec)
        self.device = device
        self._handle = None

    def open(self) -> None:
        try:
            # Placeholder: users replace this with vendor Python bindings.
            import tekscan_sdk  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Tekscan backend selected but no vendor SDK is importable.\n"
                "Install your Tekscan / I-Scan / Evolution Python bindings, then\n"
                "implement open()/read_frame()/tare() in forcepad/tekscan_pad.py,\n"
                "or dry-run with pad.backend: mock in config/multicam_force.yml."
            ) from exc

    def close(self) -> None:
        self._handle = None

    def tare(self) -> None:
        raise NotImplementedError("Implement tare() with your Tekscan SDK")

    def read_frame(self) -> Tuple[np.ndarray, int]:
        raise NotImplementedError("Implement read_frame() with your Tekscan SDK")


def make_pad(backend: str, spec: PadSpec, **kwargs) -> PressurePad:
    backend = (backend or "mock").lower().strip()
    if backend == "mock":
        return MockPressurePad(spec, **kwargs)
    if backend in {"tekscan", "tekscan5330", "real"}:
        return TekscanPressurePad(spec, **kwargs)
    raise ValueError(f"Unknown pad backend: {backend!r} (use mock|tekscan)")
