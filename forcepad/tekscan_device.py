"""
Plug in your Tekscan / I-Scan / Evolution SDK here.

`forcepad.TekscanPressurePad` imports this module's `TekscanDevice` when present.
Until the vendor SDK is installed, keep using `backend: mock`.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from forcepad import PadSpec


class TekscanDevice:
    """
    Minimal wrapper your SDK should satisfy.

    Replace the body of open/tare/read_frame with real device calls.
    Expected pressure map shape: (spec.rows, spec.cols) float counts.
    """

    def __init__(self, spec: PadSpec, device: Optional[str] = None):
        self.spec = spec
        self.device = device
        self._handle = None

    def open(self) -> None:
        # Example:
        #   self._handle = vendor.open(self.device)
        raise NotImplementedError(
            "Wire your Tekscan SDK in forcepad/tekscan_device.py "
            "(open/tare/read_frame), then set pad.backend: tekscan"
        )

    def close(self) -> None:
        self._handle = None

    def tare(self) -> None:
        raise NotImplementedError("Implement tare() with empty pad")

    def read_frame(self) -> Tuple[np.ndarray, int]:
        """Return (HxW counts, time.time_ns())."""
        raise NotImplementedError("Implement read_frame() → (pressure_map, t_ns)")
        # Example return:
        # arr = self._handle.capture()  # HxW
        # return np.asarray(arr, dtype=np.float32), time.time_ns()
