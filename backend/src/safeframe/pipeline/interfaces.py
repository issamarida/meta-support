"""Component protocols. Real implementations wrap ML models; tests use fakes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from safeframe.pipeline.types import FaceDetection, PersonObservation


class PersonTracker(Protocol):
    """Detects people and assigns identities that persist across frames.

    A fresh instance must be used per video: implementations keep tracker
    state between calls to ``update``.
    """

    def update(self, frame: np.ndarray) -> list[PersonObservation]: ...


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceDetection]: ...


class AgeEstimator(Protocol):
    def estimate(self, faces: Sequence[np.ndarray]) -> list[tuple[float, float]]:
        """For each BGR face crop return ``(p_under_18, expected_age)``."""
        ...


class CancelToken(Protocol):
    def is_set(self) -> bool: ...


class Cancelled(Exception):
    """Raised when a job is cancelled mid-way."""
