"""Construct the real ML components (lazily, so importing SafeFrame stays cheap)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from safeframe.models import YUNET, fetch
from safeframe.pipeline.age import DEFAULT_AGE_MODEL
from safeframe.pipeline.detection import DEFAULT_YOLO_WEIGHTS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS
    age_model: str = DEFAULT_AGE_MODEL
    age_model_revision: str | None = None
    device: str | None = None  # None = auto (CUDA if available)


class ModelBundle:
    """Holds models that are safe to share between jobs (the age estimator)
    and builds the per-video stateful ones (tracker, face detector)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._lock = threading.Lock()
        self._age = None

    def new_tracker(self):
        from safeframe.pipeline.detection import YoloPersonTracker

        return YoloPersonTracker(self.config.yolo_weights, device=self.config.device)

    def new_face_detector(self):
        from safeframe.pipeline.faces import YuNetFaceDetector

        return YuNetFaceDetector(fetch(YUNET))

    def age_estimator(self):
        with self._lock:
            if self._age is None:
                from safeframe.pipeline.age import ViTAgeEstimator

                log.info("loading age model %s", self.config.age_model)
                self._age = ViTAgeEstimator(
                    self.config.age_model,
                    device=self.config.device,
                    revision=self.config.age_model_revision,
                )
            return self._age

    def warm_up(self) -> None:
        """Download/load everything once so the first job doesn't pay for it."""
        self.new_tracker()
        self.new_face_detector()
        self.age_estimator()
