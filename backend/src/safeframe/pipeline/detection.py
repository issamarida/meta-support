"""Person detection (YOLO) and multi-object tracking (BoT-SORT with ReID)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

from safeframe.models import yolo_weights_path
from safeframe.pipeline.types import Box, PersonObservation

PERSON_CLASS = 0  # COCO "person"
DEFAULT_YOLO_WEIGHTS = "yolo11n.pt"


def tracker_config() -> Path:
    """Tuned BoT-SORT config: long track buffer + appearance ReID so identities
    survive head turns and occlusions of a second or two."""
    return Path(str(resources.files("safeframe.pipeline") / "botsort_safeframe.yaml"))


class YoloPersonTracker:
    """Ultralytics YOLO detector + BoT-SORT tracker, restricted to people.

    Detection confidence is intentionally low: BoT-SORT's second association
    stage uses low-score boxes to keep tracks alive through blur and partial
    occlusion, and for a privacy tool recall matters more than precision.
    """

    def __init__(
        self,
        weights: str = DEFAULT_YOLO_WEIGHTS,
        device: str | None = None,
        conf: float = 0.1,
        iou: float = 0.6,
        imgsz: int = 960,
    ) -> None:
        from ultralytics import YOLO

        path = Path(weights)
        if not path.is_absolute() and path.parent == Path("."):
            # Bare model name: keep the download out of the working directory.
            path = yolo_weights_path(weights)
        self._model = YOLO(str(path))
        self._kwargs = dict(
            persist=True,
            tracker=str(tracker_config()),
            classes=[PERSON_CLASS],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
        if device:
            self._kwargs["device"] = device

    def update(self, frame: np.ndarray) -> list[PersonObservation]:
        result = self._model.track(frame, **self._kwargs)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        return [
            PersonObservation(track_id=int(tid), box=Box(*map(float, b)), confidence=float(c))
            for b, tid, c in zip(xyxy, ids, confs, strict=True)
        ]
