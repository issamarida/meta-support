"""Face detection with OpenCV's YuNet, and matching faces to tracked people."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from safeframe.pipeline.types import Box, FaceDetection, PersonObservation


class YuNetFaceDetector:
    """Lightweight CNN face detector (opencv_zoo, MIT licensed).

    Frames are downscaled so the longest side is at most ``max_side`` pixels;
    YuNet is trained on that scale range and it keeps detection fast on 4K.
    """

    def __init__(
        self, model_path: Path, score_threshold: float = 0.6, max_side: int = 1280
    ) -> None:
        self.max_side = max_side
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = frame if scale == 1.0 else cv2.resize(frame, (round(w * scale), round(h * scale)))
        self._detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self._detector.detect(img)
        if faces is None:
            return []
        out = []
        for row in faces:
            x, y, fw, fh = (float(v) / scale for v in row[:4])
            out.append(FaceDetection(Box(x, y, x + fw, y + fh), float(row[-1])))
        return out


def assign_faces(
    faces: Sequence[FaceDetection],
    people: Sequence[PersonObservation],
    head_fraction: float = 0.5,
) -> dict[int, FaceDetection]:
    """Match each face to at most one tracked person and vice versa.

    A face belongs to a person if its centre lies in the top ``head_fraction``
    of the person box. When boxes overlap (a child held by an adult) the
    smallest containing box wins, because it is the most specific match.
    Each person keeps only their highest-scoring face.
    """
    assigned: dict[int, FaceDetection] = {}
    for face in faces:
        cx, cy = face.box.center
        candidates = [
            p
            for p in people
            if p.box.contains(cx, cy)
            and cy <= p.box.y1 + head_fraction * p.box.height
            and face.box.width <= 1.2 * p.box.width
        ]
        if not candidates:
            continue
        owner = min(candidates, key=lambda p: p.box.area)
        current = assigned.get(owner.track_id)
        if current is None or face.score > current.score:
            assigned[owner.track_id] = face
    return assigned


def crop_face(frame: np.ndarray, face: Box, margin: float = 0.25) -> np.ndarray | None:
    """Square crop centred on the face, with context margin, clipped to frame."""
    h, w = frame.shape[:2]
    cx, cy = face.center
    side = max(face.width, face.height) * (1 + 2 * margin)
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]
