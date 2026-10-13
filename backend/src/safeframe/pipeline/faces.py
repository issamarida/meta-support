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


# Expected head centre, as a fraction of person-box height below its top edge.
HEAD_Y = 0.12
# If the two best owners for a face are this close, the face is ambiguous.
AMBIGUITY_MARGIN = 0.05


def _head_distance(face: FaceDetection, person: PersonObservation) -> float:
    """Distance from the face centre to the person's expected head position,
    normalised by the person's height (scale-invariant)."""
    fx, fy = face.box.center
    hx = (person.box.x1 + person.box.x2) / 2
    hy = person.box.y1 + HEAD_Y * person.box.height
    return float(np.hypot(fx - hx, fy - hy)) / max(person.box.height, 1.0)


def assign_faces(
    faces: Sequence[FaceDetection],
    people: Sequence[PersonObservation],
    head_fraction: float = 0.5,
) -> dict[int, FaceDetection]:
    """Match each face to at most one tracked person.

    Candidates are people whose box contains the face centre within its top
    ``head_fraction``. The owner is the candidate whose expected head position
    is closest to the face. This handles overlaps both ways: a child held in
    front of an adult, and a child walking past someone further back.

    A face whose two best candidates are nearly tied is dropped. Attributing
    an adult's face to a child would push that child toward "adult"; dropping
    the sample only leaves the track closer to UNCERTAIN, which is blurred.
    Each person keeps only their highest-scoring face.
    """
    assigned: dict[int, FaceDetection] = {}
    for face in faces:
        cx, cy = face.box.center
        candidates = sorted(
            (
                (_head_distance(face, p), p)
                for p in people
                if p.box.contains(cx, cy)
                and cy <= p.box.y1 + head_fraction * p.box.height
                and face.box.width <= 1.2 * p.box.width
            ),
            key=lambda c: c[0],
        )
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < AMBIGUITY_MARGIN:
            continue
        owner = candidates[0][1]
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
