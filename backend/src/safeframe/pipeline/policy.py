"""Turning age evidence into blur decisions.

The policy is deliberately asymmetric. Missing a child is a privacy failure;
blurring an adult is a usability cost the reviewer can undo. So:

* a track needs ``min_face_samples`` usable face observations before it can be
  called anything other than UNCERTAIN, and UNCERTAIN is blurred;
* the threshold on P(under 18) defaults well below 0.5.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from safeframe.pipeline.types import AgeSample, Track, TrackStatus

ADULT_AGE = 18
DEFAULT_THRESHOLD = 0.35
DEFAULT_MIN_FACE_SAMPLES = 3


class Override(StrEnum):
    BLUR = "blur"
    KEEP = "keep"


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    # A track is MINOR when its aggregated P(under 18) >= threshold.
    threshold: float = DEFAULT_THRESHOLD
    min_face_samples: int = DEFAULT_MIN_FACE_SAMPLES

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if self.min_face_samples < 1:
            raise ValueError("min_face_samples must be >= 1")

    def classify(self, track: Track) -> TrackStatus:
        if len(track.samples) < self.min_face_samples or track.p_minor is None:
            return TrackStatus.UNCERTAIN
        return TrackStatus.MINOR if track.p_minor >= self.threshold else TrackStatus.ADULT

    def should_blur(self, track: Track, override: Override | None = None) -> bool:
        if override is Override.BLUR:
            return True
        if override is Override.KEEP:
            return False
        return self.classify(track) is not TrackStatus.ADULT


# --- Age-bucket arithmetic -------------------------------------------------

_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
_OPEN_UPPER = re.compile(r"(?:more than|over|>|\+)\s*(\d+)|(\d+)\s*\+")


def parse_age_bucket(label: str) -> tuple[int, int]:
    """Parse labels like ``"10-19"``, ``"(15, 20)"``, ``"more than 70"``, ``"70+"``."""
    text = label.strip().lower().replace("(", "").replace(")", "").replace(",", "-")
    if m := _RANGE.search(text):
        lo, hi = int(m.group(1)), int(m.group(2))
        return min(lo, hi), max(lo, hi)
    if m := _OPEN_UPPER.search(text):
        lo = int(m.group(1) or m.group(2))
        return lo, max(lo, 100)
    raise ValueError(f"unrecognised age bucket label: {label!r}")


def minor_fraction(lo: int, hi: int) -> float:
    """Fraction of integer ages in [lo, hi] that are under 18 (uniform prior)."""
    total = hi - lo + 1
    minors = max(0, min(hi, ADULT_AGE - 1) - lo + 1)
    return minors / total


def bucket_stats(labels: Sequence[str]) -> tuple[list[float], list[float]]:
    """(minor fraction, midpoint age) for each bucket label."""
    fractions, mids = [], []
    for label in labels:
        lo, hi = parse_age_bucket(label)
        fractions.append(minor_fraction(lo, hi))
        mids.append((lo + hi) / 2)
    return fractions, mids


def aggregate(samples: Iterable[AgeSample]) -> tuple[float | None, float | None]:
    """Weighted mean of per-sample P(minor) and expected age."""
    total_w = p_sum = age_sum = 0.0
    for s in samples:
        total_w += s.weight
        p_sum += s.weight * s.p_minor
        age_sum += s.weight * s.expected_age
    if total_w <= 0:
        return None, None
    return p_sum / total_w, age_sum / total_w
