"""Render stage: apply blur decisions to every frame and encode the output."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from safeframe.pipeline.anonymize import DEFAULT_PADDING, anonymize_regions
from safeframe.pipeline.interfaces import Cancelled, CancelToken
from safeframe.pipeline.policy import DecisionPolicy, Override
from safeframe.pipeline.types import Analysis, Box, ManualRegion
from safeframe.video.io import VideoWriter, iter_frames


@dataclass(slots=True)
class RenderPlan:
    regions: dict[int, list[Box]] = field(default_factory=dict)
    blurred_tracks: list[int] = field(default_factory=list)
    kept_tracks: list[int] = field(default_factory=list)

    def boxes_for(self, frame: int) -> list[Box]:
        return self.regions.get(frame, [])


def plan_render(
    analysis: Analysis,
    policy: DecisionPolicy,
    overrides: Mapping[int, Override] | None = None,
    manual_regions: Iterable[ManualRegion] = (),
) -> RenderPlan:
    """Decide which boxes to anonymise on which frames."""
    overrides = overrides or {}
    regions: dict[int, list[Box]] = defaultdict(list)
    plan = RenderPlan()
    for track in analysis.tracks:
        if policy.should_blur(track, overrides.get(track.track_id)):
            plan.blurred_tracks.append(track.track_id)
            for frame, box in track.boxes.items():
                regions[frame].append(box)
        else:
            plan.kept_tracks.append(track.track_id)
    last = max(analysis.video.frame_count - 1, 0)
    for region in manual_regions:
        for frame in range(max(0, region.start_frame), min(region.end_frame, last) + 1):
            regions[frame].append(region.box)
    plan.regions = dict(regions)
    return plan


def render_video(
    source: Path,
    output: Path,
    analysis: Analysis,
    plan: RenderPlan,
    progress: Callable[[float], None] | None = None,
    cancel: CancelToken | None = None,
    padding: float = DEFAULT_PADDING,
) -> None:
    """Write ``output`` with every planned region blurred and greyscaled.

    Resolution, exact frame rate and audio are preserved; untouched pixels are
    only re-encoded (visually lossless CRF 18).
    """
    info = analysis.video
    total = max(info.frame_count, 1)
    with VideoWriter(output, info, audio_source=source) as writer:
        for index, frame in iter_frames(source):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            boxes = plan.boxes_for(index)
            if boxes:
                anonymize_regions(frame, boxes, padding)
            writer.write(frame)
            if progress is not None and index % 10 == 0:
                progress(min(0.999, (index + 1) / total))
    if progress is not None:
        progress(1.0)
