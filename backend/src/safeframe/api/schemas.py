"""Request/response models for the HTTP API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from safeframe.pipeline.policy import DEFAULT_MIN_FACE_SAMPLES, DEFAULT_THRESHOLD, Override

PROBABILISTIC_NOTICE = (
    "Age estimation is probabilistic. SafeFrame estimates apparent age from visible "
    "faces and can be wrong, especially for teenagers, for people whose faces are "
    "small, turned away or partly hidden, and across demographic groups. Anyone it "
    "cannot confidently judge to be an adult is blurred by default. Always review the "
    "result before sharing."
)
INTENDED_USE = (
    "SafeFrame is intended to protect children's privacy in video you share. It is not "
    "an age-verification system and must not be used to identify, profile or make "
    "decisions about people."
)


class JobState(StrEnum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    REVIEW = "review"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_STATES = {JobState.QUEUED, JobState.ANALYZING, JobState.RENDERING}


class VideoInfoOut(BaseModel):
    width: int
    height: int
    fps: float
    fps_rational: str
    frame_count: int
    duration: float
    has_audio: bool


class JobOut(BaseModel):
    id: str
    filename: str
    state: JobState
    stage: str
    progress: float = Field(ge=0, le=1)
    error: str | None = None
    keep_upload: bool
    upload_deleted: bool
    output_ready: bool
    created_at: float
    updated_at: float
    expires_at: float
    video: VideoInfoOut | None = None


class ManualRegionIn(BaseModel):
    region_id: str = Field(min_length=1, max_length=64)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    box: tuple[float, float, float, float]

    @model_validator(mode="after")
    def _check(self) -> ManualRegionIn:
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        x1, y1, x2, y2 = self.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must have positive width and height")
        return self


class RenderRequest(BaseModel):
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0, le=1)
    min_face_samples: int = Field(DEFAULT_MIN_FACE_SAMPLES, ge=1, le=100)
    # Track id (as string, JSON object keys) -> forced decision.
    overrides: dict[int, Override] = Field(default_factory=dict)
    manual_regions: list[ManualRegionIn] = Field(default_factory=list, max_length=1000)


class AppConfigOut(BaseModel):
    version: str
    max_upload_mb: int
    job_ttl_hours: float
    default_threshold: float
    default_min_face_samples: int
    notice: str = PROBABILISTIC_NOTICE
    intended_use: str = INTENDED_USE
