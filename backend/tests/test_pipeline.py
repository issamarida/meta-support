"""End-to-end analysis + render on synthetic video with scripted detectors."""

import json
import subprocess
import threading

import numpy as np
import pytest

from safeframe.pipeline.analyze import AnalysisSettings, analyze_video
from safeframe.pipeline.interfaces import Cancelled
from safeframe.pipeline.policy import DecisionPolicy, Override
from safeframe.pipeline.render import plan_render, render_video
from safeframe.pipeline.types import Box, ManualRegion, TrackStatus
from safeframe.video.io import iter_frames, probe

from .conftest import HeadFaceDetector, ScriptedTracker, SizeAgeEstimator, person, requires_ffmpeg

pytestmark = requires_ffmpeg

CHILD, ADULT, BACK_TURNED = 1, 2, 3
CHILD_BOX = (20, 40, 100, 220)
ADULT_BOX = (180, 20, 300, 230)
BACK_BOX = (120, 60, 170, 200)
OCCLUDED = range(20, 25)


def _script(frames=50):
    script = {}
    for f in range(frames):
        people = [person(ADULT, *ADULT_BOX)]
        if f not in OCCLUDED:
            people.append(person(CHILD, *CHILD_BOX))
        if 10 <= f <= 30:
            people.append(person(BACK_TURNED, *BACK_BOX))
        script[f] = people
    return script


def _analyze(video, **settings):
    tracker = ScriptedTracker(_script())
    faces = HeadFaceDetector(tracker, face_sizes={CHILD: 34, ADULT: 50})
    ages = SizeAgeEstimator(child_below_px=60)
    return analyze_video(video, tracker, faces, ages, AnalysisSettings(age_every=1, **settings))


def test_analysis_classifies_and_keeps_identity_through_occlusion(make_video):
    result = _analyze(make_video())
    analysis = result.analysis
    tracks = {t.track_id: t for t in analysis.tracks}
    policy = DecisionPolicy()

    assert policy.classify(tracks[CHILD]) is TrackStatus.MINOR
    assert policy.classify(tracks[ADULT]) is TrackStatus.ADULT
    assert policy.classify(tracks[BACK_TURNED]) is TrackStatus.UNCERTAIN

    # Same track id before and after the occlusion, with the gap interpolated.
    child = tracks[CHILD]
    assert all(f in child.boxes for f in OCCLUDED)
    assert set(OCCLUDED) <= child.interpolated

    # Uncertain track is padded by pad_seconds (0.25 s @ 25 fps ≈ 6 frames).
    assert tracks[BACK_TURNED].first_frame == 4
    assert tracks[BACK_TURNED].last_frame == 36

    assert set(result.thumbnails) == {CHILD, ADULT, BACK_TURNED}
    assert analysis.video.frame_count == 50


def test_progress_is_reported(make_video):
    seen = []
    tracker = ScriptedTracker({})
    analyze_video(
        make_video(),
        tracker,
        HeadFaceDetector(tracker, {}),
        SizeAgeEstimator(1),
        progress=seen.append,
    )
    assert seen[-1] == 1.0
    assert seen == sorted(seen)


def test_plan_respects_threshold_overrides_and_manual_regions(make_video):
    analysis = _analyze(make_video()).analysis
    manual = ManualRegion("m1", 5, 7, Box(0, 0, 10, 10))

    plan = plan_render(analysis, DecisionPolicy(), {}, [manual])
    assert sorted(plan.blurred_tracks) == [CHILD, BACK_TURNED]
    assert plan.kept_tracks == [ADULT]
    assert Box(0, 0, 10, 10) in plan.boxes_for(6)
    assert Box(0, 0, 10, 10) not in plan.boxes_for(8)

    # Reviewer keeps the uncertain person and forces the adult to be blurred.
    plan = plan_render(
        analysis, DecisionPolicy(), {BACK_TURNED: Override.KEEP, ADULT: Override.BLUR}
    )
    assert sorted(plan.blurred_tracks) == [CHILD, ADULT]

    # Maximum sensitivity blurs everyone; minimum keeps everyone with evidence.
    assert len(plan_render(analysis, DecisionPolicy(threshold=0.0)).blurred_tracks) == 3
    assert plan_render(analysis, DecisionPolicy(threshold=1.0)).blurred_tracks == [BACK_TURNED]


def _channel_spread(region):
    region = region.astype(int)
    return np.abs(region - region.mean(axis=2, keepdims=True)).mean()


def test_render_blurs_minors_preserves_format_audio_and_strips_metadata(make_video, tmp_path):
    source = make_video(rate="30000/1001", metadata={"location": "+51.5074-000.1278/"})
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()))

    src_info, out_info = probe(source), probe(out)
    assert (out_info.width, out_info.height) == (src_info.width, src_info.height)
    assert out_info.fps_rational == "30000/1001"
    assert out_info.has_audio and out_info.audio_codec == "aac"

    tags = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-print_format", "json", str(out)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["format"].get("tags", {})
    assert "location" not in tags

    src_frames = dict(iter_frames(source))
    out_frames = dict(iter_frames(out))
    assert len(out_frames) == len(src_frames) == 60  # 2 s @ 29.97 fps

    x1, y1, x2, y2 = CHILD_BOX
    ax1, ay1, ax2, ay2 = ADULT_BOX
    for f in (0, 22, 49):  # 22 is mid-occlusion: interpolated, still blurred
        child = out_frames[f][y1 + 10 : y2 - 10, x1 + 10 : x2 - 10]
        assert _channel_spread(child) < 4, f"child not greyscale at frame {f}"
        adult_out = out_frames[f][ay1 + 10 : ay2 - 10, ax1 + 30 : ax2 - 10].astype(int)
        adult_src = src_frames[f][ay1 + 10 : ay2 - 10, ax1 + 30 : ax2 - 10].astype(int)
        assert np.abs(adult_out - adult_src).mean() < 6, f"adult altered at frame {f}"
        assert _channel_spread(adult_out) > 10


def test_non_mp4_audio_is_transcoded(make_video, tmp_path):
    source = make_video(name="in.mkv", audio_codec="pcm_s16le")
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()))
    assert probe(out).audio_codec == "aac"


def test_video_without_audio(make_video, tmp_path):
    source = make_video(audio=False)
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()))
    assert not probe(out).has_audio


def test_cancel_stops_render_and_removes_partial_output(make_video, tmp_path):
    source = make_video()
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()), cancel=cancel)
    assert not out.exists()
