from safeframe.pipeline.smoothing import fill_gaps, pad_ends, smooth_track
from safeframe.pipeline.types import (
    AgeSample,
    Analysis,
    Box,
    ManualRegion,
    Track,
    VideoInfo,
)


def test_short_gap_is_interpolated():
    t = Track(1, boxes={0: Box(0, 0, 10, 10), 4: Box(40, 0, 50, 10)})
    fill_gaps(t, max_gap=5)
    assert sorted(t.boxes) == [0, 1, 2, 3, 4]
    assert t.boxes[2] == Box(20, 0, 30, 10)
    assert t.interpolated == {1, 2, 3}


def test_long_gap_is_left_alone():
    t = Track(1, boxes={0: Box(0, 0, 10, 10), 10: Box(0, 0, 10, 10)})
    fill_gaps(t, max_gap=5)
    assert sorted(t.boxes) == [0, 10]


def test_pad_ends_clips_to_video():
    t = Track(1, boxes={1: Box(0, 0, 1, 1), 8: Box(5, 5, 6, 6)})
    pad_ends(t, pad=3, frame_count=10)
    assert t.first_frame == 0
    assert t.last_frame == 9
    assert t.boxes[9] == Box(5, 5, 6, 6)


def test_smooth_track_combines_both():
    t = Track(1, boxes={5: Box(0, 0, 1, 1), 7: Box(2, 2, 3, 3)})
    smooth_track(t, max_gap=2, pad=1, frame_count=100)
    assert sorted(t.boxes) == [4, 5, 6, 7, 8]


def test_analysis_round_trip():
    info = VideoInfo(640, 360, 29.97, "30000/1001", 100, 3.3, True, "aac")
    track = Track(
        3,
        boxes={0: Box(1, 2, 3, 4), 1: Box(2, 3, 4, 5)},
        interpolated={1},
        samples=[AgeSample(0, 0.7, 12.0, 0.5)],
        p_minor=0.7,
        expected_age=12.0,
    )
    analysis = Analysis(video=info, tracks=[track], settings={"age_every": 3})
    restored = Analysis.from_dict(analysis.to_dict())
    assert restored.video == info
    t = restored.tracks[0]
    assert t.boxes == track.boxes
    assert t.interpolated == {1}
    assert t.samples == track.samples
    assert t.p_minor == 0.7


def test_manual_region_round_trip_and_cover():
    r = ManualRegion("a", 5, 9, Box(0, 0, 10, 10))
    assert ManualRegion.from_dict(r.to_dict()) == r
    assert r.covers(5) and r.covers(9) and not r.covers(10)


def test_box_geometry():
    b = Box(10, 10, 30, 50)
    assert b.center == (20, 30)
    assert b.expand(0.5) == Box(0, -10, 40, 70)
    assert b.expand(0.5).clip(35, 60) == Box(0, 0, 35, 60)
    assert Box(5, 5, 1, 1).area == 0
