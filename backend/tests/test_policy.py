import pytest

from safeframe.pipeline.policy import (
    DecisionPolicy,
    Override,
    aggregate,
    bucket_stats,
    minor_fraction,
    parse_age_bucket,
)
from safeframe.pipeline.types import AgeSample, Box, Track, TrackStatus


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("0-2", (0, 2)),
        ("10-19", (10, 19)),
        ("(15, 20)", (15, 20)),
        ("more than 70", (70, 100)),
        ("70+", (70, 100)),
    ],
)
def test_parse_age_bucket(label, expected):
    assert parse_age_bucket(label) == expected


def test_parse_age_bucket_rejects_garbage():
    with pytest.raises(ValueError):
        parse_age_bucket("adult")


def test_minor_fraction_splits_straddling_bucket():
    assert minor_fraction(0, 2) == 1.0
    assert minor_fraction(10, 19) == pytest.approx(0.8)  # ages 10..17 of 10..19
    assert minor_fraction(15, 20) == pytest.approx(3 / 6)
    assert minor_fraction(18, 25) == 0.0
    assert minor_fraction(20, 29) == 0.0


def test_bucket_stats_for_fairface_labels():
    labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
    fractions, mids = bucket_stats(labels)
    assert fractions == pytest.approx([1, 1, 0.8, 0, 0, 0, 0, 0, 0])
    assert mids[0] == 1.0 and mids[2] == 14.5


def _track(samples):
    t = Track(track_id=1, boxes={0: Box(0, 0, 10, 10)}, samples=samples)
    t.p_minor, t.expected_age = aggregate(samples)
    return t


def _s(p, w=1.0, age=10.0):
    return AgeSample(frame=0, p_minor=p, expected_age=age, weight=w)


def test_aggregate_is_weighted_mean():
    p, age = aggregate([_s(1.0, w=3, age=10), _s(0.0, w=1, age=30)])
    assert p == pytest.approx(0.75)
    assert age == pytest.approx(15)
    assert aggregate([]) == (None, None)


def test_too_few_samples_is_uncertain_and_blurred():
    policy = DecisionPolicy(threshold=0.35, min_face_samples=3)
    track = _track([_s(0.0), _s(0.0)])  # two confident "adult" samples are not enough
    assert policy.classify(track) is TrackStatus.UNCERTAIN
    assert policy.should_blur(track)


def test_threshold_splits_minor_and_adult():
    policy = DecisionPolicy(threshold=0.35, min_face_samples=2)
    assert policy.classify(_track([_s(0.4), _s(0.4)])) is TrackStatus.MINOR
    assert policy.classify(_track([_s(0.1), _s(0.1)])) is TrackStatus.ADULT
    assert not policy.should_blur(_track([_s(0.1), _s(0.1)]))


def test_overrides_win():
    policy = DecisionPolicy(min_face_samples=1)
    adult = _track([_s(0.0)])
    minor = _track([_s(1.0)])
    assert policy.should_blur(adult, Override.BLUR)
    assert not policy.should_blur(minor, Override.KEEP)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_policy_validates_threshold(bad):
    with pytest.raises(ValueError):
        DecisionPolicy(threshold=bad)
