import numpy as np

from safeframe.pipeline.anonymize import anonymize_regions, gaussian_blur_strong, pixel_rect
from safeframe.pipeline.faces import assign_faces, crop_face
from safeframe.pipeline.types import Box, FaceDetection

from .conftest import person


def _noise(h=200, w=300, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_region_is_blurred_and_greyscale_outside_untouched():
    frame = _noise()
    original = frame.copy()
    anonymize_regions(frame, [Box(50, 40, 150, 160)], padding=0.0)

    region = frame[40:160, 50:150].astype(int)
    assert np.array_equal(region[..., 0], region[..., 1])
    assert np.array_equal(region[..., 1], region[..., 2])
    # High-frequency noise should be almost entirely gone.
    assert region[..., 0].std() < original[40:160, 50:150].std() / 5

    mask = np.ones(frame.shape[:2], bool)
    mask[40:160, 50:150] = False
    assert np.array_equal(frame[mask], original[mask])


def test_padding_grows_region_and_clips_to_frame():
    assert pixel_rect(Box(0, 0, 100, 100), 300, 200, padding=0.1) == (0, 0, 110, 110)
    assert pixel_rect(Box(290, 190, 310, 210), 300, 200, padding=0.0) == (290, 190, 300, 200)
    assert pixel_rect(Box(400, 400, 500, 500), 300, 200) is None


def test_edge_and_tiny_regions_do_not_crash():
    frame = _noise()
    anonymize_regions(frame, [Box(-50, -50, 10, 10), Box(299, 199, 300, 200), Box(5, 5, 5, 5)])


def test_blur_strength_scales_with_region_size():
    big = gaussian_blur_strong(_noise(900, 400, seed=1))
    assert big.std() < 10


def test_face_goes_to_smallest_containing_person():
    adult = person(1, 0, 0, 200, 400)
    child = person(2, 50, 100, 150, 300)  # held by the adult, inside their box
    face = FaceDetection(Box(85, 110, 115, 140), 0.9)
    assert assign_faces([face], [adult, child]) == {2: face}


def test_face_of_child_passing_in_front_is_not_given_to_person_behind():
    # Regression: a smaller box further back used to win on area alone.
    behind = person(3, 300, 90, 370, 300)
    child = person(1, 290, 120, 380, 340)
    face = FaceDetection(Box(318, 122, 352, 156), 0.9)
    assert assign_faces([face], [behind, child]) == {1: face}


def test_ambiguous_face_is_dropped_rather_than_guessed():
    a = person(1, 100, 100, 200, 400)
    b = person(2, 102, 101, 202, 401)  # near-identical boxes
    face = FaceDetection(Box(135, 120, 165, 150), 0.9)
    assert assign_faces([face], [a, b]) == {}


def test_face_in_lower_body_is_ignored():
    p = person(1, 0, 0, 100, 300)
    hand_level = FaceDetection(Box(40, 250, 60, 270), 0.9)
    assert assign_faces([hand_level], [p]) == {}


def test_best_face_per_person():
    p = person(1, 0, 0, 100, 300)
    weak = FaceDetection(Box(10, 10, 30, 30), 0.6)
    strong = FaceDetection(Box(40, 10, 60, 30), 0.95)
    assert assign_faces([weak, strong], [p]) == {1: strong}


def test_crop_face_is_square_with_margin():
    crop = crop_face(_noise(), Box(100, 50, 140, 90), margin=0.25)
    assert crop is not None and crop.shape[:2] == (60, 60)
