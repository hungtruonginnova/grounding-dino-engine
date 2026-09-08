"""Pure label/box handling - no torch, no model download."""

from app.boxes import clean_labels, normalise


def test_labels_are_lowercased_and_deduped_in_order():
    assert clean_labels(["Battery", "  battery ", "Air Filter Box"]) == [
        "battery",
        "air filter box",
    ]


def test_dots_are_stripped():
    """A label containing '.' makes the processor take a path that yields
    garbage, so it must never reach it."""
    assert clean_labels(["a.c. line"]) == ["a c line"]
    assert all("." not in label for label in clean_labels(["engine.", ".hood"]))


def test_empty_labels_are_dropped():
    assert clean_labels(["", "   ", "battery"]) == ["battery"]


def test_boxes_are_normalised_to_the_frame():
    detections = normalise([[128, 100, 256, 200]], [0.8], ["battery"], 1280, 1000)
    assert detections[0]["box"] == {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}
    assert detections[0]["score"] == 0.8


def test_out_of_frame_coordinates_are_clamped():
    """post_process_grounded_object_detection does not clamp, so x0 can be
    negative and x1 can exceed the width."""
    detections = normalise([[-50, -20, 1400, 1100]], [0.5], ["engine"], 1280, 1000)
    box = detections[0]["box"]
    assert box == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def test_degenerate_boxes_are_dropped():
    detections = normalise([[10, 10, 12, 400]], [0.9], ["battery"], 1280, 1000)
    assert detections == []


def test_empty_phrases_are_dropped():
    """A query attending to nothing above text_threshold decodes as ''."""
    detections = normalise([[100, 100, 300, 300]], [0.9], [""], 1280, 1000)
    assert detections == []


def test_detections_are_sorted_by_descending_score():
    detections = normalise(
        [[10, 10, 300, 300], [20, 20, 400, 400], [30, 30, 500, 500]],
        [0.3, 0.9, 0.6],
        ["a", "b", "c"],
        1280,
        1000,
    )
    assert [d["label"] for d in detections] == ["b", "c", "a"]
