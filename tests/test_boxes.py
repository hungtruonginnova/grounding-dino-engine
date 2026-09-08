"""Pure label/box handling - no torch, no model download."""

from app.boxes import clean_labels, label_spans, normalise


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


def test_label_spans_finds_one_span_per_label():
    """'[CLS] car battery . black plastic fuse box lid . [SEP] [PAD]'"""
    ids = [101, 2482, 6046, 1012, 2304, 4145, 24856, 3482, 11876, 1012, 102, 0]
    assert label_spans(ids) == [(1, 3), (4, 9)]


def test_label_spans_ignores_trailing_padding():
    assert label_spans([101, 6046, 1012, 102, 0, 0, 0]) == [(1, 2)]


def test_label_spans_handles_a_prompt_with_no_final_delimiter():
    assert label_spans([101, 6046]) == [(1, 2)]


def test_question_marks_are_stripped_so_indices_cannot_shift():
    """'?' is a phrase delimiter to the model: left in, one label would occupy
    two spans and every index after it would be out by one."""
    labels = clean_labels(["is this a hose?", "battery"])
    assert labels == ["is this a hose", "battery"]
    assert all("?" not in label for label in labels)


def test_attribution_survives_a_dropped_noise_box():
    """A box culled by MIN_BOX_SIDE must not shift the labels of the rest."""
    detections = normalise(
        [[10, 10, 12, 400], [128, 100, 256, 200]],
        [0.9, 0.8],
        ["battery", "air filter box"],
        1280,
        1000,
        indices=[0, 1],
        margins=[0.5, 0.4],
    )
    assert [(d["label"], d["labelIndex"]) for d in detections] == [("air filter box", 1)]


def test_attribution_is_absent_when_not_supplied():
    """The legacy phrase path stays byte-compatible for an older caller."""
    detections = normalise([[128, 100, 256, 200]], [0.8], ["battery"], 1280, 1000)
    assert "labelIndex" not in detections[0]
