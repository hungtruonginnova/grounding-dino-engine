"""Pure label and box handling. Deliberately free of torch, so it tests
without a 2.5GB dependency and without a model download."""

# Boxes thinner than this fraction of the frame in either dimension are
# decoding noise, not parts.
MIN_BOX_SIDE = 0.01


# The token ids GroundingDINO itself treats as phrase boundaries when it
# builds its block-diagonal text attention mask: [CLS], [SEP], "." and "?",
# plus [PAD]. Segmenting on the same set is what makes our label spans mean
# the same thing the model means by "a phrase".
DELIMITER_IDS = frozenset({101, 102, 1012, 1029, 0})


def clean_labels(labels: list[str]) -> list[str]:
    """Lowercase, strip, drop delimiters, dedupe, preserve order.

    The processor lowercases and dot-joins the prompt itself; all we owe it is
    clean, deduped, delimiter-free phrases. Dots matter because
    GroundingDinoProcessor._is_list_of_candidate_labels rejects any element
    containing one, and the whole list then takes a path that yields garbage.
    Question marks matter for a different reason: they are a phrase boundary
    to the model, so one inside a label would split it into two spans and put
    every label index after it out by one.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for label in labels:
        name = " ".join(label.strip().lower().replace(".", " ").replace("?", " ").split())
        if name and name not in seen:
            seen.add(name)
            clean.append(name)
    return clean


def label_spans(input_ids: list[int]) -> list[tuple[int, int]]:
    """The [start, end) token range of each label in the merged prompt.

    The prompt is ". ".join(labels) + ".", so the labels are exactly the runs
    of non-delimiter tokens. Attributing a box by which span its hottest text
    token falls in is what replaces decoding a phrase out of the position map
    - that decode is lossy by construction, and is where boxes were getting
    attached to the wrong component.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, token in enumerate(input_ids):
        if token in DELIMITER_IDS:
            if start is not None:
                spans.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append((start, len(input_ids)))
    return spans


def normalise(boxes, scores, labels, width: int, height: int, indices=None, margins=None) -> list[dict]:
    """Absolute xyxy pixels to normalised xywh, clamped, noise dropped.

    The coordinates arrive unclamped - they are normalised centre-format
    coordinates scaled by the target size, so x0 can be negative and x1 can
    exceed the width.

    `indices` and `margins` carry span attribution: which label the box was
    won by, and by how much it beat the runner-up. A caller that has them
    never has to guess a component from the label text.
    """
    indices = indices if indices is not None else [None] * len(scores)
    margins = margins if margins is not None else [None] * len(scores)
    detections = []
    for box, score, label, index, margin in zip(boxes, scores, labels, indices, margins):
        x0, y0, x1, y1 = box
        nx0 = min(max(x0 / width, 0.0), 1.0)
        ny0 = min(max(y0 / height, 0.0), 1.0)
        nx1 = min(max(x1 / width, 0.0), 1.0)
        ny1 = min(max(y1 / height, 0.0), 1.0)
        w, h = nx1 - nx0, ny1 - ny0
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            continue
        phrase = (label or "").strip()
        # An empty label is only possible on the legacy phrase-decoding path,
        # where it means "no text token cleared the threshold". With span
        # attribution there is always a winning label.
        if not phrase and index is None:
            continue
        detection = {
            "label": phrase,
            "score": round(float(score), 4),
            "box": {
                "x": round(nx0, 5),
                "y": round(ny0, 5),
                "w": round(w, 5),
                "h": round(h, 5),
            },
        }
        if index is not None:
            detection["labelIndex"] = int(index)
            detection["margin"] = round(float(margin), 4) if margin is not None else None
        detections.append(detection)
    detections.sort(key=lambda d: -d["score"])
    return detections
