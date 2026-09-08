"""Pure label and box handling. Deliberately free of torch, so it tests
without a 2.5GB dependency and without a model download."""

# Boxes thinner than this fraction of the frame in either dimension are
# decoding noise, not parts.
MIN_BOX_SIDE = 0.01


def clean_labels(labels: list[str]) -> list[str]:
    """Lowercase, strip, drop dots, dedupe, preserve order.

    The processor lowercases and dot-joins the prompt itself; all we owe it is
    clean, deduped, dot-free phrases. Dots matter because
    GroundingDinoProcessor._is_list_of_candidate_labels rejects any element
    containing one, and the whole list then takes a path that yields garbage.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for label in labels:
        name = " ".join(label.strip().lower().replace(".", " ").split())
        if name and name not in seen:
            seen.add(name)
            clean.append(name)
    return clean


def normalise(boxes, scores, labels, width: int, height: int) -> list[dict]:
    """Absolute xyxy pixels to normalised xywh, clamped, noise dropped.

    post_process_grounded_object_detection does not clamp - it just scales
    normalised centre-format coordinates by the target size, so x0 can be
    negative and x1 can exceed the width.
    """
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x0, y0, x1, y1 = box
        nx0 = min(max(x0 / width, 0.0), 1.0)
        ny0 = min(max(y0 / height, 0.0), 1.0)
        nx1 = min(max(x1 / width, 0.0), 1.0)
        ny1 = min(max(y1 / height, 0.0), 1.0)
        w, h = nx1 - nx0, ny1 - ny0
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            continue
        phrase = (label or "").strip()
        if not phrase:
            continue
        detections.append(
            {
                "label": phrase,
                "score": round(float(score), 4),
                "box": {
                    "x": round(nx0, 5),
                    "y": round(ny0, 5),
                    "w": round(w, 5),
                    "h": round(h, 5),
                },
            }
        )
    detections.sort(key=lambda d: -d["score"])
    return detections
