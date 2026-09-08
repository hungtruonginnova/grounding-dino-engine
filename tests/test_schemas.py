"""Request validation. The dot rule is the one that matters."""

import pytest
from pydantic import ValidationError

from app.schemas import MAX_IMAGE_B64_CHARS, GroundRequest


def test_valid_request():
    req = GroundRequest(imageBase64="Zm9v", labels=["battery", "air filter box"])
    assert req.boxThreshold is None  # falls back to the service default


def test_labels_containing_a_dot_are_rejected():
    """GroundingDinoProcessor._is_list_of_candidate_labels rejects any element
    containing '.', and the whole list then produces garbage. Fail loudly."""
    with pytest.raises(ValidationError, match="must not contain"):
        GroundRequest(imageBase64="Zm9v", labels=["a.c. line"])


def test_empty_label_list_is_rejected():
    with pytest.raises(ValidationError):
        GroundRequest(imageBase64="Zm9v", labels=[])


def test_oversized_payload_is_rejected():
    """Stops anything on the internal network using this as a memory bomb."""
    with pytest.raises(ValidationError):
        GroundRequest(imageBase64="A" * (MAX_IMAGE_B64_CHARS + 1), labels=["battery"])


def test_thresholds_must_be_probabilities():
    with pytest.raises(ValidationError):
        GroundRequest(imageBase64="Zm9v", labels=["battery"], boxThreshold=1.5)
