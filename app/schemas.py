"""Request/response models for the grounding sidecar."""

from pydantic import BaseModel, Field, field_validator

# A 1280px JPEG at quality 85 base64-encodes to roughly 300KB. 8MB is a
# generous ceiling that still stops anything on the internal network from
# using this service as a memory bomb.
MAX_IMAGE_B64_CHARS = 8_000_000


class Box(BaseModel):
    """Normalised to the frame of the submitted image, clamped to [0, 1]."""

    x: float
    y: float
    w: float
    h: float


class Detection(BaseModel):
    # The raw phrase GroundingDINO decoded, NOT a canonical component name.
    # Mapping it back onto the vocabulary is the caller's job - the sidecar
    # deliberately knows nothing about the component vocabulary.
    label: str
    score: float
    box: Box


class GroundRequest(BaseModel):
    imageBase64: str = Field(..., max_length=MAX_IMAGE_B64_CHARS)
    labels: list[str] = Field(..., min_length=1, max_length=100)
    boxThreshold: float | None = Field(default=None, ge=0.0, le=1.0)
    textThreshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("labels")
    @classmethod
    def _no_dots(cls, labels: list[str]) -> list[str]:
        # GroundingDinoProcessor._is_list_of_candidate_labels rejects any
        # element containing ".", and the whole list then falls through the
        # candidate-label path unmerged, producing garbage. Fail loudly here
        # rather than silently returning nonsense boxes.
        if any("." in label for label in labels):
            raise ValueError("labels must not contain '.'")
        return labels


class GroundResponse(BaseModel):
    detections: list[Detection]
    frame: tuple[int, int]
    meta: dict
