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
    # The label this box was won by, verbatim from the cleaned request list -
    # NOT a canonical component name. Mapping it onto a vocabulary is the
    # caller's job; the sidecar deliberately knows nothing about one.
    label: str
    score: float
    box: Box
    # Index into GroundResponse.labels. Prefer it over `label`: it is exact,
    # where the string is only as good as the caller's matching.
    labelIndex: int | None = None
    # How far the winning label beat the runner-up. Small means the query was
    # genuinely torn between two labels.
    margin: float | None = None


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
        # "?" is a phrase delimiter to the model, so one inside a label splits
        # it into two spans and puts every label index after it out by one.
        if any("?" in label for label in labels):
            raise ValueError("labels must not contain '?'")
        return labels


class GroundResponse(BaseModel):
    detections: list[Detection]
    frame: tuple[int, int]
    # The cleaned, fitted label list that labelIndex points into.
    labels: list[str] = []
    meta: dict
