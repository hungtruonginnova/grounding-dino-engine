"""GroundingDINO wrapper: bytes + label phrases in, normalised boxes out."""

import base64
import io
import logging
import os
import time

# torch.roll in Swin's shifted-window attention has no MPS kernel on some
# torch builds. This must be set BEFORE torch is imported to take effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from .boxes import clean_labels, normalise  # noqa: E402

logger = logging.getLogger("grounding")

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


def torch_build() -> dict:
    """What this torch install can actually do.

    Reported by /readyz because the failure that matters here is silent: a
    CPU-only wheel on a GPU host makes cuda.is_available() False, pick_device
    falls back to cpu, and everything keeps working at a fraction of the
    speed with nothing in the logs to say why.
    """
    return {
        "torch": torch.__version__,
        # None on a CPU-only build, a version string on a CUDA build.
        "cudaBuild": torch.version.cuda,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDevices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def pick_device() -> str:
    """cuda > mps > cpu. GROUNDING_DEVICE overrides, for A/B sanity checks."""
    forced = os.getenv("GROUNDING_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class DeviceUnavailable(RuntimeError):
    """The device the deployment requires is not usable."""


def require_device(device: str) -> None:
    """Refuse to start on the wrong device when the deployment demands one.

    Set GROUNDING_REQUIRE_DEVICE=cuda on a GPU host. Without it, a CPU-only
    image or a missing --gpus flag degrades to CPU inference silently, which
    is far harder to notice than a container that will not start.
    """
    required = os.getenv("GROUNDING_REQUIRE_DEVICE", "").strip().lower()
    if not required or required == device:
        return
    build = torch_build()
    raise DeviceUnavailable(
        f"GROUNDING_REQUIRE_DEVICE={required} but the usable device is "
        f"{device!r}. torch={build['torch']} cudaBuild={build['cudaBuild']} "
        f"cudaAvailable={build['cudaAvailable']}. "
        "A None cudaBuild means this image was built with the CPU-only wheel "
        "(build with REQUIREMENTS=requirements.txt); a CUDA build with "
        "cudaAvailable=False means the container was started without GPU "
        "access (nvidia-container-toolkit, and --gpus all or a compose "
        "device reservation)."
    )


class Detector:
    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or os.getenv("GROUNDING_MODEL_ID", DEFAULT_MODEL_ID)
        self.device = device or pick_device()
        self.processor = None
        self.model = None
        self.max_text_tokens = 254
        self.ready = False

    def load(self) -> None:
        require_device(self.device)

        if self.device == "cpu":
            # torch otherwise grabs every core and starves whatever else is
            # on the box.
            torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))

        logger.info(
            "loading %s onto %s (%s)", self.model_id, self.device, torch_build()
        )
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # fp32 everywhere. GroundingDINO fills attention masks with -inf
        # (modeling_grounding_dino.py), which is a NaN factory in fp16 - and
        # MPS has no float64 at all, so fp32 is the only safe choice.
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        # config.max_text_len is 256; reserve [CLS] and [SEP].
        self.max_text_tokens = int(getattr(self.model.config, "max_text_len", 256)) - 2
        if self.device == "cuda":
            # One line that makes "is it actually on the GPU?" answerable
            # from the logs alone.
            logger.info("cuda device: %s", torch.cuda.get_device_name(0))
        logger.info(
            "loaded in %.1fs (max_text_tokens=%d)",
            time.perf_counter() - t0,
            self.max_text_tokens,
        )

    def warmup(self) -> None:
        """One real forward pass.

        The first MPS inference spends seconds compiling Metal kernels, and a
        broken install should surface at startup rather than on the first user
        request. /readyz stays false until this completes.
        """
        t0 = time.perf_counter()
        dummy = Image.new("RGB", (64, 64), (128, 128, 128))
        self._forward(dummy, ["object"], 0.9, 0.9)
        self.ready = True
        logger.info("warm-up forward pass took %.1fs", time.perf_counter() - t0)

    def _forward(self, image, labels, box_threshold, text_threshold):
        inputs = self.processor(images=image, text=[labels], return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            # Plain Python ints: MPS cannot hold float64, so never hand this a
            # numpy array or a tensor built from one.
            target_sizes=[(image.height, image.width)],
        )[0]

    def _fit_labels(self, labels: list[str]) -> tuple[list[str], list[str]]:
        """Trim labels until the merged prompt fits the text encoder.

        Overflow is silent upstream: the model slices input_ids to
        max_text_len without warning while post_process_* still indexes the
        untruncated ids, so labels past the cut just never appear. Callers
        send labels most-confident first, so dropping from the tail sacrifices
        the least.
        """
        kept = list(labels)
        dropped: list[str] = []
        while len(kept) > 1 and self._count_tokens(kept) > self.max_text_tokens:
            dropped.append(kept.pop())
        return kept, dropped

    def _count_tokens(self, labels: list[str]) -> int:
        merged = ". ".join(labels) + "."
        return len(self.processor.tokenizer(merged, add_special_tokens=True).input_ids)

    def infer(
        self,
        image_b64: str,
        labels: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> dict:
        raw = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw))
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size

        clean = clean_labels(labels)
        if not clean:
            return {
                "detections": [],
                "frame": (width, height),
                "meta": self._meta(0, [], 0.0, 0),
            }

        clean, truncated = self._fit_labels(clean)
        text_tokens = self._count_tokens(clean)

        t0 = time.perf_counter()
        result = self._forward(image, clean, box_threshold, text_threshold)
        inference_ms = round((time.perf_counter() - t0) * 1000)

        # text_labels, not labels: reading result["labels"] raises a
        # FutureWarning about returning integer ids in a later release.
        detections = normalise(
            result["boxes"].tolist(),
            result["scores"].tolist(),
            result["text_labels"],
            width,
            height,
        )
        return {
            "detections": detections,
            "frame": (width, height),
            "meta": self._meta(text_tokens, truncated, inference_ms, len(clean)),
        }

    def _meta(self, text_tokens, truncated, inference_ms, label_count) -> dict:
        return {
            "device": self.device,
            "model": self.model_id,
            "textTokens": text_tokens,
            "truncatedLabels": truncated,
            "labelCount": label_count,
            "inferenceMs": inference_ms,
        }
