# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /build
# requirements-cpu.txt pins the CPU wheel index, which keeps the image at
# ~1.5GB instead of ~6GB by skipping the nvidia-* CUDA wheels that plain
# `torch` pulls in as dependencies.
#
# On a CUDA host (DGX and friends) build with the full requirements instead:
#   docker build --build-arg REQUIREMENTS=requirements.txt ./grounding
# or `docker compose --profile grounding-gpu build`, which sets it for you.
# A CPU-only wheel on a GPU host is a SILENT failure - cuda.is_available()
# returns False and inference quietly runs on CPU - so pair the GPU build
# with GROUNDING_REQUIRE_DEVICE=cuda.
ARG REQUIREMENTS=requirements-cpu.txt
COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir --prefix=/install -r "$REQUIREMENTS"


FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libjpeg62-turbo zlib1g curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY app/ ./app/

# HF_HOME is a named volume in compose so the 933MB checkpoint survives a
# rebuild. Weights are downloaded on first start, not baked into the image.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models
USER appuser

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    OMP_NUM_THREADS=4
EXPOSE 8001

# Long start period: the checkpoint download plus load plus the warm-up
# forward pass. /readyz stays 503 for all of it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8001/readyz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
