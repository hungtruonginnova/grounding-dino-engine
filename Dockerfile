# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /build
# This image is CUDA-only: it's built and deployed on GPU hosts (DGX Spark
# and friends), so it always installs the full requirements.txt (CUDA
# wheels). Pair it with GROUNDING_REQUIRE_DEVICE=cuda so the container
# refuses to start rather than silently falling back to CPU if it ever ends
# up on a host without GPU access.
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


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
