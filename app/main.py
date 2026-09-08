"""GroundingDINO sidecar: a pure inference box.

Holds no credentials and knows nothing about the component vocabulary - it
takes an image plus free-text label phrases and returns normalised boxes.
"""

import logging
import os
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from .detector import Detector, torch_build
from .schemas import GroundRequest, GroundResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("grounding")

BOX_THRESHOLD = float(os.getenv("GROUNDING_BOX_THRESHOLD", "0.3"))
TEXT_THRESHOLD = float(os.getenv("GROUNDING_TEXT_THRESHOLD", "0.25"))
# One device, one model instance. Two concurrent forwards on MPS mostly just
# double the latency while risking OOM.
MAX_CONCURRENCY = int(os.getenv("GROUNDING_MAX_CONCURRENCY", "1"))
QUEUE_WAIT_SECONDS = float(os.getenv("GROUNDING_QUEUE_WAIT_SECONDS", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = Detector()
    app.state.detector = detector
    app.state.slots = anyio.Semaphore(MAX_CONCURRENCY)
    await run_in_threadpool(detector.load)
    await run_in_threadpool(detector.warmup)
    logger.info("ready: device=%s model=%s", detector.device, detector.model_id)
    yield


app = FastAPI(
    title="Innova Grounding Sidecar",
    version="1.0.0",
    description="Zero-shot bounding boxes for text-named objects.",
    lifespan=lifespan,
)


@app.get("/healthz", summary="Liveness probe")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz", summary="Readiness probe (false until warm-up completes)")
async def readyz():
    detector = app.state.detector
    if not detector.ready:
        raise HTTPException(status_code=503, detail="model still loading")
    return {
        "status": "ready",
        "device": detector.device,
        "model": detector.model_id,
        "maxTextTokens": detector.max_text_tokens,
        # Included so a CPU-only wheel on a GPU host is visible without
        # reading the container logs.
        "torch": torch_build(),
    }


@app.post("/ground", response_model=GroundResponse, summary="Ground label phrases")
async def ground(req: GroundRequest):
    detector = app.state.detector
    if not detector.ready:
        raise HTTPException(status_code=503, detail="model still loading")

    # A bounded wait, not an unbounded queue: without it requests pile up
    # until the caller's timeout fires and we have burned device time on work
    # nobody is waiting for any more. A fast 503 lets the caller move on.
    try:
        with anyio.fail_after(QUEUE_WAIT_SECONDS):
            await app.state.slots.acquire()
    except TimeoutError:
        raise HTTPException(status_code=503, detail="grounding busy")

    try:
        return await run_in_threadpool(
            detector.infer,
            req.imageBase64,
            req.labels,
            req.boxThreshold if req.boxThreshold is not None else BOX_THRESHOLD,
            req.textThreshold if req.textThreshold is not None else TEXT_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc
    finally:
        app.state.slots.release()
