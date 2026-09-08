# Grounding engine

Standalone GroundingDINO service. Takes an image plus free-text label
phrases, returns bounding boxes normalised to the submitted frame, over a
plain HTTP API.

Deliberately dumb: no S3 credentials, no component vocabulary. This is a
separate deployable service - callers (e.g. the
[innova-automotive-detection](https://gitlab.innovavietnam.com/fullstack-dev/ai-project/innova-automotive-detection)
API) reach it over the network via `GROUNDING_URL` and own the job of mapping
loose DINO phrases back onto their own canonical names.

## Run natively (required for MPS on Apple Silicon)

Docker on macOS cannot reach the Metal device, so MPS means running on the
host:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --port 8001
```

Then point the caller at it with `GROUNDING_URL=http://host.docker.internal:8001`
(caller in Docker) or `http://127.0.0.1:8001` (caller on the host too).

## Run in Docker (CPU)

```bash
docker build -t innova-grounding .
docker run --rm -p 8001:8001 -v hf-cache:/models innova-grounding
```

## Deploy on a GPU host (DGX Spark and friends)

Requires `nvidia-container-toolkit` on the host:

```bash
docker compose up --build
```

This builds with `REQUIREMENTS=requirements.txt` (the CUDA wheel) and sets
`GROUNDING_REQUIRE_DEVICE=cuda`, so the service refuses to start rather than
silently falling back to CPU. A CPU-only wheel on a GPU host is otherwise a
silent failure - `cuda.is_available()` returns `False` and inference quietly
runs on CPU with nothing in the logs to say why.

## Environment

| Var | Default | Notes |
|---|---|---|
| `GROUNDING_MODEL_ID` | `IDEA-Research/grounding-dino-base` | `-tiny` is 689MB vs 933MB, lower recall on small parts |
| `GROUNDING_DEVICE` | auto (`cuda` > `mps` > `cpu`) | Force it to A/B MPS against CPU |
| `GROUNDING_REQUIRE_DEVICE` | unset | Set to `cuda` to refuse to start instead of falling back |
| `GROUNDING_BOX_THRESHOLD` | `0.3` | Per-request override available |
| `GROUNDING_TEXT_THRESHOLD` | `0.25` | Per-request override available |
| `GROUNDING_MAX_CONCURRENCY` | `1` | One device, one model instance |
| `GROUNDING_QUEUE_WAIT_SECONDS` | `30` | Past this, return 503 rather than queue |
| `OMP_NUM_THREADS` | `4` | CPU only; torch otherwise takes every core |

`/readyz` returns 503 until a real warm-up forward pass has completed - the
first MPS inference spends seconds compiling Metal kernels.

## API

- `GET /healthz` - liveness probe.
- `GET /readyz` - readiness probe; 503 until warm-up completes, then reports
  `device`, `model`, `maxTextTokens`, `torch`.
- `POST /ground` - body `{"imageBase64": ..., "labels": [...], "boxThreshold"?: ..., "textThreshold"?: ...}`,
  returns normalised detections.

## Tests

```bash
./.venv/bin/pytest
```
