# Phase 2: Lightning AI Studio GPU staging

This guide uses the existing ShriVastra modular FastAPI backend on Lightning AI Studios.
**Do not merge to main.** Studio deployments are for private staging and do not represent
a hardened multi-tenant production service.

Official docs: https://lightning.ai/docs/platform/overview/getting-started
and https://lightning.ai/docs/platform/build/ai-studio/deploy-on-public-ports/

## 1. Create / open a Studio and select GPU

1. Visit https://studio.lightning.ai and open a private Studio.
2. Use the environment/machine selector to **Switch to GPU**. Start with a
   compatible NVIDIA 24 GB GPU if one is available; use 40 GB or 80 GB if actual
   model loading/inference shows insufficient VRAM.
3. In the Studio terminal run `nvidia-smi`. GPU availability must be confirmed
   before installing or running the VTON model.
4. A Studio preserves home/project files across machine changes, but Docker images
   may not persist. For the first test, prefer a Python venv on the GPU Studio.

## 2. Checkout the reviewed development branch

```bash
git clone https://github.com/datavskl/fashn-vton-1.5.git
cd fashn-vton-1.5
git fetch origin
git checkout refactor/shrivastra-single-backend
```

Do not run destructive Git commands on `main`.

## 3. Set up the Python GPU environment

Use Python 3.11 if provided by the Studio. Create a clean venv, then install
dependencies. Confirm the Studio's preinstalled CUDA/PyTorch environment or install
a suitable PyTorch CUDA wheel according to the official compatibility matrix.
Avoid blindly installing a CPU-only wheel.

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The check must show CUDA availability. FASHN VTON depends on compatible PyTorch,
ONNX Runtime, CUDA, and NVIDIA driver versions.

Download weights (several gigabytes) onto persistent Studio storage:

```bash
python scripts/download_weights.py --weights-dir ./weights
```

## 4. Configure server secrets and upload limits

Create a private `.env` (already gitignored) via an interactive, private editor,
or via Studio's supported secrets/env variable facilities. Example values:

```dotenv
VTON_API_TOKEN=REPLACE_WITH_LONG_RANDOM_SERVER_SECRET
VTON_CORS_ORIGINS=
VTON_MAX_UPLOAD_BYTES=10485760
VTON_MAX_IMAGE_PIXELS=20000000
VTON_QUEUE_SIZE=4
VTON_JOB_TTL_SECONDS=3600
VTON_WEIGHTS_DIR=./weights
```

To generate a token: `python -c 'import secrets; print(secrets.token_urlsafe(48))'`.
Never commit a real token or expose it in the browser, Studio share link, WordPress
JavaScript or logs. The FastAPI process reads exported environment variables; a
file named `.env` is **not automatically loaded** by `uvicorn`. Either export
variables via Studio environment settings, or start with `uvicorn --env-file .env`
(after confirming `python-dotenv` is installed: `pip install python-dotenv`).

For local-only validation, export the token from a secure private environment variable.
Keep `VTON_CORS_ORIGINS` empty when no browser needs direct access: WordPress
should connect server-to-server, so its browser does not need access to GPU API.

## 5. Start local-only FastAPI

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second Studio terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
python -m pytest -q tests/test_backend_api.py
```

The health endpoint reports GPU visibility but **does not** load the model or prove
inference works. Start one server process only. The job queue is in-memory.

## 6. Run real-image GPU acceptance tests

Use only images you have permission to process. Keep test input and outputs private.
The test script creates actual inference jobs with no model mocking.

```bash
export VTON_API_TOKEN='THE_SAME_SECRET_USED_BY_YOUR_RUNNING_API'
python scripts/gpu_smoke_test.py \
  --api-url http://127.0.0.1:8000 \
  --person /private/path/person.jpg \
  --top /private/path/top.jpg \
  --bottom /private/path/bottom.jpg \
  --timeout 600
```

Three cases should complete: top-only, bottom-only, combined top+bottom.
Review generated images for correct clothing, face identity and pose, log cold and
warm latency and GPU memory. Images/results are written to
`test_outputs/gpu-smoke/`, which is ignored by Git.

Manually test negative cases: missing bearer token (401), invalid uploads (4xx),
over-limit images (413), invalid parameters (422), queue pressure (429) and
restart behavior (jobs are not persistent). Verify that no real images or tokens
leak into logs or version control.

## 7. API exposure and WordPress integration (only after private GPU tests)

Lightning AI has a Studio public-port / API-builder workflow. Use the current
Lightning platform UI/docs for endpoint configuration; do not guess a public URL.
An externally reachable endpoint **must use HTTPS and additional ingress
restrictions / rate limits**, and restrict accepted request sizes. A shared API
token alone is not sufficient for an openly advertised inference service.

For first staging tests, prefer Studio terminal / SSH port forwarding. If using
the Lightning AI API builder/public ports feature later, configure TLS and verify
that it preserves `Authorization` headers and supports multipart images.
The WordPress bridge must send the bearer token server-to-server; it must not
include the secret in client code. Do not enable public anonymous GPU inference.

## 8. Success criteria and limitations

- Real single-top, single-bottom, combined tests passed with inspected PNG outputs
- Peak VRAM and response latency captured for chosen GPU
- HTTPS/ingress authentication and body-size limits independently verified before
  opening network access
- Model weights persist across Studio restart; test reloading
- User-controlled Studio spending / GPU usage reviewed

The current implementation lacks durable jobs, GPU timeouts, customer isolation,
persistent result storage, scheduled cleanup, and durable deployment across
Studio sleep/restarts. Add these before production rollout.
