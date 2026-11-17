# ShriVastra FASHN VTON v1.5 — Single modular FastAPI backend

Entry point: `app/main.py` (`uvicorn app.main:app`). The original FASHN AI inference model remains in `src/fashn_vton/`. The Gradio `app.py` is a separate optional demo, not a production backend.


This backend is intended for a **single GPU and one Uvicorn worker**.
GPU inference has not yet been verified on a target host. It requires a CUDA-capable NVIDIA GPU.

## Install

1. Use Python 3.10–3.12, preferably 3.11, and a GPU-compatible PyTorch install.
2. Create a virtual environment and install the model package: `pip install -e .`
3. Install backend dependencies: `pip install -r requirements.txt`
4. Download model weights: `python scripts/download_weights.py --weights-dir ./weights`
5. Validate CUDA installation using `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| VTON_API_TOKEN | Unset | Required shared server-side bearer token; unset fails closed |
| VTON_CORS_ORIGINS | Empty | Comma-separated exact trusted browser origins |
| VTON_MAX_UPLOAD_BYTES | 10485760 | Per-image file limit (10 MiB) |
| VTON_MAX_IMAGE_PIXELS | 20000000 | Maximum decoded image pixels |
| VTON_QUEUE_SIZE | 20 | Maximum queued jobs |
| VTON_JOB_TTL_SECONDS | 86400 | Completed job memory retention |
| VTON_WEIGHTS_DIR | ./weights | Model weights directory |

Generate a token using `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
Store the token in a protected environment or secret manager, not in the repository.

Example: `export VTON_CORS_ORIGINS="https://your-store.example"`.

Run: `uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1`.

Deploy behind an HTTPS reverse proxy (such as Nginx). Configure a request body cap (e.g.
`client_max_body_size 25m`), network firewall, authentication on upstream applications,
and per-IP rate limiting. FastAPI's per-image check happens after multipart parsing:
**the reverse proxy body limit remains essential**.

**Never expose VTON_API_TOKEN in public WordPress JavaScript or mobile clients.**
Proxy requests through a trusted backend that authenticates the end-user.

## Endpoints

- GET /health: basic process health, GPU availability and model-loading state.
- POST /api/v1/virtual-tryon: authenticated synchronous single-garment API (legacy-compatible).
- POST /api/v1/virtual-tryon-separate: authenticated synchronous outfit API (legacy-compatible).
- POST /api/v1/jobs: authenticated asynchronous submission, returns HTTP 202 with a job ID.
- GET /api/v1/jobs/{job_id}: authenticated polling, states queued / processing / completed / failed.

All inference endpoints require the header `Authorization: Bearer YOUR_TOKEN`.
Uploads use multipart form fields `person_image`, `garment_image`,
`upper_garment_image` and `lower_garment_image` (choose single or separate garment mode).
Supported image types: JPEG, PNG and WebP.
Parameters: category tops/bottoms/one-pieces, garment_photo_type model/flat-lay,
num_timesteps 10–50, guidance_scale 1–3, seed >=0, segmentation_free true/false.

Example server-side command:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H "Authorization: Bearer $VTON_API_TOKEN" \
  -F person_image=@person.png -F garment_image=@garment.png -F category=tops
```

Poll `GET /api/v1/jobs/<job_id>` with the same authorization header.
The completed response includes `generated_image` (base64 encoded PNG).

## Test

Run CPU-only API contract tests (inference mocked):

```bash
python -m pytest -q tests/test_backend_api.py
```

Then run a real GPU smoke test for single-garment and upper/lower try-on on the target server.
Verify CUDA compatibility, model loading, inference quality, VRAM, latency, proxy limits,
restart behavior and HTTPS/CORS before production traffic.

## Current operational limitations

- The job queue, state and generated base64 images are **process-local in-memory**.
  A restart deletes jobs. Multiple Uvicorn workers must **not** be used.
- Job cleanup is opportunistic on API requests, not a scheduled background purge.
  Long-running or high-volume use can consume significant process memory.
- Legacy synchronous endpoints share a GPU lock but can hold requests open for inference.
- There is no persistent Redis queue, individual user isolation, distributed worker,
  durable output storage, GPU execution timeout, or API-level per-customer rate limiting.
- The server token is shared across trusted services; upstream applications must perform
  user authentication and enforce per-user access to jobs.
- For production, add a persistent queue, signed result URLs, retention service,
  quotas and alerting as the next phase.
