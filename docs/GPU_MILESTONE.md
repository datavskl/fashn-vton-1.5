# Phase 2: NVIDIA GPU staging and ShriVastra WordPress integration

## GPU host prerequisites

- Linux host with a compatible NVIDIA GPU, functional driver, Docker, Docker Compose, and NVIDIA Container Toolkit.
- Test `nvidia-smi` on host and `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- Check out develop/shrivastra-vton-backend. Leave main untouched.
- Generate a long random server token: `python3 -c 'import secrets; print(secrets.token_urlsafe(48))'`.
- Create a private root `.env` with `VTON_API_TOKEN=<random token>` and `VTON_CORS_ORIGINS=https://your-store.example`.

## Start the single-worker GPU service

```bash
docker compose -f compose.gpu.yml build
docker compose -f compose.gpu.yml up -d
docker compose -f compose.gpu.yml logs --tail=100 vton
curl http://127.0.0.1:8000/health
```

Port 8000 is intentionally bound to localhost. Use a TLS-enabled, authenticated reverse proxy
or an SSH tunnel for remote testing, and limit request body size at the proxy.
Ensure the PyTorch, CUDA runtime, ONNX and driver versions are compatible on the actual host.
The model loads on the first real inference request; /health is not an inference-readiness guarantee.

## Real-inference GPU smoke test

Use consented test images only. Run on the host or through a protected tunnel:

```bash
export VTON_API_TOKEN='your-server-token'
python scripts/gpu_smoke_test.py --api-url http://127.0.0.1:8000 --person person.jpg --top top.jpg --bottom bottom.jpg
```

The script submits real GPU jobs for top, bottom and combined outfits, polls for completion,
validates generated PNG output, records request latency, and samples device-wide GPU used-memory.
Results and images are saved in test_outputs/gpu-smoke and must be kept private.
Set --timeout 600 if the target GPU is slow.

Acceptance: all three inference jobs complete; manual inspection confirms pose/garment/face quality;
record latency, output dimensions and GPU memory usage; verify auth failures, proxy limits, TLS,
CORS, one-worker behavior and restart handling. Cold-start and warm-run timings should be measured.

## WordPress staging bridge

Install wordpress/shrivastra-vton-bridge/shrivastra-vton-bridge.php as a plugin on a staging
WordPress site (PHP 8+ with cURL), then set these two constants in wp-config.php:

```php
define('SHRIVASTRA_VTON_API_URL', 'https://your-protected-gpu-api.example');
define('SHRIVASTRA_VTON_API_TOKEN', 'your-long-server-token');
```

Authenticated WordPress REST endpoints:

- POST /wp-json/shrivastra-vton/v1/try-on: upload person_image plus garment_image
  or separate upper_garment_image / lower_garment_image; optional category field.
- GET /wp-json/shrivastra-vton/v1/jobs/{job_id}: retrieve the logged-in submitter's job.

Browser calls must use WordPress login cookies and X-WP-Nonce from wp_create_nonce('wp_rest').
The GPU API secret is held on the server and never exposed to browsers.
Plugin requests are limited to 5 submissions per user per minute (basic transient limit).
Integrating the existing ShriVastra frontend garment selection and WooCommerce catalog is still pending.

## Not production ready

The GPU queue and results are in-memory and lost on restart; there is no durable Redis queue,
per-user GPU isolation, model inference timeout, signed image URLs, or scheduled data deletion.
Large images and job results held in memory can consume significant RAM.
Do not deploy multiple Uvicorn workers or expose this service directly to public clients.
Use staging first, then add durable jobs, object storage, user-level quotas and observability.

### Verification state

GitHub Actions runs CPU-only mocked API contract tests and PHP syntax validation.
Actual GPU performance and WordPress staging integration require host access and real deployments.