# ShriVastra AI — Lightning AI L40S Deployment Guide

ShriVastra uses [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) as its underlying virtual try-on model. This fork adds a modular FastAPI service, authenticated endpoints, a single-GPU inference queue, automated API checks, and a WordPress server-side bridge.

**This is the staging deployment guide for an NVIDIA L40S (48 GB VRAM) in Lightning AI Studio.** Real GPU inference must be verified on your Studio; GitHub Actions tests mock inference.

- **Development branch:** `refactor/shrivastra-single-backend`
- **Production API entry point:** `app/main.py` (launch with `app.main:app`)
- **GPU process configuration:** one Uvicorn worker and one active inference at a time
- **Related documentation:** [Lightning AI detailed runbook](docs/LIGHTNING_AI_DEPLOYMENT.md) · [deployment reference](docs/DEPLOYMENT.md) · [GPU milestone](docs/GPU_MILESTONE.md)
- **Important:** do not merge this branch into `main` before real GPU staging acceptance.

## 1. Open your NVIDIA L40S Studio

Open [Lightning AI Studios](https://studio.lightning.ai), select your Studio, and choose the NVIDIA L40S GPU (48 GB VRAM). Open a terminal and verify that it is attached:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

You should see L40S and `CUDA: True`. If Torch is not installed yet, run the second command after dependency installation. Confirm sufficient persistent storage for model weights and cache. GPU prices, quota, and machine availability depend on your Lightning AI account.

## 2. Clone the development branch

```bash
git clone https://github.com/datavskl/fashn-vton-1.5.git
cd fashn-vton-1.5
git fetch origin
git switch refactor/shrivastra-single-backend
git branch --show-current
```

Do not run these setup steps on `main` or merge your staging branch.

## 3. Install Python and CUDA-compatible dependencies

Use Python 3.11 where available. Install a CUDA-compatible PyTorch build suitable for the Studio's NVIDIA driver if one is not already installed; verify with the PyTorch installation instructions rather than forcing a CPU-only package.

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pip install python-dotenv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If installation or GPU detection fails, inspect CUDA/PyTorch/ONNX Runtime compatibility before proceeding. **Do not assume a green API unit-test suite proves GPU support.**

## 4. Download model weights

```bash
python scripts/download_weights.py --weights-dir ./weights
```

This fetches the FASHN model and pose-estimation weights from Hugging Face. Additional human parser assets may download during first inference. Store model weights/cache on persistent Studio storage where possible.

## 5. Configure staging secrets and safety limits

Generate a random API bearer token **locally in your Studio**, then create a private `.env` file. Never commit it or expose the token to public browser JavaScript.

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
nano .env
```

Example `.env` (**replace the placeholder**):

```dotenv
VTON_API_TOKEN=PASTE_YOUR_NEW_RANDOM_SECRET_HERE
VTON_CORS_ORIGINS=
VTON_WEIGHTS_DIR=./weights
VTON_MAX_UPLOAD_BYTES=10485760
VTON_MAX_IMAGE_PIXELS=20000000
VTON_QUEUE_SIZE=4
VTON_JOB_TTL_SECONDS=3600
```

Keep CORS origins empty for local, server-to-server staging. Do not enable public inference access. A restricted HTTPS proxy with authentication, request size caps and rate limits is required before exposing the API. The application validates each upload, but an ingress/proxy limit is still required to stop oversized multipart requests before parsing.

## 6. Run FastAPI privately

From the project root with the virtual environment activated:

```bash
uvicorn app.main:app   --host 127.0.0.1   --port 8000   --workers 1   --env-file .env
```

In a **second terminal**:

```bash
cd fashn-vton-1.5
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/health
python -m pytest -q tests/test_backend_api.py
```

The health response indicates GPU visibility and whether the model has been loaded, but **does not execute model inference**. The model loads on the first real inference request; the first job may be slower. Keep the Studio and API endpoint private.

## 7. Run real-image tests on the L40S

Use your own or consented test photographs of a person, a top, and a bottom. Place them in a private directory (e.g., `/home/your-user/private-vton-test/`). Do not commit original customer images or generated outputs.

Activate the virtual environment in the second terminal, and export the **same token** that is in the API server's `.env` file:

```bash
export VTON_API_TOKEN='PASTE_YOUR_NEW_RANDOM_SECRET_HERE'
python scripts/gpu_smoke_test.py   --api-url http://127.0.0.1:8000   --person /path/to/person.jpg   --top /path/to/top.jpg   --bottom /path/to/bottom.jpg   --timeout 600
```

The smoke test submits three actual inference cases: **top only**, **bottom only**, and **top + bottom**. It validates generated PNGs and captures per-case latency and observed GPU memory usage. Find your private results in `test_outputs/gpu-smoke/`.

### Acceptance checklist

- [ ] L40S is detected and `torch.cuda.is_available()` is true.
- [ ] Model weights load successfully; first and subsequent inference runs work.
- [ ] Top, bottom, and combined outfit images are generated and visually reviewed.
- [ ] Record cold-start time, warm latency, memory usage, and output dimensions.
- [ ] Invalid token (401), bad uploads (4xx), and queue pressure are tested.
- [ ] Confirm a Studio restart invalidates in-memory jobs, and model weights persist.
- [ ] Do not expose customer images, API secrets or generated results publicly.

**Limitations:** This staging backend has an in-memory queue/results, a shared service token, no GPU execution timeout, no durable image storage, and no customer-level isolation. Run only one Uvicorn worker. Add persistent jobs, monitoring, quotas and protected delivery before a production launch.

## 8. WordPress integration (after GPU acceptance)

The repository contains `wordpress/shrivastra-vton-bridge/`, a server-side integration. The GPU token must stay on the WordPress server, **never** in frontend JavaScript. Expose a TLS-protected API only after ingress security and rate limits are configured. See [GPU milestone and WordPress staging notes](docs/GPU_MILESTONE.md).

---

## Upstream FASHN VTON v1.5 model documentation and attribution

The material below is retained from the upstream FASHN AI project for inference details, model usage, licensing and research citation. Its generic installation instructions are not the ShriVastra Lightning AI API deployment path.

# FASHN VTON v1.5: Efficient Maskless Virtual Try-On in Pixel Space

<div align="center">
  <a href="https://fashn.ai/research/vton-1-5"><img src='https://img.shields.io/badge/Project-Page-1A1A1A?style=flat' alt='Project Page'></a>&ensp;
  <a href='https://huggingface.co/fashn-ai/fashn-vton-1.5'><img src='https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=flat&logo=HuggingFace&logoColor=FFD21E' alt='Hugging Face Model'></a>&ensp;
  <a href="https://huggingface.co/spaces/fashn-ai/fashn-vton-1.5"><img src='https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=flat&logo=HuggingFace&logoColor=FFD21E' alt='Hugging Face Spaces'></a>&ensp;
  <a href=""><img src='https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b?style=flat&logo=arXiv&logoColor=b31b1b' alt='arXiv'></a>&ensp;
  <a href="LICENSE"><img src='https://img.shields.io/badge/License-Apache--2.0-gray?style=flat' alt='License'></a>
</div>

by [FASHN AI](https://fashn.ai)

Virtual try-on model that generates photorealistic images directly in pixel space without requiring segmentation masks.

<p align="center">
  <img src="https://static.fashn.ai/repositories/fashn-vton-v15/results/hero_collage.webp" alt="FASHN VTON v1.5 examples" width="900">
</p>

This repo contains minimal inference code to run virtual try-on with the FASHN VTON v1.5 model weights. Given a person image and a garment image, the model generates a photorealistic image of the person wearing the garment. Supports both model photos and flat-lay product shots as garment inputs.

---

## Local Installation

We recommend using a virtual environment:

```bash
git clone https://github.com/fashn-AI/fashn-vton-1.5.git
cd fashn-vton-1.5
## for local PC or custom server
python -m venv .venv && source .venv/bin/activate
pip install -e .

```

**Note:** Installation includes `onnxruntime-gpu` for GPU-accelerated pose detection. Ensure CUDA is properly configured on your system. For CPU-only environments, replace with the CPU version:

```bash
pip uninstall onnxruntime-gpu && pip install onnxruntime
```

---

## Model Weights

Download the required model weights (~2 GB total):

```bash
python scripts/download_weights.py --weights-dir ./weights
```

This downloads:
- `model.safetensors` — TryOnModel weights from [HuggingFace](https://huggingface.co/fashn-ai/fashn-vton-1.5)
- `dwpose/` — DWPose ONNX models for pose detection

**Note:** The human parser weights (~244 MB) are automatically downloaded on first use to the HuggingFace cache folder. Set `HF_HOME` to customize the location.

---

## Usage

```python
from fashn_vton import TryOnPipeline
from PIL import Image

# Initialize pipeline (automatically uses GPU if available)
pipeline = TryOnPipeline(weights_dir="./weights")

# Load images
person = Image.open("examples/data/model.webp").convert("RGB")
garment = Image.open("examples/data/garment.webp").convert("RGB")

# Run inference
result = pipeline(
    person_image=person,
    garment_image=garment,
    category="tops",  # "tops" | "bottoms" | "one-pieces"
)

# Save output
result.images[0].save("output.png")
```

### CLI

```bash
python examples/basic_inference.py \
    --weights-dir ./weights \
    --person-image examples/data/model.webp \
    --garment-image examples/data/garment.webp \
    --category tops
```

**Note:** The pipeline automatically uses GPU if available. The try-on model weights are stored in bfloat16 and will run in bf16 precision on Ampere+ GPUs (RTX 30xx/40xx, A100, H100). On older hardware or CPU, weights are converted to float32.

See [`examples/basic_inference.py`](examples/basic_inference.py) for additional options.

---

## Categories

| Category | Description |
|----------|-------------|
| `tops` | Upper body: t-shirts, blouses, jackets |
| `bottoms` | Lower body: pants, skirts, shorts |
| `one-pieces` | Full body: dresses, jumpsuits |

---

## API

FASHN provides a suite of [fashion AI APIs](https://fashn.ai/products/api) including virtual try-on, model generation, image-to-video, and more. See the [docs](https://docs.fashn.ai/) to get started.

---

## Citation

If you use FASHN VTON v1.5 in your research, please cite:

```bibtex
@article{bochman2026fashnvton,
  title={FASHN VTON v1.5: Efficient Maskless Virtual Try-On in Pixel Space},
  author={Bochman, Dan and Bochman, Aya},
  journal={arXiv preprint},
  year={2026},
  note={Paper coming soon}
}
```

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

**Third-party components:**
- [DWPose](https://github.com/IDEA-Research/DWPose) (Apache-2.0)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0)
- [fashn-human-parser](https://github.com/fashn-AI/fashn-human-parser) ([License](https://github.com/fashn-AI/fashn-human-parser?tab=readme-ov-file#license))

