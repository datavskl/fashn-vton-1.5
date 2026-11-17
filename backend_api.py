"""Secure, single-GPU FastAPI service for FASHN VTON v1.5.

One application process / one GPU worker is supported. Run uvicorn with --workers 1.
"""
import asyncio
import base64
import io
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from image_preprocessing import ImagePreprocessor

LOG = logging.getLogger("shrivastra.vton")
CATEGORIES = {"tops", "bottoms", "one-pieces"}
PHOTO_TYPES = {"model", "flat-lay"}
MAX_UPLOAD_BYTES = int(os.getenv("VTON_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("VTON_MAX_IMAGE_PIXELS", "20000000"))
QUEUE_SIZE = int(os.getenv("VTON_QUEUE_SIZE", "20"))
JOB_TTL_SECONDS = int(os.getenv("VTON_JOB_TTL_SECONDS", "86400"))
API_TOKEN = os.getenv("VTON_API_TOKEN", "")
ALLOWED_ORIGINS = [value.strip() for value in os.getenv("VTON_CORS_ORIGINS", "").split(",") if value.strip()]
WEIGHTS_DIR = os.getenv("VTON_WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "weights"))
preprocessor = ImagePreprocessor(max_resolution=2000)
predictor = None
_gpu_lock = asyncio.Lock()
_jobs: dict[str, "Job"] = {}
_job_queue: asyncio.Queue | None = None
_worker_task: asyncio.Task | None = None


@dataclass
class Job:
    id: str
    created_at: float
    status: str = "queued"
    result: str | None = None
    error: str | None = None
    completed_at: float | None = None
    inputs: tuple = field(default_factory=tuple, repr=False)


def _purge_jobs():
    now = time.time()
    for job_id, job in list(_jobs.items()):
        if job.completed_at is not None and now - job.completed_at > JOB_TTL_SECONDS:
            del _jobs[job_id]


def _authenticate(request: Request):
    # Fail closed if no server-side token has been configured.
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})


def _validate(category: str, garment_photo_type: str, num_timesteps: int, guidance_scale: float):
    if category not in CATEGORIES:
        raise HTTPException(422, "Invalid category")
    if garment_photo_type not in PHOTO_TYPES:
        raise HTTPException(422, "Invalid garment_photo_type")
    if not 10 <= num_timesteps <= 50:
        raise HTTPException(422, "num_timesteps must be between 10 and 50")
    if not 1.0 <= guidance_scale <= 3.0:
        raise HTTPException(422, "guidance_scale must be between 1.0 and 3.0")


async def _image_upload(upload: UploadFile) -> Image.Image:
    if upload.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Only JPEG, PNG and WebP images are accepted")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image upload exceeds size limit")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Unsupported image encoding")
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image dimensions exceed pixel limit")
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            # Force RGB before JPEG conversion; preserve proper alpha compositing.
            result, _ = preprocessor.preprocess(image)
            return result.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Invalid image content") from exc
    finally:
        await upload.close()


def _download_weights():
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.join(WEIGHTS_DIR, "dwpose"), exist_ok=True)
    for repo_id, filename, subdir in [
        ("fashn-ai/fashn-vton-1.5", "model.safetensors", WEIGHTS_DIR),
        ("fashn-ai/DWPose", "yolox_l.onnx", os.path.join(WEIGHTS_DIR, "dwpose")),
        ("fashn-ai/DWPose", "dw-ll_ucoco_384.onnx", os.path.join(WEIGHTS_DIR, "dwpose")),
    ]:
        if not os.path.isfile(os.path.join(subdir, filename)):
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=subdir)


def _get_predictor():
    global predictor
    if predictor is None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable")
        _download_weights()
        from fashn_vton import TryOnPipeline
        predictor = TryOnPipeline(weights_dir=WEIGHTS_DIR, device="cuda")
    return predictor


def _run_inference(person, upper, lower, category, photo_type, steps, guidance, seed, segmentation_free):
    pipeline = _get_predictor()
    current = person
    tasks = [(upper, category)] if upper is not None else []
    if lower is not None:
        tasks.append((lower, "bottoms"))
    for garment, selected_category in tasks:
        result = pipeline(
            person_image=current, garment_image=garment, category=selected_category,
            garment_photo_type=photo_type, num_samples=1, num_timesteps=steps,
            guidance_scale=guidance, seed=seed, segmentation_free=segmentation_free,
        )
        current = result.images[0]
    buffer = io.BytesIO()
    current.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _infer(inputs):
    async with _gpu_lock:
        return await asyncio.to_thread(_run_inference, *inputs)


async def _job_worker():
    assert _job_queue is not None
    while True:
        job_id = await _job_queue.get()
        try:
            job = _jobs.get(job_id)
            if job:
                job.status = "processing"
                try:
                    job.result = await _infer(job.inputs)
                    job.status = "completed"
                except Exception:
                    LOG.exception("Inference failed for job %s", job.id)
                    job.error = "Inference failed; check server logs"
                    job.status = "failed"
                finally:
                    job.inputs = ()
                    job.completed_at = time.time()
        finally:
            _job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _job_queue, _worker_task
    _job_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    _worker_task = asyncio.create_task(_job_worker())
    try:
        yield
    finally:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
        _job_queue = None


app = FastAPI(title="ShriVastra FASHN VTON API", version="1.5.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
async def health():
    return {"status": "running", "gpu_available": torch.cuda.is_available(),
            "model_loaded": predictor is not None}


@app.get("/")
async def root():
    return {"name": "ShriVastra FASHN VTON API", "docs": "/docs", "health": "/health"}


async def _collect(person_image, garment_image, upper_garment_image, lower_garment_image,
                   category, garment_photo_type, num_timesteps, guidance_scale, seed, segmentation_free):
    _validate(category, garment_photo_type, num_timesteps, guidance_scale)
    if garment_image is not None and (upper_garment_image is not None or lower_garment_image is not None):
        raise HTTPException(422, "Choose single garment or upper/lower garments")
    if garment_image is None and upper_garment_image is None and lower_garment_image is None:
        raise HTTPException(422, "At least one garment is required")
    person = await _image_upload(person_image)
    upper = await _image_upload(garment_image or upper_garment_image) if garment_image or upper_garment_image else None
    lower = await _image_upload(lower_garment_image) if lower_garment_image else None
    return (person, upper, lower, category, garment_photo_type, num_timesteps,
            guidance_scale, seed, segmentation_free)


@app.post("/api/v1/jobs", status_code=202, dependencies=[Depends(_authenticate)])
async def submit_job(
    person_image: UploadFile = File(...),
    garment_image: UploadFile | None = File(None),
    upper_garment_image: UploadFile | None = File(None),
    lower_garment_image: UploadFile | None = File(None),
    category: Literal["tops", "bottoms", "one-pieces"] = Form("tops"),
    garment_photo_type: Literal["model", "flat-lay"] = Form("model"),
    num_timesteps: int = Form(50, ge=10, le=50),
    guidance_scale: float = Form(1.5, ge=1.0, le=3.0),
    seed: int = Form(42, ge=0),
    segmentation_free: bool = Form(True),
):
    inputs = await _collect(person_image, garment_image, upper_garment_image, lower_garment_image,
                            category, garment_photo_type, num_timesteps, guidance_scale, seed, segmentation_free)
    _purge_jobs()
    if _job_queue is None:
        raise HTTPException(503, "Worker not running")
    if _job_queue.full():
        raise HTTPException(429, "Inference queue is full", headers={"Retry-After": "10"})
    job = Job(id=str(uuid.uuid4()), created_at=time.time(), inputs=inputs)
    _jobs[job.id] = job
    _job_queue.put_nowait(job.id)
    return {"job_id": job.id, "status": job.status, "status_url": f"/api/v1/jobs/{job.id}"}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(_authenticate)])
async def get_job(job_id: str):
    _purge_jobs()
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    return {"job_id": job.id, "status": job.status, "generated_image": job.result,
            "error": job.error}


# Keep the existing endpoints and response format working for old clients.
# Calls are serialized through the same GPU lock as asynchronous jobs.
@app.post("/api/v1/virtual-tryon", dependencies=[Depends(_authenticate)])
async def virtual_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: Literal["tops", "bottoms", "one-pieces"] = Form("tops"),
    garment_photo_type: Literal["model", "flat-lay"] = Form("model"),
    num_timesteps: int = Form(50, ge=10, le=50),
    guidance_scale: float = Form(1.5, ge=1.0, le=3.0),
    seed: int = Form(42, ge=0),
    segmentation_free: bool = Form(True),
):
    inputs = await _collect(person_image, garment_image, None, None, category, garment_photo_type,
                            num_timesteps, guidance_scale, seed, segmentation_free)
    try:
        result = await _infer(inputs)
    except Exception:
        LOG.exception("Synchronous inference failed")
        raise HTTPException(503, "Inference unavailable")
    return {"success": True, "generated_image": result,
            "parameters": {"category": category, "garment_photo_type": garment_photo_type,
                           "num_timesteps": num_timesteps, "guidance_scale": guidance_scale,
                           "seed": seed, "segmentation_free": segmentation_free}}


@app.post("/api/v1/virtual-tryon-separate", dependencies=[Depends(_authenticate)])
async def virtual_tryon_separate(
    person_image: UploadFile = File(...),
    upper_garment_image: UploadFile | None = File(None),
    lower_garment_image: UploadFile | None = File(None),
    garment_photo_type: Literal["model", "flat-lay"] = Form("model"),
    num_timesteps: int = Form(50, ge=10, le=50),
    guidance_scale: float = Form(1.5, ge=1.0, le=3.0),
    seed: int = Form(42, ge=0),
    segmentation_free: bool = Form(True),
):
    inputs = await _collect(person_image, None, upper_garment_image, lower_garment_image,
                            "tops", garment_photo_type, num_timesteps, guidance_scale, seed, segmentation_free)
    try:
        result = await _infer(inputs)
    except Exception:
        LOG.exception("Synchronous inference failed")
        raise HTTPException(503, "Inference unavailable")
    stages = [s for s, value in [("upper_garment", upper_garment_image), ("lower_garment", lower_garment_image)] if value]
    return {"success": True, "generated_image": result, "processing_steps": stages,
            "parameters": {"garment_photo_type": garment_photo_type, "num_timesteps": num_timesteps,
                           "guidance_scale": guidance_scale, "seed": seed,
                           "segmentation_free": segmentation_free, "garments_processed": len(stages)}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, workers=1)
