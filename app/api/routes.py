"""REST API endpoints and request validation."""
import io
import logging
import time
import uuid
from typing import Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from app.core.config import CATEGORIES, PHOTO_TYPES, MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS
from app.core.security import _authenticate
from app.services.preprocessing import ImagePreprocessor
from app.services.inference import _infer
from app.services import jobs

LOG = logging.getLogger("shrivastra.vton")
preprocessor = ImagePreprocessor(max_resolution=2000)
router = APIRouter()

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


@router.post("/api/v1/jobs", status_code=202, dependencies=[Depends(_authenticate)])
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
    jobs._purgejobs._jobs()
    if jobs._job_queue is None:
        raise HTTPException(503, "Worker not running")
    if jobs._job_queue.full():
        raise HTTPException(429, "Inference queue is full", headers={"Retry-After": "10"})
    job = jobs.Job(id=str(uuid.uuid4()), created_at=time.time(), inputs=inputs)
    jobs._jobs[job.id] = job
    jobs._job_queue.put_nowait(job.id)
    return {"job_id": job.id, "status": job.status, "status_url": f"/api/v1/jobs/{job.id}"}


@router.get("/api/v1/jobs/{job_id}", dependencies=[Depends(_authenticate)])
async def get_job(job_id: str):
    jobs._purgejobs._jobs()
    job = jobs._jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    return {"job_id": job.id, "status": job.status, "generated_image": job.result,
            "error": job.error}


# Keep the existing endpoints and response format working for old clients.
# Calls are serialized through the same GPU lock as asynchronous jobs.
@router.post("/api/v1/virtual-tryon", dependencies=[Depends(_authenticate)])
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


@router.post("/api/v1/virtual-tryon-separate", dependencies=[Depends(_authenticate)])
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


