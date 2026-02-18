"""FASHN VTON v1.5 FastAPI Backend for Production."""

import os
import io
import uuid
import base64
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import torch
from huggingface_hub import hf_hub_download

# ----------------- CONFIG ----------------- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")

# Create directories
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

CATEGORIES = ["tops", "bottoms", "one-pieces"]
GARMENT_PHOTO_TYPES = ["model", "flat-lay"]

# Job storage (in production, use Redis or database)
jobs_storage: Dict[str, Dict[str, Any]] = {}

# Global pipeline instance
_pipeline = None
_pipeline_lock = asyncio.Lock()

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)


# ----------------- MODELS ----------------- #

class TryOnRequest(BaseModel):
    """Request model for try-on inference."""
    category: str = Field(..., description="Garment category")
    garment_photo_type: str = Field(..., description="Type of garment photo")
    num_timesteps: int = Field(default=50, ge=10, le=50, description="Number of sampling steps")
    guidance_scale: float = Field(default=1.5, ge=1.0, le=3.0, description="Guidance scale")
    seed: int = Field(default=42, ge=0, description="Random seed")
    segmentation_free: bool = Field(default=True, description="Use segmentation-free mode")


class TryOnResponse(BaseModel):
    """Response model for try-on inference."""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status response model."""
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_available: bool
    model_loaded: bool
    version: str


# ----------------- HELPERS ----------------- #

def download_weights():
    """Download model weights from HuggingFace Hub."""
    dwpose_dir = os.path.join(WEIGHTS_DIR, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)

    # Download TryOnModel weights
    tryon_path = os.path.join(WEIGHTS_DIR, "model.safetensors")
    if not os.path.exists(tryon_path):
        print("Downloading TryOnModel weights...")
        hf_hub_download(
            repo_id="fashn-ai/fashn-vton-1.5",
            filename="model.safetensors",
            local_dir=WEIGHTS_DIR,
        )

    # Download DWPose models
    dwpose_files = ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]
    for filename in dwpose_files:
        filepath = os.path.join(dwpose_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading DWPose/{filename}...")
            hf_hub_download(
                repo_id="fashn-ai/DWPose",
                filename=filename,
                local_dir=dwpose_dir,
            )

    print("Weights downloaded successfully!")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


async def get_pipeline():
    """Load the pipeline (thread-safe singleton)."""
    global _pipeline
    
    async with _pipeline_lock:
        if _pipeline is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. This service requires a GPU.")

            # Enable TF32 for faster computation on Ampere+ GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            print("Downloading weights (if needed)...")
            await asyncio.get_event_loop().run_in_executor(executor, download_weights)

            print("Loading pipeline...")
            from fashn_vton import TryOnPipeline
            
            _pipeline = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: TryOnPipeline(weights_dir=WEIGHTS_DIR, device="cuda")
            )
            print("Pipeline loaded on CUDA!")

    return _pipeline


def run_inference(
    person_image: Image.Image,
    garment_image: Image.Image,
    category: str,
    garment_photo_type: str,
    num_timesteps: int,
    guidance_scale: float,
    seed: int,
    segmentation_free: bool,
) -> Image.Image:
    """Run virtual try-on inference (blocking)."""
    global _pipeline
    
    # Convert to RGB if needed
    if person_image.mode != "RGB":
        person_image = person_image.convert("RGB")
    if garment_image.mode != "RGB":
        garment_image = garment_image.convert("RGB")

    # Run inference
    result = _pipeline(
        person_image=person_image,
        garment_image=garment_image,
        category=category,
        garment_photo_type=garment_photo_type,
        num_samples=1,
        num_timesteps=num_timesteps,
        guidance_scale=guidance_scale,
        seed=int(seed),
        segmentation_free=segmentation_free,
    )

    return result.images[0]


async def process_job(job_id: str, job_data: Dict[str, Any]):
    """Background task to process try-on job."""
    try:
        jobs_storage[job_id]["status"] = "processing"
        jobs_storage[job_id]["progress"] = 10
        
        # Ensure pipeline is loaded
        await get_pipeline()
        jobs_storage[job_id]["progress"] = 30
        
        # Load images
        person_image = base64_to_image(job_data["person_image_b64"])
        garment_image = base64_to_image(job_data["garment_image_b64"])
        jobs_storage[job_id]["progress"] = 50
        
        # Run inference in thread pool
        result_image = await asyncio.get_event_loop().run_in_executor(
            executor,
            run_inference,
            person_image,
            garment_image,
            job_data["category"],
            job_data["garment_photo_type"],
            job_data["num_timesteps"],
            job_data["guidance_scale"],
            job_data["seed"],
            job_data["segmentation_free"],
        )
        jobs_storage[job_id]["progress"] = 90
        
        # Save result
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
        result_image.save(output_path)
        
        # Update job status
        jobs_storage[job_id]["status"] = "completed"
        jobs_storage[job_id]["progress"] = 100
        jobs_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()
        jobs_storage[job_id]["result_url"] = f"/results/{job_id}.png"
        
    except Exception as e:
        jobs_storage[job_id]["status"] = "failed"
        jobs_storage[job_id]["error"] = str(e)
        jobs_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()
        print(f"Job {job_id} failed: {e}")


# ----------------- API ----------------- #

app = FastAPI(
    title="FASHN VTON API",
    description="Virtual Try-On API for fashion garments",
    version="1.5.0",
)

# CORS middleware (configure origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "FASHN VTON API",
        "version": "1.5.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        model_loaded=_pipeline is not None,
        version="1.5.0"
    )


@app.post("/api/v1/try-on", response_model=TryOnResponse)
async def create_try_on_job(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="Person image file"),
    garment_image: UploadFile = File(..., description="Garment image file"),
    category: str = Form(..., description="Garment category (tops/bottoms/one-pieces)"),
    garment_photo_type: str = Form(..., description="Photo type (model/flat-lay)"),
    num_timesteps: int = Form(50, description="Number of sampling steps (10-50)"),
    guidance_scale: float = Form(1.5, description="Guidance scale (1.0-3.0)"),
    seed: int = Form(42, description="Random seed"),
    segmentation_free: bool = Form(True, description="Use segmentation-free mode"),
):
    """
    Create a new virtual try-on job (async).
    
    Returns job_id for status polling.
    """
    # Validate inputs
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of {CATEGORIES}")
    if garment_photo_type not in GARMENT_PHOTO_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid photo type. Must be one of {GARMENT_PHOTO_TYPES}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Read and validate images
    try:
        person_img_data = await person_image.read()
        garment_img_data = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_img_data))
        garment_img = Image.open(io.BytesIO(garment_img_data))
        
        # Convert to base64 for storage
        person_b64 = image_to_base64(person_img)
        garment_b64 = image_to_base64(garment_img)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Create job
    job_data = {
        "person_image_b64": person_b64,
        "garment_image_b64": garment_b64,
        "category": category,
        "garment_photo_type": garment_photo_type,
        "num_timesteps": num_timesteps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "segmentation_free": segmentation_free,
    }
    
    jobs_storage[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "progress": 0,
    }
    
    # Queue background task
    background_tasks.add_task(process_job, job_id, job_data)
    
    return TryOnResponse(
        job_id=job_id,
        status="pending",
        message="Job created successfully. Use /api/v1/jobs/{job_id} to check status."
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a try-on job."""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    return JobStatus(**job)


@app.get("/api/v1/results/{job_id}.png")
async def get_result_image(job_id: str):
    """Get result image for completed job."""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(output_path, media_type="image/png")


@app.post("/api/v1/try-on/sync")
async def try_on_sync(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form(...),
    garment_photo_type: str = Form(...),
    num_timesteps: int = Form(50),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42),
    segmentation_free: bool = Form(True),
):
    """
    Synchronous try-on endpoint (returns result directly).
    Use for testing or low-latency requirements.
    """
    # Validate inputs
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of {CATEGORIES}")
    if garment_photo_type not in GARMENT_PHOTO_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid photo type. Must be one of {GARMENT_PHOTO_TYPES}")
    
    # Read images
    try:
        person_img_data = await person_image.read()
        garment_img_data = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_img_data))
        garment_img = Image.open(io.BytesIO(garment_img_data))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Load pipeline
    await get_pipeline()
    
    # Run inference
    try:
        result_image = await asyncio.get_event_loop().run_in_executor(
            executor,
            run_inference,
            person_img,
            garment_img,
            category,
            garment_photo_type,
            num_timesteps,
            guidance_scale,
            seed,
            segmentation_free,
        )
        
        # Convert to base64
        result_b64 = image_to_base64(result_image)
        
        return JSONResponse({
            "status": "success",
            "result_image": result_b64,
            "format": "base64_png"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)