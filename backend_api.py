# backend_api.py
import numpy as np
from PIL import Image
import io
import base64
import platform
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from huggingface_hub import hf_hub_download
import os

# Import preprocessing utilities
from image_preprocessing import ImagePreprocessor

# ----------------- CONFIG ----------------- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")

CATEGORIES = ["tops", "bottoms", "one-pieces"]
GARMENT_PHOTO_TYPES = ["model", "flat-lay"]

# Image preprocessing settings
MAX_RESOLUTION_1K = 2000  # For Try-On v1.5/v1.6
JPEG_QUALITY = 95

# Global preprocessor instance
image_preprocessor = ImagePreprocessor(max_resolution=MAX_RESOLUTION_1K)


# ----------------- HELPERS ----------------- #

def download_weights():
    """Download model weights from HuggingFace Hub."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
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
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def numpy_to_base64(array: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    image = Image.fromarray(array.astype('uint8'))
    return image_to_base64(image)


# ----------------- FASHN VTON PREDICTOR ----------------- #

class FashnVTONPredictor(object):
    def __init__(self):
        """Initialize the FASHN VTON predictor."""
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This service requires a GPU.")

        # ---------------------------------- Diagnostics ---------------------------------- #
        print(f"Python           : {platform.python_version()}")
        print(f"PyTorch          : {torch.__version__}")
        print(f" • built for CUDA : {torch.version.cuda}")
        if torch.backends.cudnn.is_available():
            print(f" • built for cuDNN: {torch.backends.cudnn.version()}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            cc = torch.cuda.get_device_capability(dev)
            print(f"GPU {dev}: {torch.cuda.get_device_name(dev)} (compute {cc[0]}.{cc[1]})")

        # Enable TF32 for faster computation on Ampere+ GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("Downloading weights (if needed)...")
        download_weights()

        print("Loading pipeline...")
        from fashn_vton import TryOnPipeline

        self.pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR, device="cuda")
        print("Pipeline loaded on CUDA!")

    def process_virtual_tryon(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: str = "tops",
        garment_photo_type: str = "model",
        num_timesteps: int = 50,
        guidance_scale: float = 1.5,
        seed: int = 42,
        segmentation_free: bool = True,
    ):
        """Process virtual try-on inference."""
        # Handle seed (guard against None or invalid values)
        if seed is None or seed < 0:
            seed = 42

        # Convert to RGB if needed
        if person_image.mode != "RGB":
            person_image = person_image.convert("RGB")
        if garment_image.mode != "RGB":
            garment_image = garment_image.convert("RGB")

        # Run inference
        result = self.pipeline(
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


# ----------------- FASTAPI APP ----------------- #

app = FastAPI(
    title="FASHN VTON API",
    description="API for virtual try-on using FASHN VTON v1.5 model",
    version="1.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    global predictor
    predictor = FashnVTONPredictor()
    print("✅ FASHN VTON Predictor initialized successfully")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FASHN VTON API",
        "status": "running",
        "version": "1.5.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "virtual_tryon": "/api/v1/virtual-tryon",
            "virtual_tryon_separate": "/api/v1/virtual-tryon-separate"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "gpu_available": torch.cuda.is_available()
    }


@app.post("/api/v1/virtual-tryon")
async def virtual_tryon(
    person_image: UploadFile = File(..., description="Person image file"),
    garment_image: UploadFile = File(..., description="Garment image file"),
    category: str = Form("tops", description="Garment category: tops, bottoms, one-pieces"),
    garment_photo_type: str = Form("model", description="Photo type: model, flat-lay"),
    num_timesteps: int = Form(50, description="Number of sampling steps (10-50)"),
    guidance_scale: float = Form(1.5, description="Guidance scale (1.0-3.0)"),
    seed: int = Form(42, description="Random seed for reproducibility"),
    segmentation_free: bool = Form(True, description="Use segmentation-free mode"),
):
    """
    Virtual try-on endpoint (single garment).
    
    Returns the generated try-on image as base64 encoded PNG.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate inputs
        if num_timesteps < 10 or num_timesteps > 50:
            raise HTTPException(status_code=400, detail="num_timesteps must be between 10 and 50")
        
        if guidance_scale < 1.0 or guidance_scale > 3.0:
            raise HTTPException(status_code=400, detail="guidance_scale must be between 1.0 and 3.0")
        
        if category not in CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of {CATEGORIES}")
        
        if garment_photo_type not in GARMENT_PHOTO_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid garment_photo_type. Must be one of {GARMENT_PHOTO_TYPES}")
        
        # Load images
        person_img = Image.open(io.BytesIO(await person_image.read()))
        garment_img = Image.open(io.BytesIO(await garment_image.read()))
        
        # Preprocess images according to FASHN best practices
        # 1. Resize to max 2000px (maintaining aspect ratio)
        # 2. Convert to JPEG with quality 95
        # 3. Use LANCZOS resampling for quality preservation
        person_img, person_metadata = image_preprocessor.preprocess(person_img)
        garment_img, garment_metadata = image_preprocessor.preprocess(garment_img)
        
        print(f"Person image preprocessed: {person_metadata.get('resized', False)}, "
              f"size: {person_metadata.get('final_size')}")
        print(f"Garment image preprocessed: {garment_metadata.get('resized', False)}, "
              f"size: {garment_metadata.get('final_size')}")
        
        # Process virtual try-on
        generated_image = predictor.process_virtual_tryon(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
            segmentation_free=segmentation_free,
        )
        
        # Return response
        response = {
            "success": True,
            "generated_image": image_to_base64(generated_image),
            "parameters": {
                "category": category,
                "garment_photo_type": garment_photo_type,
                "num_timesteps": num_timesteps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "segmentation_free": segmentation_free
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/v1/virtual-tryon-separate")
async def virtual_tryon_separate(
    person_image: UploadFile = File(..., description="Person image file"),
    upper_garment_image: Optional[UploadFile] = File(None, description="Upper garment image (optional)"),
    lower_garment_image: Optional[UploadFile] = File(None, description="Lower garment image (optional)"),
    garment_photo_type: str = Form("model", description="Photo type: model, flat-lay"),
    num_timesteps: int = Form(50, description="Number of sampling steps (10-50)"),
    guidance_scale: float = Form(1.5, description="Guidance scale (1.0-3.0)"),
    seed: int = Form(42, description="Random seed for reproducibility"),
    segmentation_free: bool = Form(True, description="Use segmentation-free mode"),
):
    """
    Virtual try-on endpoint with separate upper and lower garments.
    
    Supports:
    - Upper garment only (tops)
    - Lower garment only (bottoms)
    - Both garments (complete outfit - processed sequentially)
    
    At least one garment must be provided.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate at least one garment is provided
        if upper_garment_image is None and lower_garment_image is None:
            raise HTTPException(
                status_code=400, 
                detail="At least one garment (upper or lower) must be provided"
            )
        
        # Validate inputs
        if num_timesteps < 10 or num_timesteps > 50:
            raise HTTPException(status_code=400, detail="num_timesteps must be between 10 and 50")
        
        if guidance_scale < 1.0 or guidance_scale > 3.0:
            raise HTTPException(status_code=400, detail="guidance_scale must be between 1.0 and 3.0")
        
        if garment_photo_type not in GARMENT_PHOTO_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid garment_photo_type. Must be one of {GARMENT_PHOTO_TYPES}")
        
        # Load person image
        person_img = Image.open(io.BytesIO(await person_image.read()))
        
        # Preprocess person image
        person_img, person_metadata = image_preprocessor.preprocess(person_img)
        print(f"Person image preprocessed: {person_metadata.get('resized', False)}, "
              f"size: {person_metadata.get('final_size')}")
        
        # Process garments
        current_result = person_img
        processing_steps = []
        preprocessing_info = {'person': person_metadata}
        
        # Process upper garment if provided
        if upper_garment_image is not None:
            upper_img = Image.open(io.BytesIO(await upper_garment_image.read()))
            
            # Preprocess upper garment
            upper_img, upper_metadata = image_preprocessor.preprocess(upper_img)
            preprocessing_info['upper_garment'] = upper_metadata
            print(f"Upper garment preprocessed: {upper_metadata.get('resized', False)}, "
                  f"size: {upper_metadata.get('final_size')}")
            
            current_result = predictor.process_virtual_tryon(
                person_image=current_result,
                garment_image=upper_img,
                category="tops",
                garment_photo_type=garment_photo_type,
                num_timesteps=num_timesteps,
                guidance_scale=guidance_scale,
                seed=seed,
                segmentation_free=segmentation_free,
            )
            
            processing_steps.append("upper_garment")
        
        # Process lower garment if provided
        if lower_garment_image is not None:
            lower_img = Image.open(io.BytesIO(await lower_garment_image.read()))
            
            # Preprocess lower garment
            lower_img, lower_metadata = image_preprocessor.preprocess(lower_img)
            preprocessing_info['lower_garment'] = lower_metadata
            print(f"Lower garment preprocessed: {lower_metadata.get('resized', False)}, "
                  f"size: {lower_metadata.get('final_size')}")
            
            current_result = predictor.process_virtual_tryon(
                person_image=current_result,
                garment_image=lower_img,
                category="bottoms",
                garment_photo_type=garment_photo_type,
                num_timesteps=num_timesteps,
                guidance_scale=guidance_scale,
                seed=seed,
                segmentation_free=segmentation_free,
            )
            
            processing_steps.append("lower_garment")
        
        # Return response
        response = {
            "success": True,
            "generated_image": image_to_base64(current_result),
            "processing_steps": processing_steps,
            "preprocessing": preprocessing_info,
            "parameters": {
                "garment_photo_type": garment_photo_type,
                "num_timesteps": num_timesteps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "segmentation_free": segmentation_free,
                "garments_processed": len(processing_steps)
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
