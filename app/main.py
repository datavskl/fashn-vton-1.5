"""Single production ASGI entry point: uvicorn app.main:app --workers 1."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from app.api.routes import router
from app.core.config import ALLOWED_ORIGINS
from app.services import inference
from app.services.jobs import lifespan

app = FastAPI(title="ShriVastra FASHN VTON API", version="1.5.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "running", "gpu_available": torch.cuda.is_available(),
            "model_loaded": inference.predictor is not None}

@app.get("/")
async def root():
    return {"name": "ShriVastra FASHN VTON API", "docs": "/docs", "health": "/health"}
