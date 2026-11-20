"""GPU model loader and exclusive inference executor."""
import asyncio
import base64
import io
import os
import torch
from app.core.config import WEIGHTS_DIR

predictor = None
_gpu_lock = asyncio.Lock()

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


