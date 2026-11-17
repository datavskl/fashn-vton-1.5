"""Backend configuration. Values are read once during app startup."""
import os

CATEGORIES = {"tops", "bottoms", "one-pieces"}
PHOTO_TYPES = {"model", "flat-lay"}
MAX_UPLOAD_BYTES = int(os.getenv("VTON_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("VTON_MAX_IMAGE_PIXELS", "20000000"))
QUEUE_SIZE = int(os.getenv("VTON_QUEUE_SIZE", "20"))
JOB_TTL_SECONDS = int(os.getenv("VTON_JOB_TTL_SECONDS", "86400"))
API_TOKEN = os.getenv("VTON_API_TOKEN", "")
ALLOWED_ORIGINS = [value.strip() for value in os.getenv("VTON_CORS_ORIGINS", "").split(",") if value.strip()]
WEIGHTS_DIR = os.getenv("VTON_WEIGHTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "weights"))

