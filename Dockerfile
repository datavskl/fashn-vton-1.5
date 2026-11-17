FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
COPY pyproject.toml README.md requirements.txt ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir -e . && python -m pip install --no-cache-dir fastapi uvicorn[standard] python-multipart httpx
COPY app/ ./app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
