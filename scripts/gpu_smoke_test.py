"""On-host real-GPU E2E smoke test and latency / VRAM recorder.

Requires the deployed API, person photograph, and at least one garment.
No model mocking occurs. Does not transmit photographs anywhere except --api-url.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import httpx
from PIL import Image
from io import BytesIO


def image_mime(path):
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp"}.get(path.suffix.lower(), "application/octet-stream")


def gpu_memory_mib():
    """Report max used-memory MiB across visible NVIDIA GPUs; None if unavailable."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        values = [int(v.strip()) for v in output.splitlines() if v.strip()]
        return max(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def run_case(client, url, token, person, garment_fields, case, timeout, outdir):
    observation = []
    stop = threading.Event()

    def observe():
        while not stop.wait(0.5):
            observation.append(gpu_memory_mib())

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    start = time.monotonic()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with person.open("rb") as person_file:
            opened = [(name, path.open("rb")) for name, path in garment_fields.items()]
            try:
                payload = {"person_image": (person.name, person_file, image_mime(person))}
                for (name, path), (_, stream) in zip(garment_fields.items(), opened):
                    mime = image_mime(path)
                    payload[name] = (path.name, stream, mime)
                response = client.post(f"{url}/api/v1/jobs", files=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            finally:
                for _, stream in opened:
                    stream.close()
        job_id = data["job_id"]
        while True:
            result = client.get(f"{url}/api/v1/jobs/{job_id}", headers=headers)
            result.raise_for_status()
            job = result.json()
            if job["status"] in {"completed", "failed"}:
                break
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f"Job {job_id} exceeded {timeout} seconds")
            time.sleep(2)
        elapsed = round(time.monotonic() - start, 2)
        if job["status"] != "completed":
            raise RuntimeError(f"{case} inference failed: {job.get('error')}")
        image_bytes = base64.b64decode(job["generated_image"], validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            size = list(image.size)
            format_name = image.format
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{case}.png").write_bytes(image_bytes)
        samples = [m for m in observation if m is not None]
        return {"case": case, "status": "passed", "latency_seconds": elapsed,
                "output_size": size, "format": format_name, "job_id": job_id,
                "observed_peak_gpu_used_mib": max(samples) if samples else None}
    finally:
        stop.set()
        observer.join(timeout=2)
        samples = [m for m in observation if m is not None]
        if samples:
            print(f"{case}: observed peak GPU used memory {max(samples)} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--person", type=Path, required=True)
    parser.add_argument("--top", type=Path)
    parser.add_argument("--bottom", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("test_outputs/gpu-smoke"))
    parser.add_argument("--timeout", type=float, default=360)
    args = parser.parse_args()
    token = os.environ.get("VTON_API_TOKEN")
    if not token:
        parser.error("VTON_API_TOKEN is required")
    if not args.top and not args.bottom:
        parser.error("Provide --top and/or --bottom")
    for path in (args.person, args.top, args.bottom):
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
        if path is not None and image_mime(path) == "application/octet-stream":
            parser.error(f"Unsupported image extension: {path}")
    cases = []
    if args.top:
        cases.append(("top", {"garment_image": args.top}))
    if args.bottom:
        cases.append(("bottom", {"garment_image": args.bottom}))
    if args.top and args.bottom:
        cases.append(("combined", {
            "upper_garment_image": args.top, "lower_garment_image": args.bottom,
        }))
    with httpx.Client(timeout=30) as client:
        health = client.get(args.api_url.rstrip("/") + "/health")
        health.raise_for_status()
        print("Server health:", json.dumps(health.json()))
        metrics = []
        for name, garments in cases:
            report = run_case(client, args.api_url.rstrip("/"), token, args.person,
                              garments, name, args.timeout, args.output_dir)
            metrics.append(report)
            print(json.dumps(report))
    report_file = args.output_dir / "metrics.json"
    report_file.write_text(json.dumps(metrics, indent=2))
    print(f"Saved test images and metrics to {args.output_dir} (keep private)")


if __name__ == "__main__":
    main()
