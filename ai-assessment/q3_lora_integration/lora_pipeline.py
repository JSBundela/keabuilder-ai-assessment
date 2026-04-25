"""
Q3 — LoRA Integration for Personalised Image Generation

Architecture:
  1. User uploads 5–15 reference images inside KeaBuilder.
  2. A background job fine-tunes a LoRA adapter (SDXL base) on those images.
  3. At inference time the adapter is merged on-the-fly for that user session.

This file shows the inference side (the training side runs as a one-time job
on a GPU worker and stores the adapter weights to S3).

Dependencies:
    pip install diffusers accelerate torch boto3 fastapi
"""

import io
import os
import uuid

import boto3
import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="KeaBuilder LoRA Image API")

S3 = boto3.client("s3")
BUCKET = os.getenv("LORA_WEIGHTS_BUCKET", "keabuilder-lora-weights")
BASE_MODEL = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")

# In-memory cache so we don't reload weights on every request.
_pipe_cache: dict[str, StableDiffusionXLPipeline] = {}


def _load_pipeline(user_id: str) -> StableDiffusionXLPipeline:
    if user_id in _pipe_cache:
        return _pipe_cache[user_id]

    local_path = f"/tmp/lora_{user_id}.safetensors"

    # Download LoRA adapter from S3 (each user has their own object).
    try:
        S3.download_file(BUCKET, f"{user_id}/adapter.safetensors", local_path)
    except S3.exceptions.NoSuchKey:
        raise HTTPException(404, detail="LoRA adapter not found. Train the model first.")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    # Load and fuse the user-specific LoRA weights.
    pipe.load_lora_weights(local_path)
    pipe.fuse_lora(lora_scale=0.9)

    _pipe_cache[user_id] = pipe
    return pipe


# ── API ───────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    user_id: str
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted face"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None


class GenerateResponse(BaseModel):
    image_url: str
    user_id: str
    prompt: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    pipe = _load_pipeline(req.user_id)

    generator = None
    if req.seed is not None:
        generator = torch.Generator("cuda").manual_seed(req.seed)

    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    ).images[0]

    # Upload result to S3 / CDN.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    key = f"outputs/{req.user_id}/{uuid.uuid4()}.png"
    S3.upload_fileobj(buf, BUCKET, key, ExtraArgs={"ContentType": "image/png"})

    url = f"https://cdn.keabuilder.io/{key}"
    return GenerateResponse(image_url=url, user_id=req.user_id, prompt=req.prompt)


@app.delete("/cache/{user_id}")
def evict_cache(user_id: str):
    """Call when a user uploads new reference images and retrains."""
    _pipe_cache.pop(user_id, None)
    return {"evicted": user_id}
