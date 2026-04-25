"""
Q2 — Multi-Provider Content Generation Router
Routes image / video / voice requests to the correct AI provider.

Providers (swappable via env):
  Images  → Stability AI  (or DALL·E 3)
  Videos  → RunwayML      (or Kling)
  Voice   → ElevenLabs    (or OpenAI TTS)

FastAPI app — run with:
    uvicorn content_router:app --reload
"""

import os
import uuid
from enum import Enum
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="KeaBuilder Content Router")


# ── enums & schemas ───────────────────────────────────────────────────────────

class ContentType(str, Enum):
    image = "image"
    video = "video"
    voice = "voice"


class ContentRequest(BaseModel):
    type: ContentType
    prompt: str
    user_id: str
    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    options: dict = {}


class ContentResponse(BaseModel):
    asset_id: str
    type: ContentType
    provider: str
    output_url: str
    status: str = "completed"
    metadata: dict = {}


# ── provider clients ──────────────────────────────────────────────────────────

class StabilityClient:
    """Stability AI — image generation."""
    BASE = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    HEADERS = {"Authorization": f"Bearer {os.getenv('STABILITY_API_KEY', '')}"}

    async def generate(self, prompt: str, options: dict) -> dict:
        payload = {
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": options.get("cfg_scale", 7),
            "height": options.get("height", 1024),
            "width": options.get("width", 1024),
            "samples": 1,
        }
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(self.BASE, headers=self.HEADERS, json=payload)
            r.raise_for_status()
            b64 = r.json()["artifacts"][0]["base64"]
            # In production: upload b64 to S3 / GCS and return the public URL.
            return {"url": f"https://cdn.keabuilder.io/images/{uuid.uuid4()}.png", "b64": b64[:20] + "…"}


class RunwayClient:
    """RunwayML — video generation (Gen-3 Alpha)."""
    BASE = "https://api.dev.runwayml.com/v1/image_to_video"
    HEADERS = {
        "Authorization": f"Bearer {os.getenv('RUNWAY_API_KEY', '')}",
        "X-Runway-Version": "2024-11-06",
    }

    async def generate(self, prompt: str, options: dict) -> dict:
        payload = {
            "promptText": prompt,
            "model": "gen3a_turbo",
            "duration": options.get("duration", 5),
            "ratio": options.get("ratio", "1280:768"),
        }
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(self.BASE, headers=self.HEADERS, json=payload)
            r.raise_for_status()
            task_id = r.json()["id"]
            return {"url": f"https://cdn.keabuilder.io/videos/{task_id}.mp4", "task_id": task_id}


class ElevenLabsClient:
    """ElevenLabs — text-to-speech."""
    BASE = "https://api.elevenlabs.io/v1/text-to-speech"
    HEADERS = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")}

    async def generate(self, prompt: str, options: dict) -> dict:
        voice_id = options.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        payload = {
            "text": prompt,
            "model_id": options.get("model_id", "eleven_multilingual_v2"),
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self.BASE}/{voice_id}", headers=self.HEADERS, json=payload)
            r.raise_for_status()
            audio_id = str(uuid.uuid4())
            return {"url": f"https://cdn.keabuilder.io/audio/{audio_id}.mp3"}


# ── router logic ──────────────────────────────────────────────────────────────

PROVIDERS: dict[ContentType, tuple[Any, str]] = {
    ContentType.image: (StabilityClient(), "stability-ai"),
    ContentType.video: (RunwayClient(),    "runwayml"),
    ContentType.voice: (ElevenLabsClient(), "elevenlabs"),
}


@app.post("/generate", response_model=ContentResponse)
async def generate_content(req: ContentRequest):
    client, provider_name = PROVIDERS.get(req.type, (None, None))
    if client is None:
        raise HTTPException(status_code=400, detail=f"Unknown content type: {req.type}")

    try:
        result = await client.generate(req.prompt, req.options)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"{provider_name} error: {exc.response.text}")

    # Persist asset record to DB (pseudo-code):
    # await db.assets.insert({asset_id, user_id, type, url, provider, created_at})

    return ContentResponse(
        asset_id=req.asset_id,
        type=req.type,
        provider=provider_name,
        output_url=result["url"],
        metadata={k: v for k, v in result.items() if k != "url"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
