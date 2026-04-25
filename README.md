# KeaBuilder AI + ML Assessment — Manvendra Singh Bundela

> **Role:** AI Architect + Machine Learning Engineer — Dream Reflection Media (VARYNT)

---

## Repository Structure

```
keabuilder-ai-assessment/
├── ai-assessment/
│   ├── q1_lead_classification/   # Lead scoring + personalised reply
│   ├── q2_content_routing/       # Multi-provider content router (FastAPI)
│   ├── q3_lora_integration/      # LoRA inference pipeline
│   ├── q4_similarity_search/     # FAISS-based text & image similarity
│   ├── q5_fallback_strategy/     # Circuit-breaker + fallback orchestrator
│   └── q6_high_volume/           # Architecture doc for high-volume AI
└── ml-assessment/
    ├── q1_similarity/            # Cosine similarity system (runnable script)
    ├── q2_model_serving/         # Node.js model gateway
    ├── q3_schema/                # PostgreSQL schema (inputs + predictions)
    ├── q4_slow_responses/        # SSE streaming demo (FastAPI)
    ├── q5_notebook_to_prod/      # Written answer
    └── q6_lora_face/             # Written answer
```

---

## Quick Start

```bash
pip install anthropic python-dotenv fastapi uvicorn httpx tenacity cachetools \
            sentence-transformers faiss-cpu numpy Pillow sse-starlette

cp .env.example .env    # add your ANTHROPIC_API_KEY

# Run lead classifier demo
python ai-assessment/q1_lead_classification/lead_classifier.py

# Run similarity demo
python ml-assessment/q1_similarity/similarity_system.py

# Run SSE streaming demo
uvicorn ml-assessment.q4_slow_responses.streaming_demo:app --reload
# open http://localhost:8000
```

---

---

# AI Engineer Assessment — Answers

---

## Q1 — Lead Classification & Intelligent Response

### a. Classification logic

Each form submission is stringified to JSON and sent to Claude with the
`CLASSIFICATION_PROMPT` (see `q1_lead_classification/prompts.py`).

Scoring rubric the prompt encodes:

| Signal | HOT | WARM | COLD |
|---|---|---|---|
| Budget | Confirmed | Mentioned vaguely | Not mentioned |
| Timeline | < 2 weeks | 1–3 months | "someday" / blank |
| Role | Decision-maker | Influencer | Unknown |
| Intent signals | "ready to buy", pricing page | comparison, research | newsletter, blog |
| Company size | > 20 employees | 5–20 | 1–5 or blank |

### b. Prompts

**Classification prompt** (see `prompts.py → CLASSIFICATION_PROMPT`):
- Instructs the model to return strict JSON: `{classification, confidence, reasoning, missing_fields, follow_up_questions}`.
- Enumerates the exact criteria for HOT/WARM/COLD so outputs are deterministic.

**Response generation prompt** (see `prompts.py → RESPONSE_GENERATION_PROMPT`):
- Passes the lead's name, raw message, detected classification, and industry.
- Constrains length (2–3 sentences), tone-mirroring, one specific reference to their message, and a classification-specific CTA.

### c. Human-feel personalisation

1. **Name injection** — use first name if available.
2. **Tone mirror** — the prompt instructs the model to match formal/casual register.
3. **Specificity anchor** — always reference one concrete detail from the lead's message.
4. **Classification-aware CTA** — HOT gets "book a demo now", COLD gets a case study.
5. **Temperature = 0.7** — enough variation to avoid robotic repetition across leads.

### d. Incomplete / unclear inputs

- A word-count heuristic (`is_too_vague`) catches submissions under 4 words.
- The `INCOMPLETE_INPUT_HANDLER_PROMPT` extracts whatever signal exists and generates one targeted clarifying question.
- Missing structured fields (budget, timeline, role) are listed in `missing_fields` with suggested `follow_up_questions` in the classification JSON.

### Sample input → output (JSON)

See [`ai-assessment/q1_lead_classification/sample_io.json`](ai-assessment/q1_lead_classification/sample_io.json) for three examples:
- HOT lead (confirmed budget, Friday deadline) → immediate demo CTA
- COLD lead (passive discovery) → case study CTA
- Vague lead ("hi") → clarification question, no classification attempted

---

## Q2 — Multi-Provider Content Generation Router

### Routing logic

The `ContentRouter` (FastAPI app in `q2_content_routing/content_router.py`) maps `content_type` to a provider client:

| `type` | Provider | API |
|---|---|---|
| `image` | Stability AI (SDXL) | `/v1/generation/…` |
| `video` | RunwayML Gen-3 | `/v1/image_to_video` |
| `voice` | ElevenLabs | `/v1/text-to-speech/{voice_id}` |

Providers are swappable via environment variables — no code change needed to
switch from Stability → DALL·E 3 or from ElevenLabs → OpenAI TTS.

### Frontend ↔ Backend interaction

```
Builder UI
  │  POST /generate  { type, prompt, user_id, options }
  │  ← 202 Accepted  { job_id }          (async path for video/image)
  │  ← SSE / WebSocket push when ready   { asset_id, output_url }
  │
  ▼
Content Router (FastAPI)
  → validates type, enqueues job, returns job_id immediately
  → Worker calls provider API, uploads result to S3/CDN
  → Notifies client via WebSocket

For voice (fast, < 3 s): synchronous path returns URL directly.
For images (4–10 s) and video (30–120 s): async job + push notification.
```

### Output management

- All outputs are uploaded to a CDN bucket (`s3://keabuilder-assets/outputs/`).
- An `assets` DB record is inserted: `{ asset_id, user_id, type, provider, url, created_at }`.
- The Builder UI references assets by `asset_id`; URLs are pre-signed on demand.

---

## Q3 — LoRA Integration for Personalised Image Generation

### Integration into inference pipeline

Full implementation: [`ai-assessment/q3_lora_integration/lora_pipeline.py`](ai-assessment/q3_lora_integration/lora_pipeline.py)

**Training (one-time, per user):**
1. User uploads 5–15 reference photos inside KeaBuilder.
2. A GPU worker (triggered by a Celery task) runs DreamBooth / LoRA fine-tuning
   on SDXL base (~20 min on A100) with the user's images.
3. The resulting adapter weights (`.safetensors`, ~30 MB) are stored at
   `s3://keabuilder-lora-weights/{user_id}/adapter.safetensors`.

**Inference (per generation request):**
1. API receives `{ user_id, prompt }`.
2. The pipeline checks an in-memory `_pipe_cache`. If miss, downloads the user's
   adapter from S3, loads it into the base SDXL pipeline, and caches the fused pipe.
3. `pipe.load_lora_weights()` + `pipe.fuse_lora(lora_scale=0.9)` merges the adapter
   weights into the UNet — no inference overhead vs. the base model.
4. Image is generated, uploaded to CDN, URL returned.

**Cache invalidation:** `DELETE /cache/{user_id}` evicts the cached pipeline when
the user uploads new reference images and retrains.

### User flow inside KeaBuilder

```
KeaBuilder Dashboard
  → "Brand AI" tab → upload reference photos (5–15)
  → "Train" button → triggers background GPU job (spinner, ~20 min)
  → Job complete notification → "Generate Images" unlocked
  → Prompt input → /generate POST → personalised image in ~6 s
```

---

## Q4 — Face & Text Similarity Search

Full implementation: [`ai-assessment/q4_similarity_search/similarity_search.py`](ai-assessment/q4_similarity_search/similarity_search.py)

### Storage

- **Vectors** stored in PostgreSQL with the `pgvector` extension.
  Column type: `vector(384)` for text, `vector(512)` for CLIP image embeddings.
- **HNSW index** (`CREATE INDEX USING hnsw`) gives sub-millisecond ANN search at
  millions of vectors without leaving the database.
- Raw assets (images, templates) live in S3; only their embeddings are in Postgres.

### Retrieval

```sql
-- Find the 5 most similar templates to a given embedding
SELECT id, name, 1 - (embedding <=> $1) AS similarity
FROM templates
ORDER BY embedding <=> $1
LIMIT 5;
```

### Matching logic

| Use case | Embedding model | Distance |
|---|---|---|
| Text / prompt similarity | `all-MiniLM-L6-v2` (384-dim) | Cosine |
| Image / template similarity | OpenCLIP ViT-B/32 (512-dim) | Cosine |
| Face consistency | InsightFace ArcFace (512-dim) | Cosine |

- Face search uses InsightFace to detect + crop faces before embedding,
  so body background noise does not affect the similarity score.
- A similarity threshold of **0.85** is used for "near-duplicate" detection;
  **0.70** for "similar style" recommendations.

---

## Q5 — Fallback Strategy for AI Service Failures

Full implementation: [`ai-assessment/q5_fallback_strategy/ai_fallback.py`](ai-assessment/q5_fallback_strategy/ai_fallback.py)

### Strategy (layered)

```
Request
  │
  ▼
Primary provider  ──fail──► Retry × 2 (exponential back-off 0.5 s, 2 s)
  │ fail                          │ still fails
  ▼                               ▼
Secondary provider ──fail──► Cached last-good response (TTL 5 min)
  │ fail
  ▼
Graceful placeholder ("request queued") + background re-queue
```

### Circuit Breaker

- Opens after **3 consecutive failures** for a provider.
- Stays open for **30 seconds** (recovery timeout), then transitions to half-open.
- Half-open: allow one probe request; success → close, failure → reopen.
- Prevents cascade: if OpenAI is down, the circuit trips instantly so all requests
  skip it rather than waiting for each timeout.

### UX — no broken experience

- For **streaming** (LLM text): client renders partial stream; on error, inserts
  a soft message: "Connection interrupted — here's what we have so far."
- For **async jobs** (image/video): job stays in queue; user sees "processing"
  badge; retried automatically when provider recovers.
- Status page widget shows a green/amber/red dot per AI service.

---

## Q6 — High-Volume AI Request Architecture

See full architecture doc: [`ai-assessment/q6_high_volume/architecture.md`](ai-assessment/q6_high_volume/architecture.md)

### Summary

| Concern | Solution |
|---|---|
| **Performance** | Async job queue (Redis/Celery); SSE streaming for LLM; warm worker pool |
| **Cost** | Prompt-level cache in Redis (30–40 % cache-hit rate); spot/preemptible GPU workers for batch lane; per-user credit system |
| **Reliability** | Circuit breaker + provider fallback; HPA autoscaling on queue depth; min 2 warm replicas always up |

---

## Q7 — Tools, Frameworks & Platforms

**LLM / AI APIs:** Anthropic Claude (SDK, tool use, streaming, prompt caching), OpenAI GPT-4o, Stability AI SDXL, ElevenLabs, RunwayML

**ML / Deep Learning:** PyTorch, Hugging Face Diffusers & Transformers, sentence-transformers, FAISS, ONNX Runtime, InsightFace, OpenCLIP, scikit-learn

**Backend:** FastAPI, Python asyncio, Celery + Redis, Node.js (Express), PostgreSQL + pgvector, SQLAlchemy

**Infrastructure:** AWS (EC2 G-instances, S3, CloudFront, ECS), Docker, Kubernetes (HPA), GitHub Actions CI/CD

**Frontend (light):** React, Server-Sent Events, WebSockets

---
---

# ML Engineer Assessment — Answers

---

## Q1 — Text Similarity System

Full runnable script: [`ml-assessment/q1_similarity/similarity_system.py`](ml-assessment/q1_similarity/similarity_system.py)

**Approach:** sentence-transformers (`all-MiniLM-L6-v2`) to embed texts → cosine similarity against all stored vectors → return top-k.

**Sample output:**
```json
{
  "Set up automated email sequences for new leads": [
    {"text": "Automate follow-up emails after webinar sign-up", "score": 0.7812},
    {"text": "I want to build a sales funnel for my coaching business", "score": 0.5234}
  ],
  "Build a page to collect emails for my online course": [
    {"text": "How do I capture leads with a landing page?", "score": 0.8341},
    {"text": "I want to build a sales funnel for my coaching business", "score": 0.6129}
  ]
}
```

---

## Q2 — Serving an ML Model in Node.js Production

See [`ml-assessment/q2_model_serving/model_server.js`](ml-assessment/q2_model_serving/model_server.js)

**Recommended architecture:**
- Node.js acts as the **API gateway** — handles auth, rate-limiting, request validation.
- A **Python FastAPI microservice** runs the actual model inference (better GPU/CPU library support, torch, ONNX).
- Node.js proxies to it via HTTP (internal network, no public exposure).
- For **lightweight models** (classifiers, small embedders): use `onnxruntime-node`
  to load `.onnx` models directly in Node — zero Python dependency.

**Production checklist:**
- Health-check endpoint (`/health`) for load-balancer probes.
- Request timeout (10 s) + structured error responses.
- Horizontal scaling: stateless workers behind a load balancer.
- Model loaded once at startup; never reloaded per request.

---

## Q3 — Schema: User Inputs & Predictions

See [`ml-assessment/q3_schema/schema.sql`](ml-assessment/q3_schema/schema.sql)

**Tables:**

`user_inputs` — stores every text submission with its embedding for similarity search.  
Key fields: `id, user_id, input_type, raw_text, embedding (vector 384), metadata JSONB`

`ml_predictions` — one row per inference call, linked to the input.  
Key fields: `id, input_id, model_id, prediction_type, output JSONB, confidence, latency_ms, status`

`prediction_feedback` — optional human corrections for retraining loop.  
Key fields: `prediction_id, user_id, rating, correct_label`

---

## Q4 — Handling Slow ML Responses in UI

**Primary technique: Server-Sent Events (SSE) streaming**  
Stream tokens to the browser as they arrive. First word visible in < 1 s; user
reads while the model is still generating. Full demo:
[`ml-assessment/q4_slow_responses/streaming_demo.py`](ml-assessment/q4_slow_responses/streaming_demo.py)

**Supporting techniques:**
- **Skeleton / shimmer loaders** — show placeholder cards instantly on request.
- **Optimistic UI** — insert a "pending" asset card immediately; replace with real content on completion.
- **Background queue + WebSocket push** — for video/image (30–120 s), the UI shows
  a progress indicator and gets a push notification on completion.
- **Client-side timeout + retry** — abort after 15 s, show retry button with cached suggestion.

---

## Q5 — Three Challenges: Notebook → Production

1. **Dependency & environment drift**  
   A notebook runs on the data-scientist's machine with pinned libraries; production
   uses a different Python version or library versions that behave differently.  
   *Fix:* `requirements.txt` + Docker image built from the same base; lock all versions.

2. **Data distribution shift (train-serve skew)**  
   The model was trained on historical data; live inputs look different (new vocabulary,
   different distributions, missing fields). Accuracy silently degrades.  
   *Fix:* Log live inputs; run periodic distribution checks; set up model monitoring with
   drift alerts (e.g. Evidently AI).

3. **Latency vs. accuracy trade-off**  
   A model accurate in a notebook (batch inference, no timeout) may be too slow for
   real-time API use (< 200 ms budget). Large models, heavy preprocessing, or ONNX
   incompatibilities surface only under production latency constraints.  
   *Fix:* Profile inference time early; consider quantisation, ONNX export, or a lighter
   distilled model for the serving path.

---

## Q6 — LoRA for Face Consistency

**Goal:** generate images that consistently show the same face/brand character
across different prompts and scenes.

**Approach:**

1. **Data collection:** User provides 10–20 photos of the subject (varied angles,
   lighting, expressions). Quality > quantity.

2. **Training:** Fine-tune a LoRA adapter (rank 8–16) on top of SDXL base using
   DreamBooth with a unique token (e.g. `[subject]`). Train for ~1 500–2 000 steps
   on an A100 (~20 min). Use regularisation images to prevent over-fitting.

3. **Inference:** Load the adapter weights into the base pipeline
   (`pipe.load_lora_weights()`), set `lora_scale=0.8–1.0`.
   Always include the trigger token in the prompt: `"a photo of [subject] presenting on stage"`.

4. **Consistency tips:**
   - Fix the seed for reproducible outputs.
   - Use a face-consistency check (ArcFace cosine similarity > 0.85) to auto-reject
     images where the face drifted; re-roll with a different seed.
   - Maintain a library of per-user LoRA adapters in S3; cache loaded pipelines in memory.

---

## Q7 — ML Tools, Frameworks & Platforms

**ML / Training:** PyTorch, Hugging Face (Transformers, Diffusers, PEFT/LoRA), scikit-learn, XGBoost, sentence-transformers, OpenCLIP, InsightFace

**Model Serving:** FastAPI, ONNX Runtime, Celery + Redis, FAISS, pgvector

**Data / Experimentation:** pandas, NumPy, Jupyter, DVC, MLflow (experiment tracking)

**Infrastructure:** Docker, Kubernetes, AWS (EC2 GPU instances, S3, SageMaker), GitHub Actions

**Monitoring:** Evidently AI (drift detection), Prometheus + Grafana (latency/error metrics)
