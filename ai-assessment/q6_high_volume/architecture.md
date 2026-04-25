# Q6 — High-Volume AI Request Architecture for KeaBuilder

## Goal
Handle thousands of concurrent AI generation requests (images, video, voice, LLM) without
degrading latency, driving up cost linearly, or causing reliability incidents.

---

## Architecture Overview

```
Browser / Builder UI
        │  REST / WebSocket
        ▼
  API Gateway (rate-limit per user tier)
        │
        ▼
  Request Service  ──► Redis Queue (priority lanes)
        │                   │
        │            ┌──────┴──────┐
        │            │             │
        │       Worker Pool A  Worker Pool B   (GPU / CPU autoscaled)
        │       (image/video)  (LLM/voice)
        │            │             │
        │            └──────┬──────┘
        │                   │
        ▼                   ▼
  Results DB (Postgres)   Object Storage (S3/GCS)
        │
        ▼
  WebSocket push → Browser (job complete notification)
```

---

## 1 · Async Job Queue (Redis + Celery or BullMQ)

- Every request is enqueued immediately; the user gets a `job_id` in < 100 ms.
- Workers pull jobs from the queue; no direct HTTP calls to AI providers from the API layer.
- Priority queues:  `urgent` (paid tiers) > `normal` > `batch`.

```python
# Pseudo-code — Celery task
@celery.task(bind=True, max_retries=3)
def generate_image(self, job_id, prompt, user_id):
    try:
        result = stability_client.generate(prompt)
        store_result(job_id, result)
        push_notification(user_id, job_id, "done")
    except ProviderError as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

---

## 2 · Caching Layer

| Cache type | What | TTL | Store |
|---|---|---|---|
| Prompt-level | Identical prompts return cached output | 1 h | Redis |
| Embedding cache | Re-use embeddings for similar text | 24 h | Redis |
| CDN | Serve generated media assets | 7 d | CloudFront |

Cost saving: ~30–40 % of image requests from repeat prompts are cache-hits.

---

## 3 · Rate Limiting & Cost Control

- Per-user credits: deduct on enqueue, refund on failure.
- Tier caps enforced at the API Gateway (nginx `limit_req` or Kong).
- Budget alerts: if daily spend > threshold → auto-scale workers down & queue backlog.

---

## 4 · Autoscaling Workers

- GPU workers (for image/video): Kubernetes HPA on queue depth metric.
- Min 2 replicas always warm; scale to 20 on spike; scale-in after 5 min idle.
- Spot/Preemptible instances for batch lane (60–70 % cost reduction).

---

## 5 · Observability

- Emit `job_enqueued`, `job_started`, `job_completed`, `job_failed` events.
- Dashboard: p50/p95/p99 latency by provider, error rate, queue depth.
- Alerts: queue depth > 500 for > 2 min → page on-call.

---

## Performance Targets

| Metric | Target |
|---|---|
| Enqueue latency | < 100 ms |
| Image generation (SDXL) | < 8 s p95 |
| LLM reply (streaming) | First token < 1 s |
| System availability | 99.9 % |
| Cost per 1 000 image requests | < $1.50 |
