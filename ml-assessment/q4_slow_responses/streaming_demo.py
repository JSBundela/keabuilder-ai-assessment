"""
ML Q4 — Handling Slow ML Responses in the UI

Best single approach: **Server-Sent Events (SSE) streaming**
Stream tokens/chunks to the browser as they arrive → user sees progress immediately.
Perceived latency drops from "8 seconds blank screen" to "first words in < 1 s".

Secondary techniques (all used together in KeaBuilder):
  1. Skeleton / shimmer loaders while the request is in-flight.
  2. Optimistic UI — show a placeholder asset card immediately.
  3. Background job + WebSocket push for long-running tasks (video, LoRA training).
  4. Request timeout + retry with exponential back-off on the client.

This file demonstrates SSE streaming with FastAPI + Anthropic streaming API.

Dependencies:
    pip install fastapi uvicorn anthropic python-dotenv sse-starlette
"""

import os

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

load_dotenv()

app = FastAPI()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@app.get("/stream")
async def stream_reply(prompt: str = "Write a short KeaBuilder funnel description."):
    """
    Streams Claude tokens back to the browser via SSE.
    The browser renders each chunk as it arrives — no blank-screen wait.
    """

    async def generator():
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text_chunk in stream.text_stream:
                yield {"data": text_chunk}
        yield {"data": "[DONE]"}

    return EventSourceResponse(generator())


# Minimal browser demo page
@app.get("/", response_class=HTMLResponse)
async def demo_page():
    return """
<!DOCTYPE html><html><body>
<h3>KeaBuilder AI — SSE Streaming Demo</h3>
<div id="out" style="white-space:pre-wrap;font-family:monospace;padding:1rem;
     background:#f5f5f5;min-height:100px;border-radius:6px">Waiting…</div>
<script>
  const out = document.getElementById('out');
  out.textContent = '';
  const es = new EventSource('/stream?prompt=Describe+a+high-converting+lead+funnel+in+3+sentences');
  es.onmessage = e => {
    if (e.data === '[DONE]') { es.close(); return; }
    out.textContent += e.data;
  };
</script>
</body></html>
"""
