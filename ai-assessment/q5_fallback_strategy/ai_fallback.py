"""
Q5 — AI Service Fallback & Resilience Layer

Pattern: primary → secondary → cached response, with circuit-breaker per provider.
The user always sees a result; degradation is silent and logged.

Dependencies:
    pip install httpx tenacity cachetools
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import httpx
from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ── circuit breaker ───────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    recovery_timeout: float = 30.0      # seconds before attempting again
    _failures: int = 0
    _opened_at: float = 0.0
    _state: str = "closed"              # closed | open | half-open

    def record_success(self):
        self._failures = 0
        self._state = "closed"

    def record_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()
            logger.warning("Circuit %s OPENED after %d failures", self.name, self._failures)

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.monotonic() - self._opened_at > self.recovery_timeout:
                self._state = "half-open"
                logger.info("Circuit %s → half-open", self.name)
                return False
            return True
        return False


# ── provider wrapper ──────────────────────────────────────────────────────────

@dataclass
class AIProvider:
    name: str
    call: Callable[..., Coroutine]
    circuit: CircuitBreaker = field(init=False)

    def __post_init__(self):
        self.circuit = CircuitBreaker(self.name)

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def invoke(self, **kwargs) -> Any:
        if self.circuit.is_open:
            raise RuntimeError(f"Circuit open for {self.name}")
        try:
            result = await self.call(**kwargs)
            self.circuit.record_success()
            return result
        except Exception as exc:
            self.circuit.record_failure()
            raise exc


# ── orchestrator ──────────────────────────────────────────────────────────────

CACHE: TTLCache = TTLCache(maxsize=256, ttl=300)   # 5-min cache of last-good responses


async def call_with_fallback(
    providers: list[AIProvider],
    cache_key: str,
    **kwargs,
) -> dict:
    """
    Try providers in order.  On exhaustion fall back to the cached response.
    The final fallback is a graceful placeholder so the UI never breaks.
    """
    for provider in providers:
        try:
            result = await provider.invoke(**kwargs)
            CACHE[cache_key] = result          # store as fresh cached value
            return {"source": provider.name, "data": result, "degraded": False}
        except Exception as exc:
            logger.error("Provider %s failed: %s", provider.name, exc)

    if cache_key in CACHE:
        logger.warning("All providers failed — serving cached result for %s", cache_key)
        return {"source": "cache", "data": CACHE[cache_key], "degraded": True}

    # Ultimate fallback: deterministic placeholder so UI renders something.
    return {
        "source": "fallback",
        "data": {"message": "Our AI service is temporarily unavailable. Your request has been queued."},
        "degraded": True,
    }


# ── demo: fake providers ──────────────────────────────────────────────────────

async def primary_llm(prompt: str) -> dict:
    raise httpx.TimeoutException("primary timed out")   # simulate failure


async def secondary_llm(prompt: str) -> dict:
    await asyncio.sleep(0.2)
    return {"text": f"[secondary] Reply to: {prompt}"}


async def main():
    p1 = AIProvider(name="openai-gpt4o",  call=primary_llm)
    p2 = AIProvider(name="claude-sonnet", call=secondary_llm)

    result = await call_with_fallback(
        providers=[p1, p2],
        cache_key="lead_reply:user123",
        prompt="How can KeaBuilder help my agency?",
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
