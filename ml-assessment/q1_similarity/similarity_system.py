"""
ML Q1 — Text / Vector Similarity System for KeaBuilder

Finds the most similar stored input to a given query using cosine similarity.
Works with raw text (auto-embedded) or pre-computed vectors.

Dependencies:
    pip install numpy sentence-transformers
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # lightweight, 384-dim


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


@dataclass
class SimilarityStore:
    model: SentenceTransformer = field(default_factory=lambda: SentenceTransformer(MODEL_NAME))
    _texts: list[str] = field(default_factory=list)
    _vectors: list[np.ndarray] = field(default_factory=list)

    def add(self, texts: list[str]) -> None:
        for text in texts:
            emb = self.model.encode(text, normalize_embeddings=True)
            self._texts.append(text)
            self._vectors.append(emb)

    def find_top_match(self, query: str, top_k: int = 1) -> list[dict]:
        if not self._vectors:
            return []
        q_emb = self.model.encode(query, normalize_embeddings=True)
        scored = [
            {"text": t, "score": round(cosine_similarity(q_emb, v), 4)}
            for t, v in zip(self._texts, self._vectors)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ── demo ──────────────────────────────────────────────────────────────────────

SAMPLE_INPUTS = [
    "I want to build a sales funnel for my coaching business",
    "How do I capture leads with a landing page?",
    "Automate follow-up emails after webinar sign-up",
    "Create an AI chatbot for my website",
    "Generate social media content using AI",
]


def main():
    store = SimilarityStore()
    store.add(SAMPLE_INPUTS)

    test_queries = [
        "Set up automated email sequences for new leads",
        "Build a page to collect emails for my online course",
    ]

    results = {}
    for query in test_queries:
        top = store.find_top_match(query, top_k=2)
        results[query] = top
        print(f"\nQuery : {query}")
        for r in top:
            print(f"  ↳ [{r['score']:.4f}] {r['text']}")

    with open("similarity_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results written to similarity_output.json")


if __name__ == "__main__":
    main()
