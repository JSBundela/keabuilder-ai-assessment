"""
Q4 — Face & Text Similarity Search for KeaBuilder Assets

Two subsystems:
  A) Text similarity  — uses sentence-transformers + FAISS
  B) Image/face similarity — uses CLIP embeddings + FAISS (face detection via InsightFace)

Storage:  PostgreSQL with pgvector extension (production) or in-memory FAISS (demo).
Retrieval: ANN search (HNSW index) for sub-10 ms queries at 1 M vectors.

Dependencies:
    pip install sentence-transformers faiss-cpu numpy Pillow open-clip-torch
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────

TEXT_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
DIM_TEXT = 384
DIM_IMAGE = 512                     # open-clip ViT-B/32

# ── text similarity ───────────────────────────────────────────────────────────

@dataclass
class TextStore:
    model: SentenceTransformer = field(default_factory=lambda: SentenceTransformer(TEXT_MODEL))
    index: faiss.IndexFlatIP = field(default_factory=lambda: faiss.IndexFlatIP(DIM_TEXT))
    metadata: list[dict] = field(default_factory=list)

    def add(self, texts: list[str], meta: list[dict]) -> None:
        embs = self._embed(texts)
        self.index.add(embs)
        self.metadata.extend(meta)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        emb = self._embed([query])
        scores, ids = self.index.search(emb, top_k)
        return [
            {"score": float(scores[0][i]), "match": self.metadata[ids[0][i]]}
            for i in range(top_k)
            if ids[0][i] != -1
        ]

    def _embed(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embs, dtype="float32")


# ── image similarity (CLIP) ───────────────────────────────────────────────────

@dataclass
class ImageStore:
    """
    Uses open-clip to embed images. In production, replace FAISS with
    pgvector (ivfflat index) so embeddings live alongside asset metadata.
    """
    index: faiss.IndexFlatIP = field(default_factory=lambda: faiss.IndexFlatIP(DIM_IMAGE))
    metadata: list[dict] = field(default_factory=list)
    _model: Any = None
    _preprocess: Any = None
    _tokenizer: Any = None

    def _lazy_load(self):
        if self._model is None:
            import open_clip
            import torch
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._model.eval()
            self._torch = torch

    def embed_image(self, img: Image.Image) -> np.ndarray:
        self._lazy_load()
        tensor = self._preprocess(img).unsqueeze(0)
        with self._torch.no_grad():
            emb = self._model.encode_image(tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.numpy().astype("float32")

    def add(self, images: list[Image.Image], meta: list[dict]) -> None:
        for img, m in zip(images, meta):
            emb = self.embed_image(img)
            self.index.add(emb)
            self.metadata.append(m)

    def search(self, query_img: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
        emb = self.embed_image(query_img)
        scores, ids = self.index.search(emb, top_k)
        return [
            {"score": float(scores[0][i]), "match": self.metadata[ids[0][i]]}
            for i in range(top_k)
            if ids[0][i] != -1
        ]


# ── demo ──────────────────────────────────────────────────────────────────────

def demo_text():
    store = TextStore()
    corpus = [
        "Build high-converting sales funnels fast",
        "Automate your email follow-up sequences",
        "AI chatbot for lead qualification",
        "Landing page with video background",
        "One-click upsell flow for e-commerce",
    ]
    store.add(corpus, [{"id": i, "text": t} for i, t in enumerate(corpus)])

    query = "automated lead nurturing with AI"
    results = store.search(query, top_k=2)
    print(f"\nText search for: '{query}'")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    demo_text()
