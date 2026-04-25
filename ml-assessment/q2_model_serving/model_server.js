/**
 * ML Q2 — Serving an ML Model from a Node.js Backend
 *
 * Recommended pattern: Node.js acts as the API gateway; the model runs in a
 * dedicated Python microservice (FastAPI) or via ONNX Runtime Node.js bindings.
 *
 * This file shows BOTH approaches:
 *   Option A) Proxy to a Python FastAPI inference service  ← recommended for heavy models
 *   Option B) Run ONNX model directly in Node.js           ← for lightweight models
 */

const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json());

// ── Option A: Proxy to Python inference microservice ─────────────────────────

const INFERENCE_SERVICE_URL =
  process.env.INFERENCE_SERVICE_URL || "http://ml-service:8000";

app.post("/api/predict", async (req, res) => {
  const { inputs, model_id } = req.body;

  if (!inputs) {
    return res.status(400).json({ error: "inputs required" });
  }

  try {
    const { data } = await axios.post(
      `${INFERENCE_SERVICE_URL}/predict`,
      { inputs, model_id },
      { timeout: 10_000 }
    );
    return res.json(data);
  } catch (err) {
    const status = err.response?.status || 502;
    const detail = err.response?.data?.detail || err.message;
    return res.status(status).json({ error: detail });
  }
});

// ── Option B: ONNX Runtime directly in Node.js ───────────────────────────────
// Best for small models (e.g. text classifiers, embedders).
//
// const ort = require("onnxruntime-node");
//
// let session; // loaded once at startup
// async function loadModel() {
//   session = await ort.InferenceSession.create("./model.onnx");
// }
//
// app.post("/api/classify", async (req, res) => {
//   const inputTensor = new ort.Tensor("float32", req.body.vector, [1, 384]);
//   const feeds = { input: inputTensor };
//   const results = await session.run(feeds);
//   const logits = results.logits.data;
//   res.json({ logits: Array.from(logits) });
// });

// ── health check ─────────────────────────────────────────────────────────────

app.get("/health", (_req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Model gateway listening on :${PORT}`));
