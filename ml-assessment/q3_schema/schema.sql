-- ML Q3 — Database Schema: User Inputs & ML Predictions
-- Database: PostgreSQL (with pgvector for embedding columns)

-- Extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── user_inputs ───────────────────────────────────────────────────────────────
-- Every piece of text / prompt / form submission a user sends to an ML feature.

CREATE TABLE user_inputs (
    id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       UUID        NOT NULL,
    session_id    UUID,
    input_type    TEXT        NOT NULL,          -- 'lead_form' | 'chat_message' | 'image_prompt' | 'voice_prompt'
    raw_text      TEXT        NOT NULL,
    embedding     vector(384),                   -- sentence-transformer embedding for similarity search
    metadata      JSONB       DEFAULT '{}',      -- e.g. {"source": "pricing_page", "funnel_id": "..."}
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON user_inputs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON user_inputs (user_id, created_at DESC);


-- ── ml_predictions ────────────────────────────────────────────────────────────
-- Every inference result produced by an ML model for a given input.

CREATE TABLE ml_predictions (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_id        UUID        NOT NULL REFERENCES user_inputs(id) ON DELETE CASCADE,
    model_id        TEXT        NOT NULL,        -- e.g. 'lead-classifier-v2', 'claude-sonnet-4-6'
    model_version   TEXT,
    prediction_type TEXT        NOT NULL,        -- 'classification' | 'generation' | 'embedding' | 'ranking'
    output          JSONB       NOT NULL,        -- raw model output (label, score, text, url, …)
    confidence      FLOAT,                       -- 0.0 – 1.0 where applicable
    latency_ms      INTEGER,                     -- inference latency
    token_count     INTEGER,                     -- for LLM predictions
    provider        TEXT,                        -- 'anthropic' | 'openai' | 'stability' | 'internal'
    status          TEXT        NOT NULL DEFAULT 'completed',  -- 'pending' | 'completed' | 'failed'
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON ml_predictions (input_id);
CREATE INDEX ON ml_predictions (model_id, created_at DESC);
CREATE INDEX ON ml_predictions (status) WHERE status = 'failed';


-- ── feedback (optional but valuable for model retraining) ────────────────────

CREATE TABLE prediction_feedback (
    id             UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id  UUID        NOT NULL REFERENCES ml_predictions(id),
    user_id        UUID        NOT NULL,
    rating         SMALLINT    CHECK (rating BETWEEN 1 AND 5),
    correct_label  TEXT,                         -- human-corrected label for supervised retraining
    comment        TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
