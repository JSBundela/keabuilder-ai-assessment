"""
Q1 — Lead Classification & Intelligent Response System
Runs end-to-end: classify → generate personalised reply → handle edge cases.

Dependencies:
    pip install anthropic python-dotenv
"""

import json
import os
import re
from typing import Any

import anthropic
from dotenv import load_dotenv
from prompts import (
    CLASSIFICATION_PROMPT,
    INCOMPLETE_INPUT_HANDLER_PROMPT,
    RESPONSE_GENERATION_PROMPT,
)

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "gpt-4o"


# ── helpers ──────────────────────────────────────────────────────────────────

def _chat(prompt: str, max_tokens: int = 512) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {raw}")
    return json.loads(match.group())


def is_too_vague(lead: dict) -> bool:
    """True when the lead message is extremely short or key fields are blank."""
    message = lead.get("message", "")
    return len(message.split()) < 4 or not message.strip()


# ── core pipeline ─────────────────────────────────────────────────────────────

def classify_lead(lead: dict) -> dict:
    prompt = CLASSIFICATION_PROMPT.format(lead_data=json.dumps(lead, indent=2))
    raw = _chat(prompt)
    return _extract_json(raw)


def generate_response(lead: dict, classification: dict) -> str:
    name = lead.get("name", "")
    industry = lead.get("industry", "not specified")
    prompt = RESPONSE_GENERATION_PROMPT.format(
        name=name,
        message=lead.get("message", ""),
        classification=classification["classification"],
        industry=industry,
    )
    return _chat(prompt, max_tokens=200)


def handle_incomplete(raw_input: str) -> dict:
    prompt = INCOMPLETE_INPUT_HANDLER_PROMPT.format(raw_input=raw_input)
    raw = _chat(prompt)
    return _extract_json(raw)


def process_lead(lead: dict) -> dict[str, Any]:
    """Full pipeline: classify → respond → handle gaps."""
    if is_too_vague(lead):
        clarification = handle_incomplete(lead.get("message", ""))
        return {
            "status": "needs_clarification",
            "lead": lead,
            "clarification": clarification,
        }

    classification = classify_lead(lead)
    reply = generate_response(lead, classification)

    return {
        "status": "processed",
        "lead": lead,
        "classification": classification,
        "response": reply,
    }


# ── demo ──────────────────────────────────────────────────────────────────────

SAMPLE_LEADS = [
    {
        "name": "Sarah Mitchell",
        "email": "sarah@scalehq.io",
        "phone": "+1-555-0192",
        "company": "ScaleHQ",
        "employees": "50-200",
        "message": "We're evaluating funnel builders for our Q3 product launch. Budget is approved (~$500/mo). Need to decide by Friday.",
        "industry": "SaaS",
        "source": "pricing_page",
    },
    {
        "name": "Raj Patel",
        "email": "raj@consultingco.in",
        "phone": "",
        "company": "ConsultingCo",
        "employees": "1-10",
        "message": "Came across KeaBuilder in a newsletter. Looks interesting. Might explore it sometime.",
        "industry": "Consulting",
        "source": "blog_post",
    },
    {
        "name": "Tom",
        "email": "tom@gmail.com",
        "phone": "",
        "company": "",
        "employees": "",
        "message": "hi",
        "industry": "",
        "source": "homepage",
    },
]


def main():
    results = []
    for lead in SAMPLE_LEADS:
        print(f"\n{'='*60}")
        print(f"Processing: {lead.get('name', 'Unknown')}")
        result = process_lead(lead)
        results.append(result)
        print(json.dumps(result, indent=2))

    with open("sample_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results written to sample_output.json")


if __name__ == "__main__":
    main()
