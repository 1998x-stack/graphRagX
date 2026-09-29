"""Prompt builders for DRIFT primer, follow-up exploration, and reduce."""

def create_drift_primer_prompt(query: str, community_reports: list) -> str:
    reports = "\n\n".join(
        f"[{item.get('id')}] {item.get('summary', '')}"
        for item in community_reports
    ) or "No community reports available."
    return f"""Use the relevant community reports to create a broad primer for the user query.
Return valid JSON with exactly this shape:

{{
  "answer": "broad evidence-grounded primer answer",
  "follow_ups": [
    {{
      "question": "specific follow-up question that can be investigated locally",
      "score": 0
    }}
  ]
}}

Requirements:
- Use only the supplied reports.
- Follow-up questions must narrow uncertainty or inspect specific entities/relationships.
- Score follow-ups from 0 to 100 by expected value for improving the final answer.
- Avoid duplicate or merely rephrased questions.

Community reports:
{reports}

User query: {query}
"""


def create_drift_followup_prompt(
    root_query: str,
    followup_query: str,
    entities: list,
    relations: list,
    claims: list,
    text_units: list,
    prior_evidence: list,
) -> str:
    entity_text = "\n".join(
        f"- {item.get('name')} ({item.get('type')}): {item.get('description', '')}"
        for item in entities
    ) or "- none"
    relation_text = "\n".join(
        f"- {item.get('source')} --[{item.get('relation_type')}]--> {item.get('target')}: {item.get('description', '')}"
        for item in relations
    ) or "- none"
    text_text = "\n".join(
        f"- [{item.get('id')}] {item.get('text', '')}"
        for item in text_units
    ) or "- none"
    claim_text = "\n".join(
        (
            f"- [{item.get('id')}] status={item.get('status')} "
            f"{item.get('subject_id')} -> {item.get('object_id')}: "
            f"{item.get('description', '')}"
        )
        for item in claims
    ) or "- none"
    prior_text = "\n".join(
        f"- {item.get('question')}: {item.get('answer')}"
        for item in prior_evidence[-6:]
    ) or "- none"

    return f"""Investigate the follow-up question using only the supplied local GraphRAG evidence.
Return valid JSON with exactly this shape:

{{
  "answer": "evidence-grounded intermediate answer",
  "confidence": 0.0,
  "follow_ups": [
    {{
      "question": "next specific question worth investigating",
      "score": 0
    }}
  ]
}}

Requirements:
- confidence must be between 0 and 1.
- New follow-ups must add information rather than repeat the root or prior questions.
- If the evidence is weak, lower confidence and return fewer or no follow-ups.
- Do not use outside knowledge.

Root query: {root_query}
Current follow-up: {followup_query}

Entities:
{entity_text}

Relationships:
{relation_text}

Source text units:
{text_text}

Claims / covariates:
{claim_text}

Prior evidence:
{prior_text}
"""


def create_drift_reduce_prompt(
    query: str,
    primer_answer: str,
    evidence: list,
) -> str:
    evidence_text = "\n\n".join(
        (
            f"[{item.get('id')}] depth={item.get('depth')} "
            f"confidence={item.get('confidence')}\n"
            f"Question: {item.get('question')}\n"
            f"Answer: {item.get('answer')}\n"
            f"Sources: {', '.join(item.get('source_ids', []))}"
        )
        for item in evidence
    ) or "No local evidence was gathered."

    return f"""Produce the final answer to the original query from the DRIFT investigation.
Use only the primer and local evidence below. Reconcile conflicts, prefer higher-confidence evidence,
and explicitly state when the evidence does not support a conclusion.

Original query: {query}

Primer:
{primer_answer}

Investigation evidence:
{evidence_text}

Final answer:
"""
