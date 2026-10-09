"""Prompt builders for community reporting and retrieval modes."""

COMMUNITY_SUMMARY_SYSTEM_PROMPT = """You are a knowledge-graph analyst. Summarize the supplied community using only the evidence provided. Cover its main theme, key entities, important relationships, and notable patterns. If finer-grained child community reports are provided, use them as supporting evidence and reconcile overlaps. Do not invent facts. Keep the report concise and information-dense."""


def create_community_summary_prompt(
    community_id: str,
    entities: list,
    relations: list,
    level: int = 0,
    child_reports: list | None = None,
) -> str:
    entity_lines = [
        f"- {item.get('name', 'Unknown')} ({item.get('type', 'Unknown')}): {item.get('description', '')}"
        for item in entities[:30]
    ]
    relation_lines = [
        f"- {item.get('source')} --[{item.get('relation_type', 'RELATED_TO')}]--> {item.get('target')}: {item.get('description', '')}"
        for item in relations[:40]
    ]
    child_lines = [
        f"- [{item.get('id')}] {item.get('summary', '')}"
        for item in (child_reports or [])
        if item.get("summary")
    ]
    child_section = (
        "\n\nFiner-grained child community reports:\n" + "\n".join(child_lines)
        if child_lines
        else ""
    )
    return f"""{COMMUNITY_SUMMARY_SYSTEM_PROMPT}

Community: {community_id}
Level: {level}
Entities:
{chr(10).join(entity_lines) or '- none'}

Relationships:
{chr(10).join(relation_lines) or '- none'}{child_section}

Write the community report."""


def create_local_search_prompt(
    query: str,
    entities: list,
    relations: list,
    text_units: list | None = None,
    claims: list | None = None,
) -> str:
    entity_text = "\n".join(
        f"- {item.get('name')} ({item.get('type')}): {item.get('description', '')}"
        for item in entities
    ) or "- none"
    relation_text = "\n".join(
        f"- {item.get('source')} --[{item.get('relation_type')}]--> {item.get('target')}: {item.get('description', '')}"
        for item in relations
    ) or "- none"
    text_unit_text = "\n".join(
        f"- [{item.get('id')}] {item.get('text', '')}"
        for item in (text_units or [])
    ) or "- none"
    claim_text = "\n".join(
        (
            f"- [{item.get('id')}] status={item.get('status')} "
            f"{item.get('subject_id')} -> {item.get('object_id')}: "
            f"{item.get('description', '')}"
        )
        for item in (claims or [])
    ) or "- none"
    return f"""Answer the user question using only the supplied GraphRAG context. If the evidence is insufficient, say so explicitly.

Entities:
{entity_text}

Relationships:
{relation_text}

Source text units:
{text_unit_text}

Claims / covariates:
{claim_text}

Question: {query}

Answer:"""


def create_global_map_prompt(query: str, community_reports: list) -> str:
    reports = "\n\n".join(
        f"[{item.get('id')}] {item.get('summary', '')}"
        for item in community_reports
    ) or "No community reports available."
    return f"""Evaluate the community reports for the user question. Extract only evidence relevant to the question. Return valid JSON with this exact shape:

{{
  "points": [
    {{
      "description": "evidence-backed point",
      "score": 0
    }}
  ]
}}

Score each point from 0 to 100 by importance for answering the question. Do not use outside knowledge. Omit irrelevant points.

Community reports:
{reports}

Question: {query}
"""


def create_global_reduce_prompt(query: str, points: list) -> str:
    ranked_points = "\n".join(
        f"- score={item.get('score', 0)}: {item.get('description', '')}"
        for item in points
    ) or "- no relevant evidence points"
    return f"""Synthesize the final answer from the ranked intermediate evidence points below. Use only the supplied points, reconcile duplicates, and state when evidence is insufficient. Do not add outside facts.

Ranked evidence:
{ranked_points}

Question: {query}

Answer:"""


def create_basic_search_prompt(query: str, text_units: list) -> str:
    context = "\n\n".join(
        f"[{item.get('id')}] {item.get('text', '')}"
        for item in text_units
    ) or "No text units available."
    return f"""Answer the question using only these retrieved source passages. If the answer is not present, say that the indexed evidence is insufficient.

Passages:
{context}

Question: {query}

Answer:"""
