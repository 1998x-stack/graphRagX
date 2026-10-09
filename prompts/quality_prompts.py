"""Prompts for graph quality augmentation."""


def create_entity_description_summary_prompt(
    name: str,
    entity_type: str,
    descriptions: list[str],
) -> str:
    evidence = "\n".join(f"- {item}" for item in descriptions)
    return f"""Consolidate multiple descriptions of the same knowledge-graph entity.
Use only the supplied descriptions. Preserve complementary facts, remove duplication,
and do not resolve contradictions by inventing information.

Entity: {name}
Type: {entity_type}
Description mentions:
{evidence}

Return one concise evidence-grounded description."""


def create_relation_description_summary_prompt(
    source: str,
    relation_type: str,
    target: str,
    descriptions: list[str],
) -> str:
    evidence = "\n".join(f"- {item}" for item in descriptions)
    return f"""Consolidate multiple descriptions of the same directed typed relationship.
Use only supplied evidence, preserve complementary facts, and do not invent facts.

Relationship: {source} --[{relation_type}]--> {target}
Description mentions:
{evidence}

Return one concise evidence-grounded relationship description."""


def create_claim_extraction_prompt(
    text: str,
    claim_description: str,
    entity_names: list[str],
    max_claims: int,
) -> str:
    entities = ", ".join(entity_names) or "none"
    return f"""Extract claims from the source text that match this description:
{claim_description}

Only extract claims whose subject is one of the indexed entities listed below.
Object may be another indexed entity or null. Do not infer unsupported claims.

Indexed entities:
{entities}

Return valid JSON exactly in this shape:
{{
  "claims": [
    {{
      "type": "claim category",
      "description": "concise factual claim",
      "subject": "indexed entity name",
      "object": "indexed entity name or null",
      "status": "TRUE|FALSE|SUSPECTED|UNKNOWN",
      "start_date": null,
      "end_date": null,
      "source_text": "short supporting source span"
    }}
  ]
}}

Return at most {max_claims} claims.

Source text:
{text}
"""
