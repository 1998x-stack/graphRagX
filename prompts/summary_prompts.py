"""Prompt builders for community reporting and retrieval modes."""

COMMUNITY_SUMMARY_SYSTEM_PROMPT = """You are a knowledge-graph analyst. Summarize the supplied community using only the evidence provided. Cover its main theme, key entities, important relationships, and notable patterns. Do not invent facts. Keep the report concise and information-dense."""


def create_community_summary_prompt(community_id: str, entities: list, relations: list, level: int = 0) -> str:
    entity_lines = [
        f"- {item.get('name', 'Unknown')} ({item.get('type', 'Unknown')}): {item.get('description', '')}"
        for item in entities[:30]
    ]
    relation_lines = [
        f"- {item.get('source')} --[{item.get('relation_type', 'RELATED_TO')}]--> {item.get('target')}: {item.get('description', '')}"
        for item in relations[:40]
    ]
    return f"""{COMMUNITY_SUMMARY_SYSTEM_PROMPT}\n\nCommunity: {community_id}\nLevel: {level}\nEntities:\n{chr(10).join(entity_lines) or '- none'}\n\nRelationships:\n{chr(10).join(relation_lines) or '- none'}\n\nWrite the community report."""


def create_local_search_prompt(query: str, entities: list, relations: list, text_units: list | None = None) -> str:
    entity_text = "\n".join(
        f"- {item.get('name')} ({item.get('type')}): {item.get('description', '')}" for item in entities
    ) or "- none"
    relation_text = "\n".join(
        f"- {item.get('source')} --[{item.get('relation_type')}]--> {item.get('target')}: {item.get('description', '')}"
        for item in relations
    ) or "- none"
    text_unit_text = "\n".join(
        f"- [{item.get('id')}] {item.get('text', '')}" for item in (text_units or [])
    ) or "- none"
    return f"""Answer the user question using only the supplied GraphRAG context. If the evidence is insufficient, say so explicitly.\n\nEntities:\n{entity_text}\n\nRelationships:\n{relation_text}\n\nSource text units:\n{text_unit_text}\n\nQuestion: {query}\n\nAnswer:"""


def create_global_search_prompt(query: str, community_summaries: list) -> str:
    summaries = "\n\n".join(
        f"[{item.get('id')}] {item.get('summary', '')}" for item in community_summaries
    ) or "No community reports available."
    return f"""Synthesize an answer to the user question from the ranked community reports below. Separate evidence from inference and do not invent unsupported facts.\n\nCommunity reports:\n{summaries}\n\nQuestion: {query}\n\nAnswer:"""


def create_basic_search_prompt(query: str, text_units: list) -> str:
    context = "\n\n".join(f"[{item.get('id')}] {item.get('text', '')}" for item in text_units) or "No text units available."
    return f"""Answer the question using only these retrieved source passages. If the answer is not present, say that the indexed evidence is insufficient.\n\nPassages:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
