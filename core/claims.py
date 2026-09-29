"""Optional claim/covariate extraction tied to indexed entities."""
from __future__ import annotations

import hashlib
from typing import Dict, List

from config import settings
from core.entity_resolution import normalize_entity_surface
from models.graph import KnowledgeGraph
from models.schemas import Claim, TextChunk
from prompts.quality_prompts import create_claim_extraction_prompt
from services.llm_service import llm_service
from utils.concurrency import concurrency_controller
from utils.json_extractor import extract_json
from utils.logger import log, log_exception


class ClaimExtractor:
    @staticmethod
    def _alias_lookup(kg: KnowledgeGraph) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for name, entity in kg.entities.items():
            for alias in [name, *entity.aliases]:
                normalized = normalize_entity_surface(alias)
                if normalized:
                    lookup.setdefault(normalized, name)
        return lookup

    async def extract(
        self,
        chunks: List[TextChunk],
        kg: KnowledgeGraph,
    ) -> List[Claim]:
        if not settings.CLAIM_EXTRACTION_ENABLED:
            return []

        alias_lookup = self._alias_lookup(kg)
        entity_names = sorted(kg.entities)

        async def extract_chunk(chunk: TextChunk) -> List[Claim]:
            try:
                prompt = create_claim_extraction_prompt(
                    text=chunk.text,
                    claim_description=settings.CLAIM_DESCRIPTION,
                    entity_names=entity_names,
                    max_claims=settings.MAX_CLAIMS_PER_CHUNK,
                )
                raw = await llm_service.generate(
                    prompt,
                    task="claim_extraction",
                )
                data = extract_json(raw)
                if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
                    return []

                claims: List[Claim] = []
                for item in data["claims"][: settings.MAX_CLAIMS_PER_CHUNK]:
                    if not isinstance(item, dict):
                        continue
                    subject_raw = str(item.get("subject", "")).strip()
                    subject = alias_lookup.get(
                        normalize_entity_surface(subject_raw)
                    )
                    if not subject:
                        continue

                    object_raw = item.get("object")
                    object_id = None
                    if object_raw:
                        object_text = str(object_raw).strip()
                        object_id = alias_lookup.get(
                            normalize_entity_surface(object_text),
                            object_text,
                        )

                    description = str(item.get("description", "")).strip()
                    if not description:
                        continue

                    status = str(item.get("status", "UNKNOWN")).upper()
                    if status not in {"TRUE", "FALSE", "SUSPECTED", "UNKNOWN"}:
                        status = "UNKNOWN"

                    digest_input = (
                        f"{chunk.id}|{subject}|{object_id or ''}|"
                        f"{item.get('type', 'FACT')}|{description}"
                    )
                    claim_id = "claim_" + hashlib.blake2b(
                        digest_input.encode("utf-8"),
                        digest_size=8,
                    ).hexdigest()
                    claims.append(
                        Claim(
                            id=claim_id,
                            type=str(item.get("type", "FACT")).strip() or "FACT",
                            description=description,
                            subject_id=subject,
                            object_id=object_id,
                            status=status,
                            start_date=item.get("start_date"),
                            end_date=item.get("end_date"),
                            source_text=str(item.get("source_text", "")).strip(),
                            text_unit_id=chunk.id,
                        )
                    )
                return claims
            except Exception as exc:
                log_exception(exc, f"extract_claims({chunk.id})")
                return []

        nested = await concurrency_controller.map_async(
            func=extract_chunk,
            items=chunks,
            return_exceptions=False,
        )
        claims = [claim for group in nested for claim in group]
        deduped = {claim.id: claim for claim in claims}
        result = [deduped[key] for key in sorted(deduped)]
        log.info("Extracted {} claims/covariates", len(result))
        return result


claim_extractor = ClaimExtractor()
