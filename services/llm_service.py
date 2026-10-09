"""LLM provider abstraction with explicit offline and OpenAI implementations."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator

from config import settings
from services.storage_service import storage_service
from utils.logger import log, log_exception, log_llm_call, log_stream_chunk


class LLMService(ABC):
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model = model or settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        task: str = "general",
        stream: bool | None = None,
        save_response: bool = True,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        task: str = "general",
        **kwargs,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    def _persist_if_enabled(self, task: str, prompt: str, response: str) -> None:
        if settings.STORE_LLM_PAYLOADS:
            storage_service.save_llm_response(
                task=task,
                prompt=prompt,
                response=response,
                metadata={"model": self.model, "provider": settings.LLM_PROVIDER},
            )


class OpenAILLMService(LLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not settings.LLM_API_KEY:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL or None,
        )

    async def _completion(self, prompt: str, stream: bool = False, **kwargs):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
            **kwargs,
        )

    async def generate(
        self,
        prompt: str,
        task: str = "general",
        stream: bool | None = None,
        save_response: bool = True,
        **kwargs,
    ) -> str:
        use_stream = settings.ENABLE_STREAM if stream is None else stream
        if use_stream:
            chunks = [
                chunk
                async for chunk in self.generate_stream(
                    prompt,
                    task=task,
                    **kwargs,
                )
            ]
            response = "".join(chunks)
        else:
            response = ""
            for attempt in range(settings.MAX_RETRY + 1):
                try:
                    result = await self._completion(prompt, stream=False, **kwargs)
                    response = result.choices[0].message.content or ""
                    break
                except Exception as exc:
                    if attempt >= settings.MAX_RETRY:
                        log_exception(exc, f"llm.generate({task})")
                        raise
                    await asyncio.sleep(settings.RETRY_DELAY * (2**attempt))

        log_llm_call(prompt, response, self.model, task)
        if save_response:
            self._persist_if_enabled(task, prompt, response)
        return response

    async def generate_stream(
        self,
        prompt: str,
        task: str = "general",
        **kwargs,
    ) -> AsyncIterator[str]:
        stream = await self._completion(prompt, stream=True, **kwargs)
        async for event in stream:
            content = event.choices[0].delta.content if event.choices else None
            if content:
                if settings.STREAM_LOG_ENABLED:
                    log_stream_chunk(content, task)
                yield content


class MockLLMService(LLMService):
    """Deterministic no-network implementation used by default."""

    async def generate(
        self,
        prompt: str,
        task: str = "general",
        stream: bool | None = None,
        save_response: bool = True,
        **kwargs,
    ) -> str:
        del stream, kwargs
        if task == "entity_extraction":
            response = (
                '{"entities": [{"name": "Sample Entity", "type": "CONCEPT", '
                '"description": "Deterministic mock entity for offline execution"}], '
                '"relations": []}'
            )
        elif task == "claim_extraction":
            response = '{"claims":[]}'
        elif task.startswith("community_summary"):
            response = "Offline mock community summary generated for development testing."
        elif task.startswith("global_map_"):
            response = (
                '{"points": [{"description": "Offline mock global evidence point.", '
                '"score": 50}]}'
            )
        elif task == "global_reduce":
            response = (
                "Offline mock global answer. Configure LLM_PROVIDER=openai "
                "for model-generated synthesis."
            )
        elif task == "drift_primer":
            response = (
                '{"answer":"Offline mock DRIFT primer.",'
                '"follow_ups":[{"question":"Which indexed entities provide the '
                'strongest local evidence?","score":80}]}'
            )
        elif task.startswith("drift_followup_"):
            response = (
                '{"answer":"Offline mock DRIFT local evidence.",'
                '"confidence":0.75,"follow_ups":[]}'
            )
        elif task == "drift_reduce":
            response = (
                "Offline mock DRIFT answer. Configure LLM_PROVIDER=openai "
                "for model-generated iterative synthesis."
            )
        elif task.endswith("query_answer") or task == "basic_query_answer":
            response = (
                "Offline mock answer. Configure LLM_PROVIDER=openai "
                "for model-generated answers."
            )
        else:
            response = f"Offline mock response for task: {task}"

        log_llm_call(prompt, response, self.model, task)
        if save_response:
            self._persist_if_enabled(task, prompt, response)
        return response

    async def generate_stream(
        self,
        prompt: str,
        task: str = "general",
        **kwargs,
    ) -> AsyncIterator[str]:
        response = await self.generate(
            prompt,
            task=task,
            save_response=False,
            **kwargs,
        )
        yield response


def create_llm_service(provider: str | None = None) -> LLMService:
    selected = provider or settings.LLM_PROVIDER
    if selected == "mock":
        return MockLLMService()
    if selected == "openai":
        return OpenAILLMService()
    raise ValueError(f"Unsupported LLM provider: {selected}")


llm_service = create_llm_service()
log.info(
    "LLM service initialized: provider={}, model={}",
    settings.LLM_PROVIDER,
    llm_service.model,
)
