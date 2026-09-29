"""Centralized structured logging helpers."""
import sys

from loguru import logger

from config import settings


def setup_logger():
    settings.ensure_directories()
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {name}:{function}:{line} | <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=False,
    )
    logger.add(
        settings.LOG_FILE,
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    return logger


log = setup_logger()


def log_exception(exc: Exception, context: str = "") -> None:
    log.opt(exception=exc).error("Exception in {}: {}", context, exc)


def log_llm_call(prompt: str, response: str, model: str, task: str) -> None:
    # Never log prompt/response content here. Payload logging is separately gated.
    log.info(
        "LLM call | task={} model={} prompt_chars={} response_chars={}",
        task,
        model,
        len(prompt),
        len(response),
    )


def log_stream_chunk(chunk: str, task: str) -> None:
    if settings.STREAM_LOG_ENABLED:
        log.debug("LLM stream | task={} chunk_chars={}", task, len(chunk))
