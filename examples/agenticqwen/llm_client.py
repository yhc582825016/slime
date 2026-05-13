from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ASYNC_MAX_CONCURRENT = int(os.environ.get("AGENTICQWEN_LLM_MAX_CONCURRENT", "16"))
ASYNC_MAX_RETRIES = int(os.environ.get("AGENTICQWEN_LLM_MAX_RETRIES", "3"))
ASYNC_BASE_DELAY = float(os.environ.get("AGENTICQWEN_LLM_RETRY_BASE_DELAY", "10"))
ASYNC_TIMEOUT = float(os.environ.get("AGENTICQWEN_LLM_TIMEOUT", "150"))

_async_semaphore = asyncio.Semaphore(ASYNC_MAX_CONCURRENT)
_async_http_client: httpx.AsyncClient | None = None


def call_llm_sync(
    *,
    user_prompt: str,
    system_prompt: str = "",
    api_base: str | None = None,
    api_key: str | None = None,
    model_name: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout, max_retries=0)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def _get_async_client() -> httpx.AsyncClient:
    global _async_http_client
    if _async_http_client is None or _async_http_client.is_closed:
        _async_http_client = httpx.AsyncClient(timeout=httpx.Timeout(ASYNC_TIMEOUT))
    return _async_http_client


async def call_llm_async(
    *,
    user_prompt: str,
    system_prompt: str = "",
    api_base: str | None = None,
    api_key: str | None = None,
    model_name: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
) -> str:
    if not api_base:
        raise ValueError("api_base is required")
    if not api_key:
        raise ValueError("api_key is required")

    async with _async_semaphore:
        url = f"{api_base.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        client = await _get_async_client()
        last_exc: Exception | None = None
        for attempt in range(1, ASYNC_MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                last_exc = exc
                if attempt < ASYNC_MAX_RETRIES:
                    await asyncio.sleep(ASYNC_BASE_DELAY * (2 ** (attempt - 1)))
        raise RuntimeError(f"LLM call failed after retries: {last_exc!r}")
