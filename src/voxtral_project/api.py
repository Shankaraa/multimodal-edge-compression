from __future__ import annotations

import time
from typing import Any

from voxtral_project.audio import encode_bytes_as_data_url


DEFAULT_PROMPT = "Transcribe this audio."


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def build_payload(model: str, audio_url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }


def list_models(*, base_url: str, timeout: int = 30) -> list[dict[str, Any]]:
    import requests

    api_base = normalize_base_url(base_url)
    response = requests.get(f"{api_base}/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def wait_for_server_ready(
    *,
    base_url: str,
    timeout: int = 600,
    interval: float = 5.0,
) -> list[dict[str, Any]]:
    import requests

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            return list_models(base_url=base_url, timeout=max(5, int(interval) + 5))
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(interval)

    if last_error is not None:
        raise TimeoutError(f"Server at {normalize_base_url(base_url)} did not become ready in {timeout} seconds.") from last_error
    raise TimeoutError(f"Server at {normalize_base_url(base_url)} did not become ready in {timeout} seconds.")


def transcribe_audio_bytes(
    *,
    base_url: str,
    model: str,
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 1000,
    timeout: int = 300,
) -> str:
    import requests

    api_base = normalize_base_url(base_url)
    payload = build_payload(
        model=model,
        audio_url=encode_bytes_as_data_url(audio_bytes, mime_type),
        prompt=prompt,
        max_tokens=max_tokens,
    )
    response = requests.post(
        f"{api_base}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
