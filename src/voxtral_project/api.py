from __future__ import annotations

import hashlib
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "Transcribe this audio."


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def list_models(*, base_url: str, timeout: int = 30) -> list[dict[str, Any]]:
    import requests

    api_base = normalize_base_url(base_url)
    response = requests.get(f"{api_base}/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def _lock_key(*, base_url: str, model: str) -> str:
    normalized = normalize_base_url(base_url)
    digest = hashlib.sha1(f"{normalized}|{model}".encode("utf-8")).hexdigest()
    return f"transcription-{digest}.lock"


def _lock_path(*, base_url: str, model: str) -> Path:
    lock_dir = Path(tempfile.gettempdir()) / "voxtral-project-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / _lock_key(base_url=base_url, model=model)


@contextmanager
def transcription_request_lock(
    *,
    base_url: str,
    model: str,
    timeout: float = 900.0,
    poll_interval: float = 0.2,
):
    lock_path = _lock_path(base_url=base_url, model=model)
    lock_file = lock_path.open("a+b")

    # Windows byte-range locking requires at least one byte to exist.
    if lock_file.tell() == 0:
        lock_file.write(b"0")
        lock_file.flush()

    deadline = time.monotonic() + timeout
    acquired = False

    try:
        while time.monotonic() < deadline:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                acquired = True
                break
            except (BlockingIOError, OSError):
                time.sleep(poll_interval)

        if not acquired:
            raise TimeoutError(
                f"Timed out waiting for transcription request lock at {lock_path}."
            )

        yield
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        lock_file.close()


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
    request_lock_timeout: float = 900.0,
) -> str:
    import requests

    api_base = normalize_base_url(base_url)
    files = {
        "file": ("audio", audio_bytes, mime_type),
    }
    data = {
        "model": model,
        "prompt": prompt,
        "response_format": "json",
        "max_completion_tokens": str(max_tokens),
    }
    with transcription_request_lock(
        base_url=api_base,
        model=model,
        timeout=request_lock_timeout,
    ):
        response = requests.post(
            f"{api_base}/audio/transcriptions",
            data=data,
            files=files,
            timeout=timeout,
        )
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/plain"):
        return response.text

    payload = response.json()
    if "text" in payload:
        return payload["text"]
    return payload["choices"][0]["message"]["content"]
