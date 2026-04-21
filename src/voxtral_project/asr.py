from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voxtral_project.api import DEFAULT_PROMPT, transcribe_audio_bytes
from voxtral_project.audio import audio_array_to_wav_bytes


WHISPER_LANGUAGE_NAMES = {
    "ar": "arabic",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "nl": "dutch",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "chinese",
}


def whisper_language_name_for_fleurs(lang_code: str) -> str | None:
    primary = lang_code.split("_", 1)[0].lower()
    return WHISPER_LANGUAGE_NAMES.get(primary)


@dataclass
class VLLMApiTranscriber:
    base_url: str
    model: str
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = 1000

    backend_name: str = "vllm_api"

    def transcribe(
        self,
        *,
        audio_array: Any,
        sample_rate: int,
        lang_code: str,
    ) -> str:
        del lang_code
        audio_bytes = audio_array_to_wav_bytes(
            audio_array=audio_array,
            sample_rate=sample_rate,
        )
        return transcribe_audio_bytes(
            base_url=self.base_url,
            model=self.model,
            audio_bytes=audio_bytes,
            mime_type="audio/wav",
            prompt=self.prompt,
            max_tokens=self.max_tokens,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "base_url": self.base_url,
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
        }


@dataclass
class WhisperTransformersTranscriber:
    model_id: str
    device: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    language_hint_mode: str = "known_if_supported"

    backend_name: str = "whisper_transformers"

    def __post_init__(self) -> None:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        self._torch = torch
        self._device = self._resolve_device()
        self._torch_dtype = self._resolve_torch_dtype()

        model_kwargs: dict[str, Any] = {
            "dtype": self._torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        self._model.to(self._device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            dtype=self._torch_dtype,
            device=self._device,
        )

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda:0" if self._torch.cuda.is_available() else "cpu"

    def _resolve_torch_dtype(self) -> Any:
        if self.torch_dtype == "auto":
            if self._device.startswith("cuda"):
                return self._torch.float16
            return self._torch.float32

        dtype_map = {
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "float32": self._torch.float32,
        }
        try:
            return dtype_map[self.torch_dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported torch dtype: {self.torch_dtype}") from exc

    def _generate_kwargs(self, *, lang_code: str) -> dict[str, Any]:
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if self.language_hint_mode == "auto":
            return generate_kwargs

        whisper_language = whisper_language_name_for_fleurs(lang_code)
        if whisper_language:
            generate_kwargs["language"] = whisper_language
        return generate_kwargs

    def transcribe(
        self,
        *,
        audio_array: Any,
        sample_rate: int,
        lang_code: str,
    ) -> str:
        result = self._pipe(
            {"array": audio_array, "sampling_rate": sample_rate},
            generate_kwargs=self._generate_kwargs(lang_code=lang_code),
        )
        if isinstance(result, dict) and "text" in result:
            return str(result["text"]).strip()
        return str(result).strip()

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "model_id": self.model_id,
            "device": self._device,
            "torch_dtype": str(self._torch_dtype).replace("torch.", ""),
            "attn_implementation": self.attn_implementation,
            "language_hint_mode": self.language_hint_mode,
        }


def build_transcriber(
    *,
    backend: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    hf_model_id: str,
    hf_device: str,
    hf_torch_dtype: str,
    hf_attn_implementation: str | None,
    hf_language_hint_mode: str,
) -> VLLMApiTranscriber | WhisperTransformersTranscriber:
    if backend == "vllm_api":
        return VLLMApiTranscriber(
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    if backend == "whisper_transformers":
        return WhisperTransformersTranscriber(
            model_id=hf_model_id,
            device=hf_device,
            torch_dtype=hf_torch_dtype,
            attn_implementation=hf_attn_implementation,
            language_hint_mode=hf_language_hint_mode,
        )
    raise ValueError(f"Unsupported backend: {backend}")
