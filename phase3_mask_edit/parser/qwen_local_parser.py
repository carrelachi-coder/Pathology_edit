"""Local Qwen parser adapter for Phase 3 semantic diffs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from phase3_mask_edit.parser.api_parser import (
    SYSTEM_PROMPT,
    build_parser_prompt,
    _messages_for_prompt_pair,
)
from phase3_mask_edit.parser.semantic_diff import (
    SEMANTIC_DIFF_SCHEMA_VERSION,
    SemanticDiffValidationError,
    extract_json_object,
    normalize_semantic_diff,
)


@dataclass(frozen=True)
class QwenLocalParserConfig:
    """Configuration for lazy local Qwen inference."""

    model_path: str
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    use_few_shot: bool = True


class QwenLocalParserError(RuntimeError):
    """Raised when the local Qwen parser cannot produce a semantic diff."""


class QwenLocalParser:
    """Lazy local Qwen parser compatible with the Phase 3 semantic-diff schema."""

    def __init__(self, config: QwenLocalParserConfig):
        if not config.model_path:
            raise QwenLocalParserError("model_path is required for qwen-local parser.")
        self.config = config
        self._model = None
        self._tokenizer = None

    def parse(self, old_prompt: str, new_prompt: str) -> dict[str, Any]:
        """Parse old/new reports into a validated semantic diff."""

        if not old_prompt or not new_prompt:
            raise QwenLocalParserError("old_prompt and new_prompt are required.")

        self._ensure_model()
        messages = _messages_for_prompt_pair(
            old_prompt,
            new_prompt,
            use_few_shot=self.config.use_few_shot,
        )
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        import torch

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)
        return canonicalize_qwen_response(response)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise QwenLocalParserError(
                "qwen-local parser requires torch and transformers."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map=self.config.device,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        self._model.eval()


def parse_prompts_with_qwen_local(
    old_prompt: str,
    new_prompt: str,
    *,
    config: QwenLocalParserConfig,
) -> dict[str, Any]:
    """Convenience wrapper for one local Qwen parse call."""

    return QwenLocalParser(config).parse(old_prompt, new_prompt)


def canonicalize_qwen_response(response: str) -> dict[str, Any]:
    """Extract and validate a Qwen JSON response without loading the model."""

    try:
        parsed = extract_json_object(response)
        parsed.setdefault("schema_version", SEMANTIC_DIFF_SCHEMA_VERSION)
        return normalize_semantic_diff(parsed, fill_missing=True)
    except SemanticDiffValidationError as exc:
        raise QwenLocalParserError(
            f"Qwen response did not match semantic_diff schema: {exc}"
        ) from exc


def qwen_prompt_preview(old_prompt: str, new_prompt: str) -> str:
    """Return a readable prompt preview for debugging local-parser requests."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append(
        {
            "role": "user",
            "content": build_parser_prompt(old_prompt, new_prompt),
        }
    )
    return json.dumps(messages, ensure_ascii=False, indent=2)
