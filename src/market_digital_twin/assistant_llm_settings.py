from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .llm.catalog import (
    DETERMINISTIC_PROVIDER,
    LLMModelSpec,
    display_provider,
    get_api_key_env,
    get_default_model,
    get_models_for_provider,
    normalize_provider,
    resolve_api_key,
    validate_provider_model,
)


ASSISTANT_PROVIDER_LABELS = ("OpenAI", "Anthropic", "Google")


@dataclass(frozen=True)
class AssistantLLMRuntimeConfig:
    provider: str
    model: str
    api_key: str = ""
    api_key_env: str | None = None
    api_key_source: str = "missing"
    has_api_key: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


def get_assistant_providers() -> list[str]:
    """Return hosted providers supported by the sidebar assistant.

    The Command Center can still use offline structured reasoning. The Platform
    Assistant is intentionally hosted-LLM only, so the offline provider is excluded here.
    """

    return list(ASSISTANT_PROVIDER_LABELS)


def _validate_hosted_provider(provider: str) -> str:
    normalized = normalize_provider(provider)
    if normalized == DETERMINISTIC_PROVIDER:
        raise ValueError("Platform Assistant supports hosted LLM providers only.")
    if display_provider(normalized) not in ASSISTANT_PROVIDER_LABELS:
        valid = ", ".join(ASSISTANT_PROVIDER_LABELS)
        raise ValueError(f"Unsupported Platform Assistant provider '{provider}'. Supported providers: {valid}.")
    return normalized


def get_assistant_models_for_provider(provider: str) -> list[LLMModelSpec]:
    return get_models_for_provider(_validate_hosted_provider(provider))


def get_assistant_default_model(provider: str) -> LLMModelSpec:
    return get_default_model(_validate_hosted_provider(provider))


def validate_assistant_provider_model(provider: str, model: str | None) -> LLMModelSpec:
    return validate_provider_model(_validate_hosted_provider(provider), model)


def resolve_assistant_runtime_config(
    provider: str,
    model: str | None,
    *,
    ui_api_key: str | None = None,
    env: Mapping[str, str] | None = None,
) -> AssistantLLMRuntimeConfig:
    normalized = _validate_hosted_provider(provider)
    selected = validate_assistant_provider_model(normalized, model)
    api_key, key_source, env_name = resolve_api_key(normalized, ui_api_key=ui_api_key, env=env)
    provider_label = display_provider(normalized)
    warnings: list[str] = []
    if not api_key:
        warnings.append(
            f"No API key is available for {provider_label}. Enter a session-only key "
            f"or set {env_name}; the Platform Assistant will wait for a real LLM key."
        )
    return AssistantLLMRuntimeConfig(
        provider=provider_label,
        model=selected.model_id,
        api_key=api_key or "",
        api_key_env=env_name or get_api_key_env(normalized),
        api_key_source=key_source,
        has_api_key=bool(api_key),
        warnings=tuple(warnings),
    )
