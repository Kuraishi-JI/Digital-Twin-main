from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


DETERMINISTIC_PROVIDER = "deterministic"
SUPPORTED_PROVIDER_LABELS = ("deterministic", "OpenAI", "Anthropic", "Google")
PROVIDER_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


@dataclass(frozen=True)
class LLMModelSpec:
    provider: str
    model_id: str
    display_name: str
    capability_tier: str
    supports_structured_output: bool
    supports_tool_calling: bool
    supports_reasoning: bool
    supports_vision: bool
    context_window: int
    cost_tier: str
    latency_tier: str
    recommended_for: tuple[str, ...] = field(default_factory=tuple)
    is_default: bool = False


@dataclass(frozen=True)
class ResolvedLLMSelection:
    requested_provider: str
    requested_model: str
    resolved_provider: str
    resolved_model: str
    selection_reason: str
    fallback_used: bool
    missing_credentials: tuple[str, ...]
    warnings: tuple[str, ...]
    api_key_env: str | None
    api_key_source: str

    @property
    def is_mock(self) -> bool:
        return self.resolved_provider == DETERMINISTIC_PROVIDER


MODEL_CATALOG: dict[str, tuple[LLMModelSpec, ...]] = {
    DETERMINISTIC_PROVIDER: (
        LLMModelSpec(
            DETERMINISTIC_PROVIDER,
            "offline-mock",
            "Offline Simulation Reasoning",
            "local",
            True,
            False,
            False,
            False,
            0,
            "none",
            "instant",
            ("offline_research", "reproducible_workflow"),
            True,
        ),
    ),
    "openai": (
        LLMModelSpec("openai", "gpt-5.5", "GPT-5.5", "flagship", True, True, True, True, 1_000_000, "high", "balanced", ("financial_reasoning",), True),
        LLMModelSpec("openai", "gpt-5.4", "GPT-5.4", "flagship", True, True, True, True, 1_000_000, "high", "balanced", ("financial_reasoning",)),
        LLMModelSpec("openai", "gpt-5.4-mini", "GPT-5.4 Mini", "fast", True, True, True, True, 256_000, "medium", "fast", ("fast_summary",)),
        LLMModelSpec("openai", "gpt-4.1", "GPT-4.1", "balanced", True, True, True, True, 1_000_000, "medium", "balanced", ("structured_extraction",)),
    ),
    "anthropic": (
        LLMModelSpec("anthropic", "claude-opus-4-7", "Claude Opus 4.7", "flagship", True, True, True, True, 200_000, "high", "balanced", ("financial_reasoning",), True),
        LLMModelSpec("anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6", "balanced", True, True, True, True, 200_000, "medium", "balanced", ("financial_reasoning",)),
        LLMModelSpec("anthropic", "claude-haiku-4-5", "Claude Haiku 4.5", "fast", True, True, False, True, 200_000, "low", "fast", ("fast_summary",)),
    ),
    "google": (
        LLMModelSpec("google", "gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", "flagship", True, True, True, True, 1_000_000, "high", "balanced", ("financial_reasoning",), True),
        LLMModelSpec("google", "gemini-3-flash-preview", "Gemini 3 Flash Preview", "fast", True, True, True, True, 1_000_000, "medium", "fast", ("fast_summary",)),
        LLMModelSpec("google", "gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite Preview", "cheap", True, True, False, True, 1_000_000, "low", "fast", ("cost_sensitive",)),
        LLMModelSpec("google", "gemini-2.5-pro", "Gemini 2.5 Pro", "balanced", True, True, True, True, 1_000_000, "medium", "balanced", ("long_context_analysis",)),
        LLMModelSpec("google", "gemini-2.5-flash", "Gemini 2.5 Flash", "fast", True, True, False, True, 1_000_000, "low", "fast", ("fast_summary",)),
    ),
}


def get_supported_providers() -> list[str]:
    return list(SUPPORTED_PROVIDER_LABELS)


def get_providers() -> list[str]:
    return get_supported_providers()


def normalize_provider(provider: str | None) -> str:
    value = (provider or "").strip().lower()
    aliases = {
        "openai": "openai",
        "open ai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "google": "google",
        "gemini": "google",
        "deterministic": DETERMINISTIC_PROVIDER,
        "mock": DETERMINISTIC_PROVIDER,
        "demo": DETERMINISTIC_PROVIDER,
        "offline": DETERMINISTIC_PROVIDER,
    }
    normalized = aliases.get(value)
    if not normalized:
        valid = ", ".join(SUPPORTED_PROVIDER_LABELS)
        raise ValueError(f"Unsupported LLM provider '{provider}'. Supported providers: {valid}.")
    return normalized


def display_provider(provider: str) -> str:
    normalized = normalize_provider(provider)
    return {
        DETERMINISTIC_PROVIDER: "deterministic",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google",
    }[normalized]


def get_models_for_provider(provider: str) -> list[LLMModelSpec]:
    return list(MODEL_CATALOG[normalize_provider(provider)])


def get_default_model(provider: str) -> LLMModelSpec:
    models = get_models_for_provider(provider)
    return next((model for model in models if model.is_default), models[0])


def validate_provider_model(provider: str, model: str | None) -> LLMModelSpec:
    models = get_models_for_provider(provider)
    if not model:
        return get_default_model(provider)
    for spec in models:
        if spec.model_id == model:
            return spec
    valid = ", ".join(spec.model_id for spec in models)
    raise ValueError(f"Model '{model}' is not valid for provider '{provider}'. Valid models: {valid}.")


def get_api_key_env(provider: str) -> str | None:
    return PROVIDER_KEY_ENVS.get(normalize_provider(provider))


def resolve_api_key(
    provider: str,
    *,
    ui_api_key: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, str, str | None]:
    if normalize_provider(provider) == DETERMINISTIC_PROVIDER:
        return None, "not_required", None
    env = env or os.environ
    clean_ui_key = (ui_api_key or "").strip()
    if clean_ui_key:
        return clean_ui_key, "ui", None
    env_name = get_api_key_env(provider)
    env_key = (env.get(env_name) or "").strip()
    if env_key:
        return env_key, "env", env_name
    return None, "missing", env_name


def resolve_model_selection(
    provider: str | None,
    model: str | None,
    *,
    ui_api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    allow_demo_fallback: bool = True,
) -> ResolvedLLMSelection:
    normalized = normalize_provider(provider)
    selected = validate_provider_model(normalized, model)
    if normalized == DETERMINISTIC_PROVIDER:
        return ResolvedLLMSelection(
            requested_provider=DETERMINISTIC_PROVIDER,
            requested_model=selected.model_id,
            resolved_provider=DETERMINISTIC_PROVIDER,
            resolved_model=selected.model_id,
            selection_reason="Using offline structured reasoning selected in the UI.",
            fallback_used=False,
            missing_credentials=(),
            warnings=(),
            api_key_env=None,
            api_key_source="not_required",
        )
    api_key, key_source, env_name = resolve_api_key(normalized, ui_api_key=ui_api_key, env=env)
    provider_label = display_provider(normalized)
    if not api_key and allow_demo_fallback:
        warning = (
            f"No API key was provided for {provider_label}. Enter a session key in the UI "
            f"or set {env_name}; offline structured reasoning was used."
        )
        return ResolvedLLMSelection(
            requested_provider=normalized,
            requested_model=selected.model_id,
            resolved_provider=DETERMINISTIC_PROVIDER,
            resolved_model="offline-mock",
            selection_reason=f"Missing {provider_label} API key; using offline structured reasoning.",
            fallback_used=True,
            missing_credentials=(env_name,) if env_name else (),
            warnings=(warning,),
            api_key_env=env_name,
            api_key_source="missing",
        )
    return ResolvedLLMSelection(
        requested_provider=normalized,
        requested_model=selected.model_id,
        resolved_provider=normalized if api_key else DETERMINISTIC_PROVIDER,
        resolved_model=selected.model_id if api_key else "offline-mock",
        selection_reason=(
            f"Using {provider_label}/{selected.model_id} with API key from {key_source}."
            if api_key
            else f"No {provider_label} API key available and fallback disabled."
        ),
        fallback_used=False if api_key else True,
        missing_credentials=() if api_key else ((env_name,) if env_name else ()),
        warnings=() if api_key else (f"Missing required API key for {provider_label}.",),
        api_key_env=env_name,
        api_key_source=key_source,
    )
