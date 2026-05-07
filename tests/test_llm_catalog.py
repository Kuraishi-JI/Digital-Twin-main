from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from market_digital_twin.agent_adapter import run_agent_reasoning
from market_digital_twin.app import _model_options_for_provider
from market_digital_twin.data_foundation import load_demo_prices
from market_digital_twin.llm.catalog import (
    DETERMINISTIC_PROVIDER,
    get_default_model,
    get_models_for_provider,
    get_providers,
    resolve_api_key,
    resolve_model_selection,
    validate_provider_model,
)
from market_digital_twin.llm.client import LLMResponse, SimpleLLMClient
from market_digital_twin.schemas import WorkflowSettings
from market_digital_twin.workflow import run_digital_twin_workflow


def test_provider_list_contains_deterministic_and_supported_hosted_providers():
    assert get_providers() == ["deterministic", "OpenAI", "Anthropic", "Google"]


def test_model_filtering_returns_only_selected_provider_models():
    deterministic_models = get_models_for_provider("deterministic")
    openai_models = get_models_for_provider("OpenAI")
    anthropic_models = get_models_for_provider("Anthropic")
    google_models = get_models_for_provider("Google")

    assert [model.model_id for model in deterministic_models] == ["offline-mock"]
    assert [model.model_id for model in openai_models] == ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-4.1"]
    assert [model.model_id for model in anthropic_models] == [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ]
    assert [model.model_id for model in google_models] == [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    assert {model.provider for model in openai_models} == {"openai"}
    assert {model.provider for model in anthropic_models} == {"anthropic"}
    assert {model.provider for model in google_models} == {"google"}


def test_invalid_provider_model_pair_is_rejected():
    with pytest.raises(ValueError, match="not valid"):
        validate_provider_model("OpenAI", "claude-sonnet-4-6")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_models_for_provider("openrouter")


def test_default_model_is_returned_for_each_provider():
    assert get_default_model("deterministic").model_id == "offline-mock"
    assert get_default_model("OpenAI").model_id == "gpt-5.5"
    assert get_default_model("Anthropic").model_id == "claude-opus-4-7"
    assert get_default_model("Google").model_id == "gemini-3.1-pro-preview"


def test_api_key_resolution_prefers_ui_key_over_environment():
    api_key, source, env_name = resolve_api_key(
        "OpenAI",
        ui_api_key="ui-secret",
        env={"OPENAI_API_KEY": "env-secret"},
    )

    assert api_key == "ui-secret"
    assert source == "ui"
    assert env_name is None


def test_api_key_resolution_uses_environment_when_ui_key_missing():
    api_key, source, env_name = resolve_api_key(
        "Anthropic",
        ui_api_key="",
        env={"ANTHROPIC_API_KEY": "env-secret"},
    )

    assert api_key == "env-secret"
    assert source == "env"
    assert env_name == "ANTHROPIC_API_KEY"


def test_deterministic_provider_never_requires_api_key():
    api_key, source, env_name = resolve_api_key(
        "deterministic",
        ui_api_key="ignored",
        env={"OPENAI_API_KEY": "env-secret"},
    )
    resolved = resolve_model_selection("deterministic", "offline-mock", env={})

    assert api_key is None
    assert source == "not_required"
    assert env_name is None
    assert resolved.resolved_provider == DETERMINISTIC_PROVIDER
    assert resolved.resolved_model == "offline-mock"
    assert resolved.fallback_used is False
    assert resolved.warnings == ()


def test_missing_api_key_triggers_offline_workflow_fallback():
    resolved = resolve_model_selection(
        "Google",
        "gemini-2.5-flash",
        env={},
        allow_demo_fallback=True,
    )

    assert resolved.resolved_provider == DETERMINISTIC_PROVIDER
    assert resolved.resolved_model == "offline-mock"
    assert resolved.fallback_used is True
    assert resolved.missing_credentials == ("GOOGLE_API_KEY",)
    assert resolved.warnings


def test_streamlit_model_helper_filters_by_provider():
    models = _model_options_for_provider("Google")

    assert models
    assert {model.provider for model in models} == {"google"}


def test_google_client_keeps_api_key_out_of_request_url():
    seen: dict[str, object] = {}

    def fake_post(url, payload, headers, timeout):
        seen["url"] = url
        seen["headers"] = dict(headers)
        return {"candidates": [{"content": {"parts": [{"text": "{\"market\": \"ok\"}"}]}}]}

    selection = resolve_model_selection(
        "Google",
        "gemini-2.5-flash",
        ui_api_key="google-secret",
        env={},
    )
    response = SimpleLLMClient(selection, api_key="google-secret", post_json=fake_post).generate_text("hello")

    assert response.provider == "google"
    assert "google-secret" not in str(seen["url"])
    assert seen["headers"]["x-goog-api-key"] == "google-secret"


def test_selected_provider_model_flows_into_reasoning_adapter(monkeypatch):
    seen: dict[str, str] = {}

    def fake_generate(self: SimpleLLMClient, prompt: str, *, system: str) -> LLMResponse:
        seen["provider"] = self.selection.resolved_provider
        seen["model"] = self.selection.resolved_model
        seen["api_key"] = self.api_key
        payload = {
            "market": "Provider market report.",
            "news": "Provider news report.",
            "social": "Provider social report.",
            "fundamentals": "Provider fundamentals report.",
            "bull_case": "Provider bull case.",
            "bear_case": "Provider bear case.",
            "research_manager_view": "Provider research manager view.",
            "trader_plan": "Provider trader plan.",
            "risk_aggressive": "Provider aggressive risk view.",
            "risk_neutral": "Provider neutral risk view.",
            "risk_conservative": "Provider conservative risk view.",
            "portfolio_manager_view": "Provider portfolio manager view.",
        }
        return LLMResponse(text=json.dumps(payload), provider=self.selection.resolved_provider, model=self.selection.resolved_model)

    monkeypatch.setattr(SimpleLLMClient, "generate_text", fake_generate)
    quant_result = _minimal_quant_result()
    settings = WorkflowSettings(
        universe=("SPY", "QQQ", "TLT"),
        benchmark="SPY",
        start_date="2024-01-01",
        end_date="2024-01-31",
        demo_mode=False,
        llm_provider="OpenAI",
        llm_model="gpt-5.5",
        llm_api_key="sk-ui-secret",
    )

    reasoning = run_agent_reasoning(quant_result, settings)

    assert seen == {"provider": "openai", "model": "gpt-5.5", "api_key": "sk-ui-secret"}
    assert reasoning.llm_selection.api_key_source == "ui"
    assert reasoning.llm_selection.resolved_provider == "openai"
    redaction_blob = reasoning.memory_log.to_json() + reasoning.provider_note
    assert "sk-ui-secret" not in redaction_blob
    assert "Provider portfolio manager view." in reasoning.portfolio_manager_view


def test_api_key_is_redacted_from_audit_report_and_memory_log(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fake_generate(self: SimpleLLMClient, prompt: str, *, system: str) -> LLMResponse:
        payload = {
            "market": "Provider market report.",
            "news": "Provider news report.",
            "social": "Provider social report.",
            "fundamentals": "Provider fundamentals report.",
            "bull_case": "Provider bull case.",
            "bear_case": "Provider bear case.",
            "research_manager_view": "Provider research manager view.",
            "trader_plan": "Provider trader plan.",
            "risk_aggressive": "Provider aggressive risk view.",
            "risk_neutral": "Provider neutral risk view.",
            "risk_conservative": "Provider conservative risk view.",
            "portfolio_manager_view": "Provider portfolio manager view.",
        }
        return LLMResponse(text=json.dumps(payload), provider=self.selection.resolved_provider, model=self.selection.resolved_model)

    monkeypatch.setattr(SimpleLLMClient, "generate_text", fake_generate)
    prices = load_demo_prices("2020-01-01", "2021-12-31", tickers=["SPY", "QQQ", "TLT", "GLD"], seed=9)
    settings = WorkflowSettings(
        universe=tuple(prices.columns),
        benchmark="SPY",
        start_date="2020-01-01",
        end_date="2021-12-31",
        demo_mode=True,
        llm_provider="OpenAI",
        llm_model="gpt-5.5",
        llm_api_key="sk-report-secret",
        top_k=2,
        retrain_every=42,
    )

    result = run_digital_twin_workflow(prices, settings)
    metadata_blob = (
        result.audit_report_markdown
        + result.agent_reasoning.memory_log.to_json()
        + result.agent_reasoning.provider_note
    )

    assert "sk-report-secret" not in metadata_blob
    assert "openai/gpt-5.5" in result.audit_report_markdown


def _minimal_quant_result() -> SimpleNamespace:
    latest_date = pd.Timestamp("2024-01-31")
    predictions = pd.DataFrame(
        {
            "date": [latest_date, latest_date, latest_date],
            "ticker": ["SPY", "QQQ", "TLT"],
            "mom_20d": [0.04, 0.07, -0.02],
            "mom_60d": [0.05, 0.09, -0.01],
            "vol_20d": [0.10, 0.16, 0.08],
            "drawdown_60d": [-0.03, -0.08, -0.02],
            "rsi_14": [57.0, 69.0, 42.0],
            "prediction": [0.012, 0.018, -0.004],
        }
    )
    prices = pd.DataFrame(
        {"SPY": [100.0], "QQQ": [101.0], "TLT": [99.0]},
        index=[latest_date],
    )
    return SimpleNamespace(
        predictions=predictions,
        prices=prices,
        market_context={
            "date": latest_date,
            "market_state": "risk_on",
            "market_mom_20d": 0.04,
            "market_vol_20d": 0.13,
            "breadth_20d": 0.67,
        },
    )
