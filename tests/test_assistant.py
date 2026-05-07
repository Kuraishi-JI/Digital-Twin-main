from __future__ import annotations

import json

import pytest

from market_digital_twin.assistant import (
    AssistantSettings,
    build_llm_prompt,
    clear_chat_state,
    create_chat_thread,
    delete_chat_message,
    delete_chat_thread,
    edit_user_chat_message_for_regeneration,
    ensure_chat_threads,
    find_context,
    generate_assistant_response,
    get_active_chat_thread,
    get_context_registry,
    unknown_context,
)
from market_digital_twin.assistant_llm_settings import (
    get_assistant_models_for_provider,
    get_assistant_providers,
    resolve_assistant_runtime_config,
)
from market_digital_twin.llm.client import LLMResponse


def test_context_registry_contains_key_platform_terms():
    registry = get_context_registry()

    for key in [
        "start_date",
        "benchmark",
        "risk_profile",
        "portfolio_weights",
        "sharpe_ratio",
        "var_95",
        "expected_shortfall_95",
        "20d_volatility",
        "feature_importance",
        "agent_reasoning",
        "final_recommendation",
        "audit_trail",
    ]:
        assert key in registry


def test_assistant_llm_settings_exclude_deterministic_provider():
    assert get_assistant_providers() == ["OpenAI", "Anthropic", "Google"]

    for provider in get_assistant_providers():
        assert {model.provider for model in get_assistant_models_for_provider(provider)} == {provider.lower()}

    with pytest.raises(ValueError, match="hosted LLM providers only"):
        get_assistant_models_for_provider("deterministic")


def test_direct_chat_works_without_active_context():
    class FakeClient:
        def __init__(self, selection, *, api_key=None):
            self.selection = selection

        def generate_text(self, prompt: str, *, system: str) -> LLMResponse:
            return LLMResponse(
                text=json.dumps({"answer": "I can explain controls, metrics, workflow, and risk outputs."}),
                provider=self.selection.resolved_provider,
                model=self.selection.resolved_model,
            )

    response = generate_assistant_response(
        "What can you explain?",
        AssistantSettings(provider="OpenAI", model="gpt-5.5", api_key="sk-test-secret", allow_demo_fallback=False),
        active_context=None,
        snapshot={},
        history=[],
        env={},
        client_factory=FakeClient,
    )

    assert "controls, metrics, workflow" in response.answer
    assert response.selection.resolved_provider == "openai"


def test_missing_api_key_requires_real_llm_for_platform_assistant():
    response = generate_assistant_response(
        "Explain VaR 95",
        AssistantSettings(provider="OpenAI", model="gpt-5.5", allow_demo_fallback=False),
        active_context=find_context("var_95"),
        snapshot={},
        history=[],
        env={},
    )

    assert response.selection.resolved_provider == "deterministic"
    assert response.fallback_used is False
    assert "hosted LLM calls only" in response.answer
    assert "No offline assistant answer" in response.answer


def test_assistant_runtime_config_prefers_ui_key_and_warns_when_missing():
    with_ui_key = resolve_assistant_runtime_config(
        "Anthropic",
        "claude-sonnet-4-6",
        ui_api_key="anthropic-ui-secret",
        env={"ANTHROPIC_API_KEY": "anthropic-env-secret"},
    )
    missing = resolve_assistant_runtime_config("Google", "gemini-2.5-flash", env={})

    assert with_ui_key.has_api_key is True
    assert with_ui_key.api_key_source == "ui"
    assert with_ui_key.api_key == "anthropic-ui-secret"
    assert missing.has_api_key is False
    assert "will wait for a real LLM key" in missing.warnings[0]


def test_provider_model_settings_are_reused_for_assistant_llm_call():
    seen: dict[str, str] = {}

    class FakeClient:
        def __init__(self, selection, *, api_key=None):
            seen["provider"] = selection.resolved_provider
            seen["model"] = selection.resolved_model
            seen["api_key"] = api_key

        def generate_text(self, prompt: str, *, system: str) -> LLMResponse:
            seen["prompt"] = prompt
            return LLMResponse(
                text=json.dumps({"answer": "Provider-backed assistant answer."}),
                provider=seen["provider"],
                model=seen["model"],
            )

    response = generate_assistant_response(
        "Explain Sharpe",
        AssistantSettings(provider="OpenAI", model="gpt-5.5", api_key="sk-test-secret"),
        active_context=find_context("sharpe_ratio"),
        snapshot={"metrics": {"sharpe_ratio": "1.23"}},
        history=[],
        env={},
        client_factory=FakeClient,
    )

    assert seen["provider"] == "openai"
    assert seen["model"] == "gpt-5.5"
    assert seen["api_key"] == "sk-test-secret"
    assert response.answer == "Provider-backed assistant answer."
    assert "sk-test-secret" not in seen["prompt"]


def test_api_key_values_are_redacted_from_assistant_answer():
    class FakeClient:
        def __init__(self, selection, *, api_key=None):
            self.selection = selection

        def generate_text(self, prompt: str, *, system: str) -> LLMResponse:
            return LLMResponse(
                text=json.dumps({"answer": "Never show sk-redaction-secret in chat."}),
                provider="openai",
                model="gpt-5.5",
            )

    response = generate_assistant_response(
        "Explain API key",
        AssistantSettings(provider="OpenAI", model="gpt-5.5", api_key="sk-redaction-secret", allow_demo_fallback=False),
        active_context=find_context("api_key"),
        snapshot={},
        history=[],
        env={},
        client_factory=FakeClient,
    )

    assert "sk-redaction-secret" not in response.answer
    assert "[REDACTED]" in response.answer


def test_clear_chat_resets_chat_state():
    state = {
        "assistant_messages": [{"role": "user", "content": "hello"}],
        "assistant_context_id": "start_date",
        "assistant_context_explained": True,
    }

    clear_chat_state(state)

    assert state["assistant_messages"] == []
    assert state["assistant_context_id"] is None
    assert state["assistant_context_explained"] is False


def test_delete_chat_message_removes_only_selected_message():
    state = {
        "assistant_messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
    }

    deleted = delete_chat_message(state, 1)

    assert deleted is True
    assert [message["content"] for message in state["assistant_messages"]] == ["first", "third"]
    assert delete_chat_message(state, 99) is False


def test_edit_user_message_updates_question_and_removes_next_answer():
    state = {"assistant_messages": [{"role": "user", "content": "old question"}]}
    ensure_chat_threads(state)
    thread = get_active_chat_thread(state)
    thread["messages"] = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "later question"},
    ]
    thread["title"] = "old question"

    edited = edit_user_chat_message_for_regeneration(state, 0, "new question")

    assert edited is True
    assert [message["content"] for message in thread["messages"]] == ["new question", "later question"]
    assert thread["title"] == "new question"
    thread["messages"].append({"role": "assistant", "content": "standalone answer"})
    assert edit_user_chat_message_for_regeneration(state, 2, "cannot edit assistant") is False
    assert edit_user_chat_message_for_regeneration(state, 0, "   ") is False


def test_multi_conversation_mode_creates_switches_clears_and_deletes_threads():
    state = {"assistant_messages": [{"role": "user", "content": "existing question"}]}

    threads = ensure_chat_threads(state)
    assert len(threads) == 1
    assert get_active_chat_thread(state)["title"] == "existing question"

    new_thread = create_chat_thread(state)
    new_thread["messages"].append({"role": "user", "content": "second chat"})
    assert state["assistant_active_thread_id"] == new_thread["id"]
    assert len(state["assistant_threads"]) == 2

    clear_chat_state(state)
    assert get_active_chat_thread(state)["messages"] == []
    assert get_active_chat_thread(state)["title"] == "New chat"

    first_thread_id = state["assistant_threads"][0]["id"]
    assert delete_chat_thread(state, first_thread_id) is True
    assert all(thread["id"] != first_thread_id for thread in state["assistant_threads"])
    assert len(state["assistant_threads"]) == 1


def test_unknown_selected_text_falls_back_to_general_context_prompt():
    context = unknown_context("custom selected browser text")
    prompt = build_llm_prompt(
        "Explain this",
        active_context=context,
        snapshot={},
        history=[],
    )

    assert context.context_id == "unknown_selection"
    assert "custom selected browser text" in prompt
    assert "No registered formula" in prompt
