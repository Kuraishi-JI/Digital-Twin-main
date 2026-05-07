from __future__ import annotations

from datetime import date
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .allocator import allocation_risk_series
from .assistant import (
    AssistantSettings,
    build_run_snapshot,
    clear_chat_state,
    create_chat_thread,
    delete_chat_message,
    delete_chat_thread,
    edit_user_chat_message_for_regeneration,
    ensure_chat_threads,
    find_context,
    generate_assistant_response,
    get_active_chat_thread,
    redact_secrets,
    set_active_thread_title_from_messages,
    unknown_context,
)
from .data_foundation import DEFAULT_SECTOR_MAP, asset_profile, load_demo_prices, load_uploaded_prices, validate_prices
from .assistant_llm_settings import (
    get_assistant_default_model,
    get_assistant_models_for_provider,
    get_assistant_providers,
    resolve_assistant_runtime_config,
)
from .llm.catalog import DETERMINISTIC_PROVIDER, get_api_key_env, get_models_for_provider, get_providers
from .schemas import RISK_PROFILES, WorkflowSettings
from .workflow import run_digital_twin_workflow


APP_TITLE = "AI-Driven Market Digital Twin Platform"
DISCLAIMER = (
    "Research and decision-support simulation only. The platform does not provide financial advice, "
    "does not connect to brokerage systems, and does not execute trades."
)


def _pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else "n/a"


def _num(value: float) -> str:
    return f"{value:.2f}" if pd.notna(value) else "n/a"


def _model_options_for_provider(provider: str):
    return get_models_for_provider(provider)


def _format_model_option(model) -> str:
    return model.display_name


def _format_provider_option(provider: str) -> str:
    return "Offline Simulation" if provider == DETERMINISTIC_PROVIDER else provider


def _model_detail(model) -> str:
    flags = []
    if model.supports_reasoning:
        flags.append("reasoning")
    if model.supports_structured_output:
        flags.append("structured")
    if model.supports_tool_calling:
        flags.append("tools")
    flag_text = ", ".join(flags) or "basic"
    return (
        f"`{model.model_id}` | {model.capability_tier} | {model.cost_tier} cost | "
        f"{model.latency_tier} latency | {flag_text}"
    )


def _provider_env_help(provider: str) -> str:
    env_name = get_api_key_env(provider)
    if env_name is None:
        return "Offline simulation reasoning runs locally and does not require an API key."
    return f"API key source: enter a session-only key below, or set `{env_name}` before launching Streamlit."


def _llm_settings_panel() -> tuple[str, str, str, bool]:
    with st.container(border=True):
        st.markdown("**LLM Settings**")
        provider_col, model_col = st.columns([0.9, 1.3])
        with provider_col:
            llm_provider = st.selectbox(
                "LLM provider",
                options=get_providers(),
                index=0,
                key="llm_provider_select",
                format_func=_format_provider_option,
                help="Choose offline simulation for local reproducible reasoning, or a hosted provider for real LLM calls.",
            )
        model_options = _model_options_for_provider(llm_provider)
        default_index = next((idx for idx, model in enumerate(model_options) if model.is_default), 0)
        with model_col:
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                index=default_index,
                format_func=_format_model_option,
                key=f"llm_model_select_{llm_provider}",
                help="The model list is filtered to the selected provider.",
            )
        llm_model = selected_model.model_id
        st.caption(_model_detail(selected_model))

        api_key_env = get_api_key_env(llm_provider)
        if api_key_env is None or llm_provider == DETERMINISTIC_PROVIDER:
            st.info("Offline simulation reasoning is selected. No API key will be requested or stored.")
            llm_api_key = ""
        else:
            st.caption(_provider_env_help(llm_provider))
            llm_api_key = st.text_input(
                f"{llm_provider} API key",
                value="",
                type="password",
                help=(
                    "Session-only. The key is used for this run and is not written to disk, "
                    "audit reports, checkpoints, or logs."
                ),
                placeholder=f"Optional; falls back to {api_key_env}",
            )
            if llm_api_key.strip():
                st.caption("Using the API key entered for this Streamlit session.")
            elif os.environ.get(api_key_env):
                st.caption(f"No UI key entered; `{api_key_env}` is available for this session.")
            else:
                st.warning(f"No API key entered and `{api_key_env}` is not set. This run will use offline structured reasoning.")
        return llm_provider, llm_model, llm_api_key, True


def _style_page() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.3rem; padding-bottom: 2.5rem; max-width: 1480px;}
        h1, h2, h3 {letter-spacing: 0;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(120, 132, 145, 0.22);
            padding: 0.65rem 0.75rem;
            border-radius: 6px;
            background: rgba(255,255,255,0.03);
        }
        .small-note {
            color: #667085;
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .warning-band {
            border-left: 4px solid #b54708;
            background: rgba(181, 71, 8, 0.08);
            padding: 0.75rem 0.9rem;
            border-radius: 4px;
        }
        .settings-section-title {
            margin: 0.2rem 0 0.25rem 0;
            color: #344054;
            font-size: 0.94rem;
            font-weight: 650;
        }
        .settings-spacer {
            height: 0.35rem;
        }
        .asset-row-anchor {
            height: 0;
            margin: 0;
            padding: 0;
        }
        div[data-testid="stVerticalBlock"] > div:has(.asset-row-anchor) + div[data-testid="stHorizontalBlock"]
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
            min-height: 2.9rem;
            height: 2.9rem;
            max-height: 2.9rem;
            align-items: center;
            overflow: hidden;
        }
        div[data-testid="stVerticalBlock"] > div:has(.asset-row-anchor) + div[data-testid="stHorizontalBlock"]
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div > div {
            flex-wrap: nowrap;
            overflow-x: auto;
            overflow-y: hidden;
            scrollbar-width: none;
        }
        div[data-testid="stVerticalBlock"] > div:has(.asset-row-anchor) + div[data-testid="stHorizontalBlock"]
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div > div::-webkit-scrollbar {
            display: none;
        }
        div[data-testid="stVerticalBlock"] > div:has(.asset-row-anchor) + div[data-testid="stHorizontalBlock"]
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
            flex: 0 0 auto;
            margin-top: 0;
            margin-bottom: 0;
        }
        div[data-testid="stVerticalBlock"] > div:has(.asset-row-anchor) + div[data-testid="stHorizontalBlock"]
        div[data-testid="stFileUploader"] section {
            min-height: 2.9rem;
            height: 2.9rem;
            max-height: 2.9rem;
            align-items: center;
            padding-top: 0;
            padding-bottom: 0;
        }
        div[data-testid="stVerticalBlock"] > div:has(.asset-row-anchor) + div[data-testid="stHorizontalBlock"]
        div[data-testid="stFileUploader"] section > div {
            align-items: center;
            min-height: 2.9rem;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            min-height: 2.9rem;
        }
        div[data-baseweb="select"] span,
        input {
            font-size: clamp(0.92rem, 1.2vw, 1rem);
            letter-spacing: 0;
        }
        div[data-baseweb="select"] span {
            white-space: normal;
            overflow-wrap: anywhere;
        }
        small,
        .stCaptionContainer {
            overflow-wrap: anywhere;
        }
        @media (max-width: 900px) {
            .block-container {padding-left: 0.9rem; padding-right: 0.9rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _cached_demo_prices(start: str, end: str, tickers: tuple[str, ...]) -> pd.DataFrame:
    return load_demo_prices(start, end, tickers=tickers)


def _assistant_settings_from_state() -> AssistantSettings:
    provider = st.session_state.get("assistant_llm_provider", "OpenAI")
    if provider == DETERMINISTIC_PROVIDER:
        provider = "OpenAI"
    try:
        default_model = get_assistant_default_model(provider).model_id
    except ValueError:
        provider = "OpenAI"
        default_model = get_assistant_default_model(provider).model_id
    model = st.session_state.get("assistant_llm_model", default_model)
    api_key = st.session_state.get("assistant_llm_api_key", "")
    return AssistantSettings(provider=provider, model=model, api_key=api_key, allow_demo_fallback=False)


def _assistant_llm_settings_panel():
    with st.expander("LLM Settings", expanded=True):
        provider = st.selectbox(
            "Assistant provider",
            options=get_assistant_providers(),
            index=0,
            key="assistant_llm_provider_select",
            help="Platform Assistant uses its own hosted LLM provider. Offline simulation is not available here.",
        )
        model_options = get_assistant_models_for_provider(provider)
        default_model_id = get_assistant_default_model(provider).model_id
        default_index = next((idx for idx, model in enumerate(model_options) if model.model_id == default_model_id), 0)
        selected_model = st.selectbox(
            "Assistant model",
            options=model_options,
            index=default_index,
            format_func=_format_model_option,
            key=f"assistant_llm_model_select_{provider}",
            help="The model list is filtered to the assistant provider.",
        )
        st.caption(_model_detail(selected_model))

        api_key_env = get_api_key_env(provider)
        st.caption(f"Enter a session-only key below, or set `{api_key_env}` before launching Streamlit.")
        api_key = st.text_input(
            f"{provider} API key",
            value="",
            type="password",
            key=f"assistant_llm_api_key_input_{provider}",
            help="Session-only. This key is not written to disk, reports, audit trails, logs, or checkpoints.",
            placeholder=f"Optional; falls back to {api_key_env}",
        )
        runtime = resolve_assistant_runtime_config(provider, selected_model.model_id, ui_api_key=api_key)
        if runtime.has_api_key:
            source_label = "UI session key" if runtime.api_key_source == "ui" else runtime.api_key_env
            st.success(f"Assistant will use {runtime.provider} / {runtime.model} via {source_label}.")
        else:
            st.warning(" ".join(runtime.warnings))

        st.session_state["assistant_llm_provider"] = provider
        st.session_state["assistant_llm_model"] = selected_model.model_id
        st.session_state["assistant_llm_api_key"] = api_key
        return AssistantSettings(provider=provider, model=selected_model.model_id, api_key=api_key, allow_demo_fallback=False), runtime


def _active_assistant_context():
    thread = get_active_chat_thread(st.session_state)
    context_id = thread.get("context_id")
    context_text = thread.get("context_text", "")
    if not context_id:
        return None
    return find_context(context_id) or unknown_context(context_text)


def _append_assistant_response(user_message: str, result) -> None:
    settings = _assistant_settings_from_state()
    active_context = _active_assistant_context()
    thread = get_active_chat_thread(st.session_state)
    messages = thread.setdefault("messages", [])
    response = generate_assistant_response(
        user_message,
        settings,
        active_context=active_context,
        snapshot=build_run_snapshot(result),
        history=messages,
    )
    messages.append({"role": "assistant", "content": response.answer})
    set_active_thread_title_from_messages(st.session_state)


def _insert_assistant_response_after(user_message: str, message_index: int, result) -> None:
    settings = _assistant_settings_from_state()
    active_context = _active_assistant_context()
    thread = get_active_chat_thread(st.session_state)
    messages = thread.setdefault("messages", [])
    response = generate_assistant_response(
        user_message,
        settings,
        active_context=active_context,
        snapshot=build_run_snapshot(result),
        history=messages[: message_index + 1],
    )
    insert_at = min(message_index + 1, len(messages))
    messages.insert(insert_at, {"role": "assistant", "content": response.answer})
    set_active_thread_title_from_messages(st.session_state)


def _render_sidebar_assistant(result) -> None:
    threads = ensure_chat_threads(st.session_state)
    st.session_state.setdefault("assistant_delete_target", None)
    st.session_state.setdefault("assistant_delete_thread_target", None)
    st.session_state.setdefault("assistant_edit_target", None)

    with st.sidebar:
        st.title("Platform Assistant")
        settings, runtime = _assistant_llm_settings_panel()
        st.caption(f"Assistant runtime: `{runtime.provider}` / `{runtime.model}`")

        thread_ids = [thread["id"] for thread in threads]
        active_id = st.session_state.get("assistant_active_thread_id", thread_ids[0])
        if active_id not in thread_ids:
            active_id = thread_ids[0]

        def _format_thread(thread_id: str) -> str:
            thread = next(item for item in threads if item["id"] == thread_id)
            count = len(thread.get("messages", []))
            suffix = f" ({count})" if count else ""
            return f"{thread.get('title') or 'New chat'}{suffix}"

        selected_thread_id = st.selectbox(
            "Conversation",
            options=thread_ids,
            index=thread_ids.index(active_id),
            format_func=_format_thread,
        )
        if selected_thread_id != st.session_state.get("assistant_active_thread_id"):
            st.session_state["assistant_active_thread_id"] = selected_thread_id
            st.session_state["assistant_delete_target"] = None
            st.session_state["assistant_edit_target"] = None
            st.session_state["assistant_delete_thread_target"] = None
            st.rerun()

        thread = get_active_chat_thread(st.session_state)
        chat_cols = st.columns(3)
        with chat_cols[0]:
            if st.button("New", use_container_width=True, help="Start a new assistant conversation."):
                create_chat_thread(st.session_state)
                st.session_state["assistant_delete_target"] = None
                st.session_state["assistant_edit_target"] = None
                st.session_state["assistant_delete_thread_target"] = None
                st.rerun()
        with chat_cols[1]:
            if st.button("Clear", use_container_width=True, help="Clear only the current conversation."):
                clear_chat_state(st.session_state)
                st.session_state["assistant_delete_target"] = None
                st.session_state["assistant_edit_target"] = None
                st.rerun()
        with chat_cols[2]:
            if st.button("Delete", use_container_width=True, help="Choose whether to delete the selected conversation."):
                st.session_state["assistant_delete_thread_target"] = thread["id"]
                st.rerun()

        if st.session_state.get("assistant_delete_thread_target") == thread["id"]:
            st.warning(f"Delete conversation `{_format_thread(thread['id'])}`?")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm", key=f"confirm_delete_thread_{thread['id']}", use_container_width=True):
                    delete_chat_thread(st.session_state, thread["id"])
                    st.session_state["assistant_delete_thread_target"] = None
                    st.session_state["assistant_delete_target"] = None
                    st.session_state["assistant_edit_target"] = None
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"cancel_delete_thread_{thread['id']}", use_container_width=True):
                    st.session_state["assistant_delete_thread_target"] = None
                    st.rerun()

        active_context = _active_assistant_context()
        if active_context is not None:
            st.info(f"Current context: {active_context.label}")
        else:
            st.caption("No active context. Ask directly in this conversation.")

        thread = get_active_chat_thread(st.session_state)
        if active_context is not None and not thread.get("context_explained", False):
            _append_assistant_response("Explain the current selected context.", result)
            thread["context_explained"] = True

        thread = get_active_chat_thread(st.session_state)
        for idx, message in enumerate(list(thread.get("messages", []))):
            with st.chat_message(message["role"]):
                is_editing = st.session_state.get("assistant_edit_target") == idx and message.get("role") == "user"
                if is_editing:
                    edit_key = f"assistant_edit_text_{idx}"
                    if edit_key not in st.session_state:
                        st.session_state[edit_key] = message["content"]
                    edited_prompt = st.text_area(
                        "Edit question",
                        key=edit_key,
                        label_visibility="collapsed",
                    )
                    regenerate_col, cancel_col = st.columns([1, 1])
                    with regenerate_col:
                        if st.button("Regenerate", key=f"regenerate_assistant_message_{idx}", use_container_width=True):
                            safe_prompt = redact_secrets(edited_prompt, [settings.api_key])
                            if edit_user_chat_message_for_regeneration(st.session_state, idx, safe_prompt):
                                st.session_state["assistant_edit_target"] = None
                                st.session_state["assistant_delete_target"] = None
                                _insert_assistant_response_after(safe_prompt, idx, result)
                                st.rerun()
                            else:
                                st.warning("Please enter a non-empty question before regenerating.")
                    with cancel_col:
                        if st.button("Cancel", key=f"cancel_edit_assistant_message_{idx}", use_container_width=True):
                            st.session_state["assistant_edit_target"] = None
                            st.rerun()
                else:
                    st.markdown(message["content"])
                if is_editing:
                    continue
                if st.session_state.get("assistant_delete_target") == idx:
                    if message.get("role") == "user":
                        modify_col, delete_col, cancel_col = st.columns([1, 1, 1])
                        with modify_col:
                            if st.button("Modify", key=f"modify_assistant_message_{idx}", help="Edit this question and regenerate its answer.", use_container_width=True):
                                st.session_state["assistant_edit_target"] = idx
                                st.session_state[f"assistant_edit_text_{idx}"] = message["content"]
                                st.session_state["assistant_delete_target"] = None
                                st.rerun()
                    else:
                        delete_col, cancel_col = st.columns([1, 1])
                    with delete_col:
                        if st.button("Delete", key=f"confirm_delete_assistant_message_{idx}", help="Delete only this message.", use_container_width=True):
                            delete_chat_message(st.session_state, idx)
                            st.session_state["assistant_delete_target"] = None
                            st.session_state["assistant_edit_target"] = None
                            st.rerun()
                    with cancel_col:
                        if st.button("Cancel", key=f"cancel_delete_assistant_message_{idx}", use_container_width=True):
                            st.session_state["assistant_delete_target"] = None
                            st.rerun()
                elif st.button("...", key=f"show_delete_assistant_message_{idx}", help="Show message actions."):
                    st.session_state["assistant_delete_target"] = idx
                    st.session_state["assistant_edit_target"] = None
                    st.rerun()

        prompt = st.chat_input("Ask about metrics, controls, risk, or outputs")
        if prompt:
            safe_prompt = redact_secrets(prompt, [settings.api_key])
            get_active_chat_thread(st.session_state)["messages"].append({"role": "user", "content": safe_prompt})
            set_active_thread_title_from_messages(st.session_state)
            _append_assistant_response(safe_prompt, result)
            st.rerun()


def _metric_strip(result) -> None:
    metrics = result.allocation.metrics
    cols = st.columns(8)
    cols[0].metric("Total Return", _pct(metrics.get("total_return", 0.0)))
    cols[1].metric("Annual Return", _pct(metrics.get("annual_return", 0.0)))
    cols[2].metric("Annual Vol", _pct(metrics.get("annual_volatility", 0.0)))
    cols[3].metric("Sharpe", _num(metrics.get("sharpe_ratio", 0.0)))
    cols[4].metric("Max Drawdown", _pct(metrics.get("max_drawdown", 0.0)))
    cols[5].metric("VaR 95", _pct(metrics.get("var_95", 0.0)))
    cols[6].metric("Turnover", _pct(metrics.get("avg_turnover", 0.0)))
    cols[7].metric("Exposure", _pct(metrics.get("avg_exposure", 0.0)))


def _settings_form() -> tuple[bool, WorkflowSettings | None, pd.DataFrame | None]:
    default_tickers = tuple(DEFAULT_SECTOR_MAP.keys())
    llm_provider, llm_model, llm_api_key, allow_demo_fallback = _llm_settings_panel()
    with st.form("command_center_form"):
        st.markdown("**Workflow Settings**")
        row1 = st.columns(3)
        with row1[0]:
            demo_mode = st.toggle("Use sample market data", value=True)
        with row1[1]:
            start_date = st.date_input("Start date", value=date(2020, 1, 1))
        with row1[2]:
            end_date = st.date_input("End date", value=date(2025, 12, 31))

        st.markdown("<div class='asset-row-anchor'></div>", unsafe_allow_html=True)
        row2 = st.columns(3)
        with row2[0]:
            selected = st.multiselect("Universe", options=list(DEFAULT_SECTOR_MAP.keys()), default=list(default_tickers))
        benchmark_options = selected or list(DEFAULT_SECTOR_MAP.keys())
        with row2[1]:
            benchmark = st.selectbox("Benchmark", options=benchmark_options, index=0)
        with row2[2]:
            uploaded = st.file_uploader("Optional price CSV", type=["csv"], disabled=demo_mode)

        st.markdown("<div class='settings-spacer'></div>", unsafe_allow_html=True)
        row3 = st.columns(4)
        with row3[0]:
            risk_profile = st.selectbox("Risk profile", options=list(RISK_PROFILES.keys()), index=1)
        with row3[1]:
            top_k = st.slider("Alpha top-K", min_value=1, max_value=max(2, min(6, len(benchmark_options))), value=min(3, len(benchmark_options)))
        with row3[2]:
            retrain_every = st.slider("Retrain cadence", min_value=5, max_value=63, value=21, step=1)
        with row3[3]:
            model_alpha = st.slider("Ridge alpha", min_value=0.1, max_value=5.0, value=1.2, step=0.1)

        row4 = st.columns(4)
        with row4[1]:
            alpha_weight = st.slider("Supervised alpha weight", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
        with row4[2]:
            llm_weight = st.slider("Reasoning weight", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        with row4[3]:
            rl_weight = st.slider("RL policy weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        enable_checkpoint = True

        run_clicked = st.form_submit_button("Run Digital Twin Workflow", type="primary", use_container_width=True)

    if not selected:
        st.warning("Select at least two assets for the universe.")
        return run_clicked, None, None

    if demo_mode:
        prices = _cached_demo_prices(str(start_date), str(end_date), tuple(selected))
    elif uploaded is not None:
        prices = load_uploaded_prices(uploaded.getvalue())
    else:
        prices = None

    settings = WorkflowSettings(
        universe=tuple(selected),
        benchmark=benchmark,
        start_date=str(start_date),
        end_date=str(end_date),
        risk_profile=risk_profile,
        demo_mode=demo_mode,
        llm_provider=llm_provider,
        llm_model=llm_model or "",
        llm_api_key=llm_api_key,
        allow_demo_fallback=allow_demo_fallback,
        enable_checkpoint=enable_checkpoint,
        alpha_weight=alpha_weight,
        llm_weight=llm_weight,
        rl_weight=rl_weight,
        top_k=top_k,
        retrain_every=retrain_every,
        model_alpha=model_alpha,
    )
    return run_clicked, settings, prices


def _run_and_store(settings: WorkflowSettings, prices: pd.DataFrame) -> None:
    with st.spinner("Running data ingestion, structured reasoning, market graph, portfolio simulation, allocator, and governance checks..."):
        try:
            result = run_digital_twin_workflow(prices, settings)
            st.session_state["workflow_result"] = result
            st.session_state["confirmed_recommendation"] = False
            st.session_state["override_note"] = ""
        except ValueError as exc:
            st.session_state.pop("workflow_result", None)
            st.error(str(exc))


def _command_center_tab() -> None:
    st.subheader("Command Center")
    st.markdown(f"<div class='warning-band'>{DISCLAIMER}</div>", unsafe_allow_html=True)
    run_clicked, settings, prices = _settings_form()
    if prices is not None:
        report = validate_prices(prices)
        profile = asset_profile(prices)
        cols = st.columns(5)
        cols[0].metric("Assets", str(report.assets))
        cols[1].metric("Observations", str(report.observations))
        cols[2].metric("Start", report.start.strftime("%Y-%m-%d"))
        cols[3].metric("End", report.end.strftime("%Y-%m-%d"))
        cols[4].metric("Schema", "Valid" if report.valid else "Review")
        if report.warnings:
            st.warning(" ".join(report.warnings))
        st.dataframe(profile, use_container_width=True, hide_index=True)
    elif run_clicked:
        st.warning("Upload a price CSV or enable sample market data.")

    if run_clicked and settings is not None and prices is not None:
        _run_and_store(settings, prices)

    result = st.session_state.get("workflow_result")
    if result is not None:
        st.divider()
        _metric_strip(result)
        st.caption(result.agent_reasoning.provider_note)
        selection = result.agent_reasoning.llm_selection
        st.info(
            f"Resolved LLM: `{selection.resolved_provider}` / `{selection.resolved_model}`. "
            f"Fallback used: `{selection.fallback_used}`. {selection.selection_reason}"
        )
        for warning in selection.warnings:
            st.warning(warning)


def _workflow_tab(result) -> None:
    st.subheader("Digital Twin Workflow")
    trace = result.explainability.structured_trace
    st.dataframe(trace, use_container_width=True, hide_index=True)
    labels = trace["stage"].tolist()
    status_color = ["#2e7d32" if status == "complete" else "#b54708" for status in trace["status"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[1] * len(labels),
                y=labels,
                orientation="h",
                marker_color=status_color,
                text=trace["artifact"],
                textposition="inside",
                insidetextanchor="middle",
            )
        ]
    )
    fig.update_layout(height=360, xaxis_visible=False, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.15, 1.0])
    with left:
        st.markdown("**Market-State Representation**")
        state_cols = [
            "ticker",
            "prediction",
            "sentiment_score",
            "confidence",
            "graph_centrality",
            "avg_abs_correlation",
            "sector",
            "risk_flags",
        ]
        st.dataframe(result.market_state.feature_frame[state_cols], use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Market Graph Summary**")
        st.json(result.market_state.graph.summary)
        st.dataframe(result.market_state.graph.edges, use_container_width=True, hide_index=True)


def _agent_tab(result) -> None:
    st.subheader("Agent Reasoning")
    st.info(result.agent_reasoning.checkpoint_note)
    reports = result.agent_reasoning.analyst_reports
    cols = st.columns(2)
    for idx, (name, report) in enumerate(reports.items()):
        with cols[idx % 2]:
            st.markdown(f"**{name.title()} Analyst**")
            st.write(report)

    st.markdown("**Structured Text Signals**")
    st.dataframe(result.agent_reasoning.signal_frame(), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Bull Researcher**")
        st.write(result.agent_reasoning.bull_case)
    with right:
        st.markdown("**Bear Researcher**")
        st.write(result.agent_reasoning.bear_case)

    st.markdown("**Risk Debate**")
    risk_cols = st.columns(3)
    for idx, (name, text) in enumerate(result.agent_reasoning.risk_debate.items()):
        with risk_cols[idx]:
            st.markdown(f"**{name.title()}**")
            st.write(text)
    st.markdown("**Final Investment Rationale**")
    st.write(result.agent_reasoning.portfolio_manager_view)
    st.dataframe(result.agent_reasoning.memory_log, use_container_width=True, hide_index=True)


def _portfolio_tab(result) -> None:
    st.subheader("Portfolio Lab")
    latest = result.allocation.final_weights.iloc[-1].sort_values(ascending=False)
    weights_frame = latest.rename("weight").reset_index().rename(columns={"index": "ticker"})
    left, right = st.columns([1.0, 1.25])
    with left:
        fig = px.bar(weights_frame, x="ticker", y="weight", color="weight", color_continuous_scale="Tealgrn")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        perf = result.allocation.performance.reset_index()
        curve = perf.melt("date", value_vars=["equity", "benchmark_equity"], var_name="series", value_name="value")
        fig = px.line(curve, x="date", y="value", color="series")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Rebalancing Decisions**")
        st.dataframe(result.allocation.decision_log.tail(60), use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown("**Turnover, Exposure, Cash**")
        perf = result.allocation.performance.reset_index()
        fig = px.line(perf, x="date", y=["turnover", "exposure", "cash_weight"])
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Component Weights, Latest Signal Date**")
    latest_signal = result.allocation.final_weights.index[-1]
    component = pd.concat(
        {
            name: frame.loc[latest_signal]
            for name, frame in result.allocation.component_weights.items()
        },
        axis=1,
    ).reset_index().rename(columns={"index": "ticker"})
    st.dataframe(component, use_container_width=True, hide_index=True)


def _risk_tab(result) -> None:
    st.subheader("Risk & Governance")
    metrics = result.allocation.metrics
    cols = st.columns(8)
    cols[0].metric("Sortino", _num(metrics.get("sortino_ratio", 0.0)))
    cols[1].metric("Calmar", _num(metrics.get("calmar_ratio", 0.0)))
    cols[2].metric("VaR 95", _pct(metrics.get("var_95", 0.0)))
    cols[3].metric("ES 95", _pct(metrics.get("es_95", 0.0)))
    cols[4].metric("Win Rate", _pct(metrics.get("win_rate", 0.0)))
    cols[5].metric("Benchmark Return", _pct(metrics.get("benchmark_total_return", 0.0)))
    cols[6].metric("Excess Return", _pct(metrics.get("benchmark_excess_total_return", 0.0)))
    cols[7].metric("Latest Exposure", _pct(metrics.get("latest_exposure", 0.0)))

    risk_series = allocation_risk_series(result.allocation).reset_index()
    left, right = st.columns(2)
    with left:
        fig = px.line(risk_series, x="date", y=["rolling_vol_20d", "rolling_var_60d", "rolling_es_60d"])
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(risk_series, x="date", y="drawdown")
        fig.update_traces(line_color="#b42318")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Suitability Checks**")
    st.dataframe(result.allocation.suitability_checks, use_container_width=True, hide_index=True)
    st.markdown("**Risk Events**")
    if result.allocation.risk_events.empty:
        st.success("No risk events were triggered.")
    else:
        st.dataframe(result.allocation.risk_events.tail(100), use_container_width=True, hide_index=True)

    st.markdown("**Human Review**")
    for warning in result.allocation.governance_warnings:
        st.warning(warning)
    override_note = st.text_area("Manual confirmation or override note", value=st.session_state.get("override_note", ""))
    confirmed = st.checkbox(
        "I confirm this recommendation is accepted only as a research memo input.",
        value=st.session_state.get("confirmed_recommendation", False),
    )
    if st.button("Record Human Review Decision", type="primary"):
        st.session_state["confirmed_recommendation"] = bool(confirmed)
        st.session_state["override_note"] = override_note
        if confirmed:
            st.success("Human review decision recorded in session state.")
        else:
            st.info("Recommendation remains unaccepted until the confirmation box is checked.")


def _explain_tab(result) -> None:
    st.subheader("Explainability & Audit")
    left, right = st.columns(2)
    with left:
        importance = result.explainability.feature_importance.head(12)
        fig = px.bar(
            importance.sort_values("permutation_importance"),
            x="permutation_importance",
            y="feature",
            orientation="h",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        buckets = result.explainability.prediction_buckets
        fig = px.bar(buckets, x="bucket", y=["avg_prediction", "avg_realized_return"], barmode="group")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Local Contribution Analysis**")
        st.dataframe(result.explainability.local_contributions.head(12), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**What-If Analysis: 20D Momentum**")
        fig = px.line(result.explainability.what_if, x="feature_value", y="predicted_return")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Structured Reasoning Trace**")
    st.dataframe(result.explainability.structured_trace, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Audit-Ready Report",
        data=result.audit_report_markdown,
        file_name="market_digital_twin_audit_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _empty_state() -> None:
    st.info("Configure the workflow in Command Center and run the digital twin pipeline.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _style_page()
    st.title(APP_TITLE)
    st.caption("Complete AI investment research platform combining quantitative signals, structured reasoning, simulation-based allocation, explainability, and governance.")

    tabs = st.tabs(
        [
            "Command Center",
            "Digital Twin Workflow",
            "Agent Reasoning",
            "Portfolio Lab",
            "Risk & Governance",
            "Explainability & Audit",
        ]
    )
    with tabs[0]:
        _command_center_tab()

    result = st.session_state.get("workflow_result")
    with tabs[1]:
        _workflow_tab(result) if result is not None else _empty_state()
    with tabs[2]:
        _agent_tab(result) if result is not None else _empty_state()
    with tabs[3]:
        _portfolio_tab(result) if result is not None else _empty_state()
    with tabs[4]:
        _risk_tab(result) if result is not None else _empty_state()
    with tabs[5]:
        _explain_tab(result) if result is not None else _empty_state()
    _render_sidebar_assistant(result)
