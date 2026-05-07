from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .llm.catalog import (
    DETERMINISTIC_PROVIDER,
    ResolvedLLMSelection,
    display_provider,
    resolve_api_key,
    resolve_model_selection,
)
from .llm.client import LLMClientError, LLMResponse, SimpleLLMClient


@dataclass(frozen=True)
class ExplainableContext:
    context_id: str
    label: str
    category: str
    short_definition: str
    detailed_description: str
    formula_or_logic: str
    interpretation: str
    related_controls: tuple[str, ...] = field(default_factory=tuple)
    related_metrics: tuple[str, ...] = field(default_factory=tuple)
    warnings_or_common_misunderstandings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AssistantSettings:
    provider: str
    model: str
    api_key: str = ""
    allow_demo_fallback: bool = True


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    selection: ResolvedLLMSelection
    fallback_used: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


ChatMessage = dict[str, str]
LLMClientFactory = Callable[..., SimpleLLMClient]
DEFAULT_THREAD_TITLE = "New chat"


SYSTEM_PROMPT = (
    "You are Platform Assistant, a guide for an AI-driven market digital twin and responsible "
    "quant advisory platform. Explain the UI, metrics, workflow, models, risk controls, and outputs. "
    "Be clear, educational, and cautious. Do not provide personalized financial advice. Explain "
    "uncertainty, assumptions, and limitations. Reference the selected context when available. "
    "Answer in the user's language when possible. Return strict JSON with one string field named answer."
)


CONTEXT_REGISTRY: dict[str, ExplainableContext] = {}


def _register(context: ExplainableContext) -> None:
    CONTEXT_REGISTRY[context.context_id] = context


def _seed_contexts() -> None:
    entries = [
        ExplainableContext(
            "start_date",
            "Start date",
            "control",
            "The first date included in the research data window.",
            "Start date controls how much historical data is available for feature construction, walk-forward model training, risk estimates, and portfolio simulation.",
            "The app filters the price panel to observations on or after this date before computing returns and rolling features.",
            "Choose a date far enough back to support rolling features and walk-forward training. Very short histories can make estimates unstable.",
            ("End date", "Universe", "Benchmark"),
            ("annual_return", "max_drawdown", "VaR 95"),
            ("A longer window is not always better if it includes market regimes that are irrelevant to the research question.",),
        ),
        ExplainableContext(
            "end_date",
            "End date",
            "control",
            "The last date included in the research data window.",
            "End date determines the final observation used for model evaluation, current market state, and latest proposed allocation.",
            "The app filters the price panel to observations on or before this date.",
            "Use the latest date for a current research memo, or an earlier date for historical scenario review.",
            ("Start date", "Universe", "Benchmark"),
            ("latest_exposure", "total_return"),
            ("If uploaded data ends before this date, the effective end date is limited by available prices.",),
        ),
        ExplainableContext(
            "benchmark",
            "Benchmark",
            "control",
            "The reference asset used for market context and performance comparison.",
            "Benchmark returns are used for beta/correlation features, market regime indicators, benchmark equity curve, and excess-return metrics.",
            "The selected benchmark must be in the active universe. If not available, the workflow falls back to the first valid asset.",
            "Pick a benchmark that represents the opportunity set, such as SPY for broad US equity exposure.",
            ("Universe", "Risk profile"),
            ("benchmark_total_return", "benchmark_excess_total_return", "beta_60d"),
            ("A poor benchmark can make excess return and beta interpretation misleading.",),
        ),
        ExplainableContext(
            "universe",
            "Universe",
            "control",
            "The set of assets included in the research run.",
            "Universe controls which price series are validated, modeled, scored, connected in the graph, and eligible for final portfolio weights.",
            "The workflow builds features and weights only for selected tickers.",
            "Use enough assets for diversification, but keep the set coherent with the research mandate.",
            ("Benchmark", "Risk profile"),
            ("portfolio_weights", "exposure", "turnover"),
            ("Changing the universe changes the model cross-section, so results are not directly comparable unless the universe is held constant.",),
        ),
        ExplainableContext(
            "risk_profile",
            "Risk profile",
            "control",
            "A preset bundle of portfolio risk limits.",
            "Risk profile sets max single-asset weight, target volatility, VaR limit, drawdown response, turnover limit, and cash buffer.",
            "Conservative, Balanced, and Growth profiles map to different `RiskProfile` parameters in `schemas.py`.",
            "Use Conservative for lower exposure and larger cash buffers, Growth for higher allowed concentration and risk tolerance.",
            ("VaR throttling", "Drawdown guard", "Cash buffer"),
            ("annual_volatility", "VaR 95", "max_drawdown"),
            ("Risk profile is not a suitability determination by itself; human review remains required.",),
        ),
        ExplainableContext(
            "demo_mode",
            "Sample data mode",
            "control",
            "A reproducible offline mode using generated sample market data.",
            "Sample data mode lets the full platform run without uploaded data or paid market data APIs.",
            "The app generates a reproducible multi-asset price panel for the selected dates and universe.",
            "Use sample data mode for classroom walkthroughs, testing, and UI exploration; use uploaded data for real research experiments.",
            ("Optional price CSV", "Universe"),
            (),
            ("Sample data is synthetic and should not be interpreted as a real investment signal.",),
        ),
        ExplainableContext(
            "llm_provider",
            "LLM provider",
            "control",
            "The model vendor used for provider-backed assistant and reasoning calls.",
            "The Command Center and Platform Assistant now keep separate LLM settings so workflow reasoning and sidebar help can use different providers.",
            "Supported hosted providers are OpenAI, Anthropic, and Google. The Command Center also keeps offline structured reasoning for reproducible workflow runs.",
            "Choose a hosted provider in the Platform Assistant sidebar when you want live chat responses from an LLM.",
            ("LLM model", "API key"),
            (),
            ("The Platform Assistant does not expose the offline workflow provider as a selectable chat provider.",),
        ),
        ExplainableContext(
            "llm_model",
            "LLM model",
            "control",
            "The specific model used under the selected provider.",
            "The model dropdown is filtered to the selected provider so invalid provider/model pairs are avoided.",
            "The selected model ID is passed into the provider client for reasoning and assistant responses.",
            "Use stronger models for complex reasoning and faster models for quick explanation tasks.",
            ("LLM provider", "API key"),
            (),
            ("A model name alone does not guarantee fresh market knowledge; the platform grounds answers in current run outputs when available.",),
        ),
        ExplainableContext(
            "api_key",
            "API key",
            "control",
            "A session-only credential used to authenticate hosted LLM calls.",
            "API keys can be typed into the password-masked UI or supplied through provider-specific environment variables.",
            "UI keys are passed only to the HTTP client; they are not placed in prompts, reports, memory logs, or exports.",
            "For Platform Assistant chat, enter a key or set the matching environment variable before asking LLM-backed questions.",
            ("LLM provider", "LLM model"),
            (),
            ("Never paste keys into normal chat messages or notes.",),
        ),
        ExplainableContext(
            "walk_forward_prediction",
            "Walk-forward prediction",
            "model",
            "A time-aware training loop that predicts future cross-sectional returns.",
            "The model trains only on data available before each signal date, then predicts the next-period return for each asset.",
            "A standardized Ridge regression is retrained on a rolling cadence. The prediction column is the forecasted next-period return.",
            "Higher positive predictions are candidates for larger alpha weights, subject to risk controls.",
            ("Retrain cadence", "Ridge alpha"),
            ("prediction", "realized_return"),
            ("Walk-forward testing reduces look-ahead bias but does not eliminate overfitting or regime risk.",),
        ),
        ExplainableContext(
            "portfolio_weights",
            "Portfolio weights",
            "output",
            "The proposed fraction of capital allocated to each asset.",
            "Weights combine supervised alpha, structured reasoning, and simulation policy components before risk overlays.",
            "Final weights are long-only and include implicit or explicit cash when exposure is below 100%.",
            "Review concentration, sector exposure, and whether risk events forced reductions.",
            ("Risk profile", "Fusion allocator"),
            ("exposure", "cash_buffer", "turnover"),
            ("Weights are research recommendations, not executable orders.",),
        ),
        ExplainableContext(
            "exposure",
            "Exposure",
            "risk_metric",
            "The invested share of the portfolio.",
            "Exposure is the sum of non-cash portfolio weights. Lower exposure means more capital is effectively held in cash.",
            "exposure = sum(asset weights)",
            "Falling exposure often indicates volatility, VaR, drawdown, or cash-buffer controls are active.",
            ("Cash buffer", "Risk profile"),
            ("latest_exposure", "avg_exposure"),
            ("Low exposure is not automatically bad; it may be the intended risk response.",),
        ),
        ExplainableContext(
            "cash_buffer",
            "Cash buffer",
            "risk_control",
            "A minimum reserve that keeps part of the portfolio uninvested.",
            "Cash buffer reduces full-risk deployment and gives the allocation room to respond to adverse conditions.",
            "Final exposure is capped so that cash weight is at least the profile buffer unless other overlays reduce exposure further.",
            "Larger cash buffers lower volatility but can reduce upside capture.",
            ("Risk profile", "Exposure"),
            ("cash_weight", "latest_exposure"),
            ("Cash here is a modeling reserve, not a brokerage cash balance.",),
        ),
        ExplainableContext(
            "turnover",
            "Turnover",
            "risk_metric",
            "The amount of portfolio weight changed during rebalancing.",
            "Turnover measures trading intensity and is penalized through transaction costs and smoothing controls.",
            "turnover = sum(abs(new_weight - previous_weight))",
            "High turnover may indicate unstable signals or overly frequent rebalancing.",
            ("Retrain cadence", "Transaction cost"),
            ("avg_turnover"),
            ("Low turnover can also mean the model is slow to adapt; interpret it with risk and return together.",),
        ),
        ExplainableContext(
            "sharpe_ratio",
            "Sharpe ratio",
            "metric",
            "Risk-adjusted return using total volatility.",
            "Sharpe compares average excess return to annualized volatility.",
            "Sharpe ratio = annual return / annual volatility in this simplified research report.",
            "Higher is generally better, but it can be unstable over short samples.",
            ("Benchmark", "Start date", "End date"),
            ("annual_return", "annual_volatility"),
            ("Sharpe treats upside and downside volatility similarly.",),
        ),
        ExplainableContext(
            "sortino_ratio",
            "Sortino ratio",
            "metric",
            "Risk-adjusted return using downside volatility.",
            "Sortino focuses on harmful volatility by penalizing negative returns more directly than Sharpe.",
            "Sortino ratio = annual return / annualized downside deviation.",
            "Useful when upside volatility is acceptable but downside volatility is not.",
            ("Risk profile",),
            ("annual_return", "downside_volatility"),
            ("Sortino can look artificially high if there are very few negative observations.",),
        ),
        ExplainableContext(
            "max_drawdown",
            "Max drawdown",
            "metric",
            "The worst peak-to-trough portfolio decline in the test period.",
            "Max drawdown measures the largest cumulative loss from a prior high-water mark.",
            "drawdown = equity / running_max(equity) - 1; max drawdown is the minimum drawdown.",
            "More negative values indicate larger historical losses.",
            ("Drawdown guard", "Risk profile"),
            ("calmar_ratio"),
            ("Drawdown is path-dependent and can be missed by volatility-only metrics.",),
        ),
        ExplainableContext(
            "var_95",
            "VaR 95",
            "risk_metric",
            "An estimate of a bad daily loss threshold at 95% confidence.",
            "VaR 95 asks how large a one-day loss could be in the worst 5% of observed or simulated days.",
            "The platform estimates VaR from recent portfolio return history.",
            "Higher VaR means larger expected tail loss and can trigger exposure throttling.",
            ("VaR throttling", "Risk profile"),
            ("expected_shortfall_95",),
            ("VaR does not describe how bad losses can be beyond the threshold.",),
        ),
        ExplainableContext(
            "expected_shortfall_95",
            "Expected Shortfall 95",
            "risk_metric",
            "The average loss after returns breach the VaR 95 threshold.",
            "Expected Shortfall looks beyond VaR and averages the tail-loss observations.",
            "ES 95 = average of returns in the worst 5% tail, expressed as a loss magnitude.",
            "It is often more informative than VaR during stressed markets.",
            ("VaR 95", "Risk profile"),
            ("rolling_es_60d",),
            ("Expected Shortfall is sensitive to sample size and outliers.",),
        ),
        ExplainableContext(
            "20d_volatility",
            "20d volatility",
            "feature",
            "Recent annualized volatility estimated from about one trading month of returns.",
            "20d volatility captures short-term market turbulence and is used by risk scoring, text-signal flags, and state features.",
            "vol_20d = rolling 20-day standard deviation of daily returns, annualized by sqrt(252).",
            "Higher values indicate more uncertain or unstable price action.",
            ("Risk profile", "Volatility targeting"),
            ("annual_volatility", "rolling_vol_20d"),
            ("A short volatility window reacts quickly but can be noisy.",),
        ),
        ExplainableContext(
            "drawdown_guard",
            "Drawdown guard",
            "risk_control",
            "An exposure-reduction rule activated by severe losses.",
            "The drawdown guard reduces exposure when portfolio drawdown breaches the selected profile limit.",
            "If drawdown is below the profile threshold, exposure is multiplied by a reduced drawdown exposure factor.",
            "It is designed to prevent the simulator from staying fully invested during adverse regimes.",
            ("Risk profile", "Max drawdown"),
            ("drawdown", "latest_exposure"),
            ("A drawdown guard can reduce losses but may also delay recovery participation.",),
        ),
        ExplainableContext(
            "volatility_targeting",
            "Volatility targeting",
            "risk_control",
            "A rule that scales exposure toward a target volatility level.",
            "If realized or estimated volatility rises above the selected target, allocation exposure can be reduced.",
            "scale = target_volatility / observed_volatility, capped by other risk rules.",
            "It seeks more stable risk through time rather than constant dollar exposure.",
            ("Risk profile", "20d volatility"),
            ("annual_volatility", "rolling_vol_20d"),
            ("Volatility targeting can sell after volatility spikes and buy after calm periods.",),
        ),
        ExplainableContext(
            "var_throttling",
            "VaR throttling",
            "risk_control",
            "A rule that reduces exposure when VaR exceeds a profile limit.",
            "VaR throttling checks recent portfolio tail risk and scales down weights if the estimated loss threshold is too high.",
            "scale = var_limit / observed_var when observed_var exceeds the limit.",
            "It is a tail-risk governor layered after signal generation.",
            ("Risk profile", "VaR 95"),
            ("rolling_var_60d", "latest_exposure"),
            ("VaR throttling is only as good as the return sample used to estimate tail risk.",),
        ),
        ExplainableContext(
            "prediction",
            "Prediction",
            "model_output",
            "The supervised model's forecasted next-period return for an asset.",
            "Prediction is one component of the final fusion allocator and is also used in reasoning consistency checks.",
            "Generated by the walk-forward Ridge model using the feature stack available at each signal date.",
            "Positive predictions support higher alpha weight; negative predictions can create caution flags.",
            ("Walk-forward prediction", "Feature importance"),
            ("realized_return", "prediction_bucket"),
            ("Prediction is not a guarantee; it is a noisy research signal.",),
        ),
        ExplainableContext(
            "realized_return",
            "Realized return",
            "model_output",
            "The return that actually occurred after a prediction date.",
            "Realized return is used to evaluate prediction quality, calibration buckets, and model diagnostics.",
            "realized_return = future price return over the configured horizon.",
            "Compare realized return to prediction to understand calibration and directional accuracy.",
            ("Prediction", "Walk-forward prediction"),
            ("win_rate", "prediction_buckets"),
            ("Realized return is only known after the fact and should not leak into model training.",),
        ),
        ExplainableContext(
            "feature_importance",
            "Feature importance",
            "explainability",
            "A global view of which inputs matter most to the model.",
            "The platform computes standardized or permutation-style importance for the latest supervised model.",
            "Features with larger importance have stronger influence on model predictions in the evaluation sample.",
            "Use it to audit whether the model relies on intuitive, stable drivers.",
            ("Prediction", "Local contribution"),
            ("permutation_importance",),
            ("Importance is not causality; correlated features can share or mask influence.",),
        ),
        ExplainableContext(
            "local_contribution",
            "Local contribution",
            "explainability",
            "A row-level explanation of what pushed one asset's prediction up or down.",
            "Local contribution decomposes a selected prediction into feature effects for interpretability.",
            "For linear models, contribution is related to standardized feature value multiplied by model coefficient.",
            "Use it to understand why the model favored or penalized a particular asset at the latest signal date.",
            ("Feature importance", "Prediction"),
            (),
            ("A local explanation can be correct for one observation but not representative globally.",),
        ),
        ExplainableContext(
            "what_if_analysis",
            "What-if analysis",
            "explainability",
            "A sensitivity check that varies one feature while holding others fixed.",
            "The what-if view shows how the model prediction changes as a selected input changes across plausible values.",
            "The platform currently uses a one-dimensional partial-dependence style curve.",
            "Use it to test whether model response is intuitive and smooth.",
            ("Local contribution", "Feature importance"),
            (),
            ("What-if curves are model diagnostics, not forecasts of what will happen in the market.",),
        ),
        ExplainableContext(
            "agent_reasoning",
            "Agent reasoning",
            "reasoning",
            "Structured narrative analysis generated by the Digital Twin Research Council.",
            "Agent reasoning converts quantitative context and optional LLM analysis into market, sentiment, risk, and portfolio rationale sections.",
            "It uses offline structured logic or the selected provider/model when a valid key is supplied.",
            "Use it to understand why the allocation proposal is framed as constructive, cautious, or balanced.",
            ("LLM provider", "LLM model"),
            ("sentiment_score", "confidence"),
            ("Narrative reasoning must be checked for bias, recency, and unsupported claims.",),
        ),
        ExplainableContext(
            "bull_bear_debate",
            "Bull/bear debate",
            "reasoning",
            "A paired argument that presents constructive and cautious cases.",
            "The bull case highlights supportive signals; the bear case highlights downside risks and weak assets.",
            "Both views are generated from the same structured signals to reduce one-sided interpretation.",
            "Use the debate to identify what evidence would change the investment memo.",
            ("Agent reasoning", "Risk debate"),
            (),
            ("Debate format improves coverage but does not guarantee truth or completeness.",),
        ),
        ExplainableContext(
            "risk_debate",
            "Risk debate",
            "reasoning",
            "A set of risk-review perspectives for the proposed allocation.",
            "The platform shows aggressive, neutral, and conservative risk views before the portfolio manager summary.",
            "These views reference exposure, volatility, drawdown, diversification, and governance controls.",
            "Use it to understand what risk trade-offs require human review.",
            ("Risk profile", "Final recommendation"),
            ("VaR 95", "max_drawdown", "exposure"),
            ("Risk debate is research commentary, not formal compliance approval.",),
        ),
        ExplainableContext(
            "final_recommendation",
            "Final recommendation",
            "governance",
            "The final research allocation memo produced by the platform.",
            "It summarizes the proposed portfolio stance after quant signals, reasoning, simulation, and risk governance are applied.",
            "The final recommendation remains pending until a human records review and acceptance in the Risk & Governance tab.",
            "Treat it as an input to investment discussion, not as an order or financial advice.",
            ("Human Review", "Audit trail"),
            ("portfolio_weights", "governance_warnings"),
            ("The platform does not execute trades and does not assess personal suitability.",),
        ),
        ExplainableContext(
            "audit_trail",
            "Audit trail",
            "governance",
            "A trace of configuration, signals, risk events, and decision evidence.",
            "The audit report is designed to make a research run reproducible and reviewable.",
            "It includes provider/model metadata but never API keys.",
            "Use it for model review, committee discussion, and reproducibility checks.",
            ("Download Audit-Ready Report", "Final recommendation"),
            ("risk_events", "decision_log"),
            ("Audit evidence supports review but does not replace compliance approval.",),
        ),
    ]
    for entry in entries:
        _register(entry)


_seed_contexts()


def get_context_registry() -> Mapping[str, ExplainableContext]:
    return CONTEXT_REGISTRY


def normalize_context_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def find_context(context_id_or_label: str | None) -> ExplainableContext | None:
    if not context_id_or_label:
        return None
    key = normalize_context_key(context_id_or_label)
    if key in CONTEXT_REGISTRY:
        return CONTEXT_REGISTRY[key]
    for context in CONTEXT_REGISTRY.values():
        aliases = {
            normalize_context_key(context.label),
            normalize_context_key(context.context_id),
            normalize_context_key(context.short_definition),
        }
        if key in aliases:
            return context
    return None


def unknown_context(selected_text: str) -> ExplainableContext:
    clean = selected_text.strip() or "selected platform content"
    return ExplainableContext(
        context_id="unknown_selection",
        label=clean[:80],
        category="unknown",
        short_definition="A user-selected item that is not yet in the platform context registry.",
        detailed_description=(
            "The assistant could not match this exact text to a registered platform concept. "
            "It will explain it cautiously using general platform knowledge and any current workflow outputs."
        ),
        formula_or_logic="No registered formula is available for this selected text.",
        interpretation="Treat this as a general explanation unless the UI provides a more specific Ask/Explain control.",
        warnings_or_common_misunderstandings=("The answer may be less specific than a registered context explanation.",),
    )


def context_to_markdown(context: ExplainableContext, snapshot: Mapping[str, Any] | None = None) -> str:
    lines = [
        f"**{context.label}**",
        "",
        f"**What it means:** {context.short_definition}",
        "",
        f"**Where it appears / how it is used:** {context.detailed_description}",
        "",
        f"**Formula or logic:** {context.formula_or_logic}",
        "",
        f"**How to interpret it:** {context.interpretation}",
    ]
    if context.related_controls:
        lines += ["", "**Related controls:** " + ", ".join(context.related_controls)]
    if context.related_metrics:
        lines += ["", "**Related metrics:** " + ", ".join(context.related_metrics)]
    if context.warnings_or_common_misunderstandings:
        lines += ["", "**Common mistakes:** " + " ".join(context.warnings_or_common_misunderstandings)]
    if snapshot:
        current = snapshot.get("current_settings") or {}
        metrics = snapshot.get("metrics") or {}
        hints = []
        if current:
            hints.append(
                "Current run settings include "
                f"benchmark={current.get('benchmark', 'n/a')}, "
                f"risk_profile={current.get('risk_profile', 'n/a')}, "
                f"sample_data_mode={current.get('demo_mode', 'n/a')}."
            )
        if metrics:
            metric_bits = []
            for name in ("total_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "var_95"):
                if name in metrics:
                    metric_bits.append(f"{name}={metrics[name]}")
            if metric_bits:
                hints.append("Current metrics snapshot: " + ", ".join(metric_bits) + ".")
        if hints:
            lines += ["", "**Current run context:** " + " ".join(hints)]
    lines += ["", "**What to check next:** Review related controls, risk warnings, and whether the run uses sample data or uploaded data."]
    return "\n".join(lines)


def build_run_snapshot(result: Any | None) -> dict[str, Any]:
    if result is None:
        return {"current_settings": {}, "metrics": {}, "risk_events": [], "audit_metadata": {}}
    settings = getattr(result, "settings", None)
    allocation = getattr(result, "allocation", None)
    reasoning = getattr(result, "agent_reasoning", None)
    metrics = dict(getattr(allocation, "metrics", {}) or {})
    compact_metrics = {
        key: _format_snapshot_value(value)
        for key, value in metrics.items()
        if key
        in {
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "var_95",
            "es_95",
            "avg_turnover",
            "latest_exposure",
            "win_rate",
        }
    }
    risk_events = []
    risk_frame = getattr(allocation, "risk_events", None)
    if risk_frame is not None and not risk_frame.empty:
        for row in risk_frame.tail(5).to_dict("records"):
            risk_events.append({str(key): str(value) for key, value in row.items()})
    selection = getattr(reasoning, "llm_selection", None)
    return {
        "current_settings": {
            "universe": list(getattr(settings, "universe", ()) or ()),
            "benchmark": getattr(settings, "benchmark", ""),
            "risk_profile": getattr(settings, "risk_profile", ""),
            "demo_mode": getattr(settings, "demo_mode", None),
            "llm_provider": getattr(settings, "llm_provider", ""),
            "llm_model": getattr(settings, "llm_model", ""),
        },
        "metrics": compact_metrics,
        "risk_events": risk_events,
        "audit_metadata": {
            "resolved_provider": getattr(selection, "resolved_provider", ""),
            "resolved_model": getattr(selection, "resolved_model", ""),
            "fallback_used": getattr(selection, "fallback_used", None),
        },
    }


def _format_snapshot_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) <= 2.0:
        return f"{number:.4f}"
    return f"{number:.2f}"


def deterministic_assistant_answer(
    user_message: str,
    *,
    active_context: ExplainableContext | None = None,
    snapshot: Mapping[str, Any] | None = None,
) -> str:
    if active_context is not None:
        base = context_to_markdown(active_context, snapshot)
        if user_message and user_message.strip() and "explain" not in user_message.lower():
            return (
                f"{base}\n\n**Follow-up answer:** For your question, "
                f"`{user_message.strip()}`, use the context above as the starting point. "
                "The platform can explain UI controls, risk metrics, and current run outputs, but it cannot provide personalized financial advice."
            )
        return base
    if user_message.strip():
        context = find_context(user_message)
        if context:
            return context_to_markdown(context, snapshot)
        return (
            "I can help explain the platform's controls, metrics, workflow, risk indicators, and reports. "
            f"I do not have a registered context match for `{user_message.strip()}`, so here is the safe approach: "
            "check whether the item is a UI control, a portfolio metric, a risk overlay, or an explainability output. "
            "Then connect it to the current run settings and risk profile. I cannot provide personalized financial advice."
        )
    return (
        "Ask me about the Command Center, Digital Twin Workflow, Agent Reasoning, Portfolio Lab, "
        "Risk & Governance, Explainability, metrics, or report fields. Select an Explain button in the app "
        "for a context-grounded explanation."
    )


def build_llm_prompt(
    user_message: str,
    *,
    active_context: ExplainableContext | None,
    snapshot: Mapping[str, Any],
    history: list[ChatMessage],
) -> str:
    safe_history = [
        {"role": item.get("role", ""), "content": item.get("content", "")[:1200]}
        for item in history[-8:]
        if item.get("role") in {"user", "assistant"}
    ]
    payload = {
        "instruction": "Answer the user's platform-help question. Return JSON with field answer.",
        "user_message": user_message,
        "active_context": _context_payload(active_context),
        "current_platform_snapshot": snapshot,
        "recent_chat_history": safe_history,
        "rules": [
            "Do not provide personalized financial advice.",
            "Do not claim hidden data access.",
            "If current workflow outputs are unavailable, say so.",
            "Never mention or reveal API keys.",
            "Answer in the user's language when possible.",
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def _context_payload(context: ExplainableContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "context_id": context.context_id,
        "label": context.label,
        "category": context.category,
        "short_definition": context.short_definition,
        "detailed_description": context.detailed_description,
        "formula_or_logic": context.formula_or_logic,
        "interpretation": context.interpretation,
        "related_controls": list(context.related_controls),
        "related_metrics": list(context.related_metrics),
        "warnings_or_common_misunderstandings": list(context.warnings_or_common_misunderstandings),
    }


def parse_assistant_payload(text: str) -> str:
    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        return parsed["answer"]
    raise ValueError("Assistant response JSON did not contain an answer field.")


def redact_secrets(text: str, secrets: list[str] | tuple[str, ...]) -> str:
    clean = text
    for secret in secrets:
        if secret:
            clean = clean.replace(secret, "[REDACTED]")
    clean = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", clean)
    clean = re.sub(r"AIza[0-9A-Za-z_\-]{12,}", "[REDACTED]", clean)
    return clean


def _hosted_llm_required_message(selection: ResolvedLLMSelection) -> str:
    provider_label = display_provider(selection.requested_provider)
    env_hint = f"`{selection.api_key_env}`" if selection.api_key_env else "the provider environment variable"
    return (
        f"Platform Assistant is set to use **{provider_label} / {selection.requested_model}**, "
        "and this assistant is configured for hosted LLM calls only. "
        f"Please enter a session-only API key in the sidebar or set {env_hint}, then ask again. "
        "No offline assistant answer was generated for this message."
    )


def generate_assistant_response(
    user_message: str,
    settings: AssistantSettings,
    *,
    active_context: ExplainableContext | None = None,
    snapshot: Mapping[str, Any] | None = None,
    history: list[ChatMessage] | None = None,
    env: Mapping[str, str] | None = None,
    client_factory: LLMClientFactory = SimpleLLMClient,
) -> AssistantResponse:
    snapshot = snapshot or {}
    history = history or []
    selection = resolve_model_selection(
        settings.provider,
        settings.model,
        ui_api_key=settings.api_key,
        env=env,
        allow_demo_fallback=settings.allow_demo_fallback,
    )
    api_key, _, _ = resolve_api_key(settings.provider, ui_api_key=settings.api_key, env=env)
    warnings = list(selection.warnings)
    if selection.resolved_provider == DETERMINISTIC_PROVIDER:
        if selection.requested_provider != DETERMINISTIC_PROVIDER and not settings.allow_demo_fallback:
            answer = _hosted_llm_required_message(selection)
            return AssistantResponse(
                answer=redact_secrets(answer, [settings.api_key]),
                selection=selection,
                fallback_used=False,
                warnings=tuple(warnings),
            )
        answer = deterministic_assistant_answer(user_message, active_context=active_context, snapshot=snapshot)
        return AssistantResponse(
            answer=redact_secrets(answer, [settings.api_key]),
            selection=selection,
            fallback_used=selection.fallback_used or selection.is_mock,
            warnings=tuple(warnings),
        )

    try:
        response: LLMResponse = client_factory(selection, api_key=api_key).generate_text(
            build_llm_prompt(user_message, active_context=active_context, snapshot=snapshot, history=history),
            system=SYSTEM_PROMPT,
        )
        answer = parse_assistant_payload(response.text)
        return AssistantResponse(
            answer=redact_secrets(answer, [settings.api_key, api_key or ""]),
            selection=selection,
            fallback_used=False,
            warnings=tuple(warnings),
        )
    except (LLMClientError, ValueError, KeyError, json.JSONDecodeError) as exc:
        safe_error = redact_secrets(str(exc), [settings.api_key, api_key or ""])
        if not settings.allow_demo_fallback:
            warnings.append(f"Provider assistant call failed. Error: {safe_error}")
            provider_label = display_provider(selection.requested_provider)
            answer = (
                f"I could not get a response from **{provider_label} / {selection.requested_model}**. "
                "Please check the API key, model access, and network connection, then try again. "
                "No offline assistant fallback is enabled."
            )
            return AssistantResponse(
                answer=redact_secrets(answer, [settings.api_key, api_key or ""]),
                selection=selection,
                fallback_used=False,
                warnings=tuple(warnings),
            )
        warnings.append(f"Provider assistant call failed; offline platform help was used. Error: {safe_error}")
        answer = deterministic_assistant_answer(user_message, active_context=active_context, snapshot=snapshot)
        return AssistantResponse(
            answer=redact_secrets(answer, [settings.api_key, api_key or ""]),
            selection=selection,
            fallback_used=True,
            warnings=tuple(warnings),
        )


def clear_chat_state(state: dict[str, Any]) -> None:
    if state.get("assistant_threads"):
        clear_active_chat_thread(state)
        return
    state["assistant_messages"] = []
    state["assistant_context_id"] = None
    state["assistant_context_explained"] = False


def delete_chat_message(state: dict[str, Any], index: int) -> bool:
    messages = get_active_chat_thread(state)["messages"] if state.get("assistant_threads") else state.setdefault("assistant_messages", [])
    if index < 0 or index >= len(messages):
        return False
    del messages[index]
    return True


def edit_user_chat_message_for_regeneration(state: dict[str, Any], index: int, new_content: str) -> bool:
    messages = get_active_chat_thread(state)["messages"] if state.get("assistant_threads") else state.setdefault("assistant_messages", [])
    if index < 0 or index >= len(messages):
        return False
    if messages[index].get("role") != "user":
        return False
    clean_content = new_content.strip()
    if not clean_content:
        return False
    messages[index]["content"] = clean_content
    if index + 1 < len(messages) and messages[index + 1].get("role") == "assistant":
        del messages[index + 1]
    if state.get("assistant_threads"):
        thread = get_active_chat_thread(state)
        first_user_index = next((idx for idx, message in enumerate(messages) if message.get("role") == "user"), None)
        if first_user_index == index:
            thread["title"] = _title_from_messages(messages) or DEFAULT_THREAD_TITLE
    return True


def ensure_chat_threads(state: dict[str, Any]) -> list[dict[str, Any]]:
    threads = state.get("assistant_threads")
    if not threads:
        legacy_messages = list(state.get("assistant_messages", []))
        thread = {
            "id": "chat_1",
            "title": _title_from_messages(legacy_messages) or DEFAULT_THREAD_TITLE,
            "messages": legacy_messages,
            "context_id": state.get("assistant_context_id"),
            "context_text": state.get("assistant_context_text", ""),
            "context_explained": bool(state.get("assistant_context_explained", False)),
        }
        state["assistant_threads"] = [thread]
        state["assistant_active_thread_id"] = "chat_1"
        state["assistant_thread_counter"] = 1
    else:
        state.setdefault("assistant_thread_counter", len(threads))
        if not state.get("assistant_active_thread_id") or not any(thread["id"] == state["assistant_active_thread_id"] for thread in threads):
            state["assistant_active_thread_id"] = threads[0]["id"]
    return state["assistant_threads"]


def get_active_chat_thread(state: dict[str, Any]) -> dict[str, Any]:
    threads = ensure_chat_threads(state)
    active_id = state.get("assistant_active_thread_id")
    for thread in threads:
        if thread["id"] == active_id:
            return thread
    state["assistant_active_thread_id"] = threads[0]["id"]
    return threads[0]


def create_chat_thread(state: dict[str, Any], *, title: str = DEFAULT_THREAD_TITLE) -> dict[str, Any]:
    threads = ensure_chat_threads(state)
    counter = int(state.get("assistant_thread_counter", len(threads))) + 1
    state["assistant_thread_counter"] = counter
    thread = {
        "id": f"chat_{counter}",
        "title": title,
        "messages": [],
        "context_id": None,
        "context_text": "",
        "context_explained": False,
    }
    threads.append(thread)
    state["assistant_active_thread_id"] = thread["id"]
    return thread


def delete_chat_thread(state: dict[str, Any], thread_id: str) -> bool:
    threads = ensure_chat_threads(state)
    index = next((idx for idx, thread in enumerate(threads) if thread["id"] == thread_id), None)
    if index is None:
        return False
    del threads[index]
    if not threads:
        state["assistant_thread_counter"] = 1
        thread = {
            "id": "chat_1",
            "title": DEFAULT_THREAD_TITLE,
            "messages": [],
            "context_id": None,
            "context_text": "",
            "context_explained": False,
        }
        threads.append(thread)
        state["assistant_active_thread_id"] = "chat_1"
        return True
    if state.get("assistant_active_thread_id") == thread_id:
        state["assistant_active_thread_id"] = threads[min(index, len(threads) - 1)]["id"]
    return True


def clear_active_chat_thread(state: dict[str, Any]) -> None:
    thread = get_active_chat_thread(state)
    thread["messages"] = []
    thread["context_id"] = None
    thread["context_text"] = ""
    thread["context_explained"] = False
    thread["title"] = DEFAULT_THREAD_TITLE


def set_active_thread_title_from_messages(state: dict[str, Any]) -> None:
    thread = get_active_chat_thread(state)
    if thread.get("title") != DEFAULT_THREAD_TITLE:
        return
    title = _title_from_messages(thread.get("messages", []))
    if title:
        thread["title"] = title


def _title_from_messages(messages: list[ChatMessage]) -> str:
    for message in messages:
        if message.get("role") == "user" and message.get("content", "").strip():
            content = re.sub(r"\s+", " ", message["content"].strip())
            return content[:34] + ("..." if len(content) > 34 else "")
    return ""
