from __future__ import annotations

from textwrap import dedent

import pandas as pd

from .agent_adapter import run_agent_reasoning
from .allocator import fuse_allocations
from .data_foundation import benchmark_or_first, clean_price_frame, validate_prices
from .digital_twin_core import build_market_graph, build_market_state
from .environment import run_portfolio_environment
from .explainability.analysis import (
    local_contributions,
    prediction_bucket_table,
    standardized_importance,
    what_if_curve,
)
from .quant.pipeline import QuantResearchConfig, run_quant_research
from .schemas import ExplainabilityBundle, WorkflowResult, WorkflowSettings


def _quant_research_config(settings: WorkflowSettings) -> QuantResearchConfig:
    profile = settings.profile
    return QuantResearchConfig(
        benchmark=settings.benchmark,
        top_k=settings.top_k,
        retrain_every=settings.retrain_every,
        model_alpha=settings.model_alpha,
        max_weight=profile.max_weight,
        target_vol=profile.target_volatility,
        var_limit=profile.var_limit,
        drawdown_limit=profile.drawdown_limit,
        drawdown_exposure=profile.drawdown_exposure,
        transaction_cost_bps=settings.transaction_cost_bps,
    )


def _build_explainability(quant_result, allocation) -> ExplainabilityBundle:
    predictions = quant_result.predictions
    eval_frame = predictions.sample(600, random_state=42) if len(predictions) > 600 else predictions
    feature_importance = standardized_importance(
        quant_result.latest_model,
        eval_frame[quant_result.feature_columns],
        eval_frame["target"],
        quant_result.feature_columns,
    )
    prediction_buckets = prediction_bucket_table(predictions, bucket_count=5)

    latest_signal = pd.Timestamp(allocation.final_weights.index[-1])
    latest_weights = allocation.final_weights.loc[latest_signal].sort_values(ascending=False)
    selected_ticker = latest_weights.index[0]
    selected_rows = predictions[(predictions["date"] == latest_signal) & (predictions["ticker"] == selected_ticker)]
    if selected_rows.empty:
        selected_row = predictions.iloc[-1]
    else:
        selected_row = selected_rows.iloc[0]
    local = local_contributions(quant_result.latest_model, selected_row, quant_result.feature_columns)
    history = quant_result.panel[quant_result.panel["date"] <= selected_row["date"]]
    what_if = what_if_curve(quant_result.latest_model, selected_row, "mom_20d", history, quant_result.feature_columns)

    trace = pd.DataFrame(
        [
            {"stage": "data_ingestion", "artifact": "validated price frame", "status": "complete"},
            {"stage": "text_insight", "artifact": "structured agent reasoning signals", "status": "complete"},
            {"stage": "market_graph", "artifact": "correlation and sector dependency graph", "status": "complete"},
            {"stage": "portfolio_simulation", "artifact": "reward and transition ledger", "status": "complete"},
            {"stage": "fusion_allocator", "artifact": "alpha + reasoning + simulation weights", "status": "complete"},
            {"stage": "risk_governance", "artifact": "overlays, suitability checks, warnings", "status": "complete"},
            {"stage": "final_recommendation", "artifact": "human-review research allocation", "status": "pending_confirmation"},
        ]
    )
    return ExplainabilityBundle(
        feature_importance=feature_importance,
        prediction_buckets=prediction_buckets,
        local_contributions=local,
        what_if=what_if,
        structured_trace=trace,
    )


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else "n/a"


def build_audit_report(result: WorkflowResult) -> str:
    metrics = result.allocation.metrics
    latest_weights = result.allocation.final_weights.iloc[-1].sort_values(ascending=False)
    weights_md = "\n".join([f"- {ticker}: {_fmt_pct(weight)}" for ticker, weight in latest_weights.items() if weight > 0.0001])
    warnings_md = "\n".join([f"- {warning}" for warning in result.allocation.governance_warnings])
    risk_events = result.allocation.risk_events.tail(10)
    if risk_events.empty:
        risk_md = "- No risk overlay events in the final ledger."
    else:
        risk_md = "\n".join(
            f"- {pd.Timestamp(row.date).date()}: {row.event} - {row.detail}"
            for row in risk_events.itertuples(index=False)
        )

    return dedent(
        f"""
        # AI-Driven Market Digital Twin Platform Report

        **Research framing:** This report is a decision-support simulation for investment research. It is not financial advice, does not connect to a broker, and does not execute trades.

        ## Configuration

        - Benchmark: {result.settings.benchmark}
        - Risk profile: {result.settings.risk_profile}
        - Sample data mode: {result.settings.demo_mode}
        - Requested LLM provider/model: {result.agent_reasoning.llm_selection.requested_provider}/{result.agent_reasoning.llm_selection.requested_model}
        - Resolved LLM provider/model: {result.agent_reasoning.llm_selection.resolved_provider}/{result.agent_reasoning.llm_selection.resolved_model}
        - LLM fallback used: {result.agent_reasoning.llm_selection.fallback_used}
        - LLM selection reason: {result.agent_reasoning.llm_selection.selection_reason}
        - Analysis date: {result.market_state.as_of_date.date()}

        ## Performance Metrics

        - Total return: {_fmt_pct(metrics.get("total_return", 0.0))}
        - Annual return: {_fmt_pct(metrics.get("annual_return", 0.0))}
        - Annual volatility: {_fmt_pct(metrics.get("annual_volatility", 0.0))}
        - Sharpe ratio: {metrics.get("sharpe_ratio", 0.0):.2f}
        - Sortino ratio: {metrics.get("sortino_ratio", 0.0):.2f}
        - Maximum drawdown: {_fmt_pct(metrics.get("max_drawdown", 0.0))}
        - VaR 95: {_fmt_pct(metrics.get("var_95", 0.0))}
        - Expected shortfall 95: {_fmt_pct(metrics.get("es_95", 0.0))}
        - Average turnover: {_fmt_pct(metrics.get("avg_turnover", 0.0))}
        - Average exposure: {_fmt_pct(metrics.get("avg_exposure", 0.0))}
        - Win rate: {_fmt_pct(metrics.get("win_rate", 0.0))}

        ## Latest Proposed Weights

        {weights_md or "- No active risk allocation; portfolio is held in cash."}

        ## Agent Rationale

        {result.agent_reasoning.portfolio_manager_view}

        ## Risk Events

        {risk_md}

        ## Governance Warnings

        {warnings_md}
        """
    ).strip()


def run_digital_twin_workflow(prices: pd.DataFrame, settings: WorkflowSettings) -> WorkflowResult:
    clean_prices = clean_price_frame(prices, settings.universe)
    validate_prices(clean_prices)
    benchmark = benchmark_or_first(clean_prices, settings.benchmark)
    if benchmark != settings.benchmark:
        settings = WorkflowSettings(
            universe=tuple(clean_prices.columns),
            benchmark=benchmark,
            start_date=settings.start_date,
            end_date=settings.end_date,
            risk_profile=settings.risk_profile,
            demo_mode=settings.demo_mode,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            llm_api_key=settings.llm_api_key,
            allow_demo_fallback=settings.allow_demo_fallback,
            enable_checkpoint=settings.enable_checkpoint,
            alpha_weight=settings.alpha_weight,
            llm_weight=settings.llm_weight,
            rl_weight=settings.rl_weight,
            top_k=settings.top_k,
            retrain_every=settings.retrain_every,
            model_alpha=settings.model_alpha,
            transaction_cost_bps=settings.transaction_cost_bps,
            correlation_threshold=settings.correlation_threshold,
            graph_lookback=settings.graph_lookback,
        )

    quant_result = run_quant_research(clean_prices, _quant_research_config(settings))
    agent_reasoning = run_agent_reasoning(quant_result, settings)
    graph = build_market_graph(quant_result.returns, settings)
    market_state = build_market_state(quant_result, agent_reasoning, graph)
    environment = run_portfolio_environment(quant_result, agent_reasoning, graph, settings)
    allocation = fuse_allocations(quant_result, agent_reasoning, graph, environment, settings)
    explainability = _build_explainability(quant_result, allocation)
    placeholder = WorkflowResult(
        settings=settings,
        quant_result=quant_result,
        agent_reasoning=agent_reasoning,
        market_state=market_state,
        environment=environment,
        allocation=allocation,
        explainability=explainability,
        audit_report_markdown="",
    )
    placeholder.audit_report_markdown = build_audit_report(placeholder)
    return placeholder
