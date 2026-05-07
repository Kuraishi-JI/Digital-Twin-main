from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd

from .llm.catalog import DETERMINISTIC_PROVIDER, resolve_api_key, resolve_model_selection
from .llm.client import LLMClientError, SimpleLLMClient
from .schemas import AgentReasoningBundle, TextSignal, WorkflowSettings


def _clip_score(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _latest_prediction_frame(quant_result) -> pd.DataFrame:
    latest_date = pd.Timestamp(quant_result.predictions["date"].max())
    frame = quant_result.predictions.loc[quant_result.predictions["date"] == latest_date].copy()
    return frame.sort_values("ticker").reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if not np.isfinite(std) or abs(std) < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def build_text_signals(quant_result, settings: WorkflowSettings) -> list[TextSignal]:
    latest = _latest_prediction_frame(quant_result)
    feature_cols = ["mom_20d", "mom_60d", "vol_20d", "drawdown_60d", "rsi_14", "prediction"]
    frame = latest[["ticker"] + feature_cols].copy()
    frame["prediction_z"] = _zscore(frame["prediction"])
    frame["momentum_blend"] = 0.55 * _zscore(frame["mom_20d"]) + 0.45 * _zscore(frame["mom_60d"])
    frame["risk_drag"] = _zscore(frame["vol_20d"].fillna(0.0)) + _zscore((-frame["drawdown_60d"]).fillna(0.0))

    signals: list[TextSignal] = []
    for row in frame.itertuples(index=False):
        score = _clip_score(
            0.45 * np.tanh(row.prediction_z)
            + 0.35 * np.tanh(row.momentum_blend)
            - 0.20 * np.tanh(row.risk_drag)
        )
        confidence = float(np.clip(0.52 + 0.30 * abs(score) + 0.10 * min(abs(row.prediction_z), 2.0) / 2.0, 0.35, 0.93))
        flags = []
        if row.vol_20d > settings.profile.target_volatility * 1.25:
            flags.append("elevated_volatility")
        if row.drawdown_60d < -settings.profile.drawdown_limit:
            flags.append("drawdown_pressure")
        if row.rsi_14 > 72:
            flags.append("overbought")
        if row.rsi_14 < 30:
            flags.append("oversold")
        if row.prediction < 0:
            flags.append("negative_alpha_forecast")

        posture = "constructive" if score > 0.18 else "cautious" if score < -0.18 else "balanced"
        evidence = (
            f"walk-forward alpha forecast {row.prediction:+.2%}",
            f"20d momentum {row.mom_20d:+.2%} and 60d momentum {row.mom_60d:+.2%}",
            f"20d volatility {row.vol_20d:.2%}; 60d drawdown {row.drawdown_60d:.2%}",
        )
        bias_warning = (
            "Offline reasoning uses price-derived proxy text signals; validate with fresh filings and news before action."
            if settings.demo_mode
            else "LLM/provider output should be checked for recency, hallucination, and confirmation bias."
        )
        rationale = (
            f"{row.ticker} receives a {posture} reasoning score because the supervised signal, momentum, "
            f"and downside-risk proxies combine to {score:+.2f} confidence-adjusted sentiment."
        )
        signals.append(
            TextSignal(
                ticker=row.ticker,
                sentiment_score=score,
                confidence=confidence,
                risk_flags=tuple(flags or ["none"]),
                evidence=evidence,
                bias_warning=bias_warning,
                investment_rationale=rationale,
            )
        )
    return signals


def run_agent_reasoning(quant_result, settings: WorkflowSettings) -> AgentReasoningBundle:
    """Produce Digital Twin Research Council reasoning with provider-aware offline continuity."""
    llm_selection = resolve_model_selection(
        settings.llm_provider,
        settings.llm_model,
        ui_api_key=settings.llm_api_key,
        allow_demo_fallback=settings.allow_demo_fallback,
    )
    runtime_api_key, _, _ = resolve_api_key(settings.llm_provider, ui_api_key=settings.llm_api_key)
    signals = build_text_signals(quant_result, settings)
    signal_frame = pd.DataFrame(
        {
            "ticker": [signal.ticker for signal in signals],
            "sentiment_score": [signal.sentiment_score for signal in signals],
            "confidence": [signal.confidence for signal in signals],
        }
    ).sort_values("sentiment_score", ascending=False)

    top = signal_frame.head(3)["ticker"].tolist()
    weak = signal_frame.tail(3)["ticker"].tolist()
    market = quant_result.market_context
    regime = str(market.get("market_state", "transition")).replace("_", " ")
    latest_date = pd.Timestamp(market.get("date", quant_result.prices.index[-1])).strftime("%Y-%m-%d")

    analyst_reports = {
        "market": (
            f"Market analyst: As of {latest_date}, the benchmark state is {regime}. "
            f"Momentum is {market.get('market_mom_20d', 0.0):+.2%}, volatility is "
            f"{market.get('market_vol_20d', 0.0):.2%}, and breadth is {market.get('breadth_20d', 0.0):.1%}."
        ),
        "news": (
            "News analyst: Offline research mode maps macro-news pressure to realized volatility, drawdown, "
            "and cross-asset dispersion. Elevated dispersion increases the need for diversified sizing."
        ),
        "social": (
            "Social analyst: Sentiment proxies are derived from momentum persistence and reversal pressure. "
            f"The strongest positive attention cluster is {', '.join(top)}."
        ),
        "fundamentals": (
            "Fundamentals analyst: Sector proxies separate duration assets, broad equity, energy, "
            "defensives, and real assets so portfolio construction does not rely on one equity factor."
        ),
    }

    bull_case = (
        f"Bull researcher: The constructive case favors {', '.join(top)} because their alpha forecasts "
        "and momentum-adjusted reasoning scores are strongest, while breadth still leaves room for selective exposure."
    )
    bear_case = (
        f"Bear researcher: The cautious case focuses on {', '.join(weak)} and any asset carrying volatility, "
        "drawdown, or negative forecast flags. Position caps and cash are required before approval."
    )
    research_manager_view = (
        "Research manager: Accept the bull case only where model alpha, text sentiment, and graph diversification "
        "agree. Otherwise treat the signal as watchlist evidence rather than a portfolio mandate."
    )
    trader_plan = (
        "Trader: Convert approved signals into long-only target weights, apply turnover smoothing, and avoid "
        "execution language because this platform is decision support rather than live trading."
    )
    risk_debate = {
        "aggressive": "Aggressive risk analyst: Allow higher exposure when reward signals are diversified across sectors.",
        "neutral": "Neutral risk analyst: Keep allocation close to target volatility and measure benchmark-relative results.",
        "conservative": "Conservative risk analyst: Throttle exposure under VaR, drawdown, or suitability warnings.",
    }
    portfolio_manager_view = (
        "Portfolio manager: The final recommendation is a research allocation memo. It requires human review, "
        "documented acceptance, and separate suitability approval before any real-world use."
    )

    provider_warnings = list(llm_selection.warnings)
    if llm_selection.resolved_provider != DETERMINISTIC_PROVIDER:
        try:
            response = SimpleLLMClient(llm_selection, api_key=runtime_api_key).generate_text(
                _provider_prompt(quant_result, signals, settings),
                system=(
                    "You are a senior AI finance research agent. Return strict JSON with keys "
                    "market, news, social, fundamentals, bull_case, bear_case, research_manager_view, "
                    "trader_plan, risk_aggressive, risk_neutral, risk_conservative, portfolio_manager_view."
                ),
            )
            payload = _parse_provider_payload(response.text)
            analyst_reports = {
                "market": payload.get("market", analyst_reports["market"]),
                "news": payload.get("news", analyst_reports["news"]),
                "social": payload.get("social", analyst_reports["social"]),
                "fundamentals": payload.get("fundamentals", analyst_reports["fundamentals"]),
            }
            bull_case = payload.get("bull_case", bull_case)
            bear_case = payload.get("bear_case", bear_case)
            research_manager_view = payload.get("research_manager_view", research_manager_view)
            trader_plan = payload.get("trader_plan", trader_plan)
            risk_debate = {
                "aggressive": payload.get("risk_aggressive", risk_debate["aggressive"]),
                "neutral": payload.get("risk_neutral", risk_debate["neutral"]),
                "conservative": payload.get("risk_conservative", risk_debate["conservative"]),
            }
            portfolio_manager_view = payload.get("portfolio_manager_view", portfolio_manager_view)
        except (LLMClientError, ValueError, KeyError) as exc:
            warning = f"Real LLM call failed for {llm_selection.resolved_provider}/{llm_selection.resolved_model}: {exc}. Offline structured reasoning was used for this run."
            provider_warnings.append(warning)

    memory_log = pd.DataFrame(
        [
            {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "requested_provider": llm_selection.requested_provider,
                "requested_model": llm_selection.requested_model or "",
                "resolved_provider": llm_selection.resolved_provider,
                "resolved_model": llm_selection.resolved_model,
                "fallback_used": llm_selection.fallback_used,
                "mode": "offline_structured" if llm_selection.is_mock else "real_provider",
                "summary": portfolio_manager_view,
                "selection_reason": llm_selection.selection_reason,
            }
        ]
    )
    checkpoint_note = (
        "Research trace continuity is enabled conceptually; offline simulation runs remain stateless and replayable."
        if settings.enable_checkpoint
        else "Research trace continuity is disabled for this run."
    )
    provider_note = _provider_note(llm_selection, provider_warnings)

    return AgentReasoningBundle(
        analyst_reports=analyst_reports,
        bull_case=bull_case,
        bear_case=bear_case,
        research_manager_view=research_manager_view,
        trader_plan=trader_plan,
        risk_debate=risk_debate,
        portfolio_manager_view=portfolio_manager_view,
        text_signals=signals,
        memory_log=memory_log,
        checkpoint_note=checkpoint_note,
        provider_note=provider_note,
        llm_selection=llm_selection,
    )


def _provider_prompt(quant_result, signals: list[TextSignal], settings: WorkflowSettings) -> str:
    context = quant_result.market_context
    signal_rows = [
        {
            "ticker": signal.ticker,
            "sentiment_score": round(signal.sentiment_score, 4),
            "confidence": round(signal.confidence, 4),
            "risk_flags": list(signal.risk_flags),
            "evidence": list(signal.evidence),
        }
        for signal in signals
    ]
    return json.dumps(
        {
            "task": "Produce concise multi-agent financial reasoning for an original market digital twin research platform.",
            "risk_profile": settings.risk_profile,
            "benchmark": settings.benchmark,
            "market_context": {k: str(v) for k, v in context.items()},
            "structured_signals": signal_rows,
            "constraints": [
                "Do not provide financial advice.",
                "Do not recommend live execution.",
                "Prefer auditable evidence and risk caveats.",
                "Keep each JSON value under 120 words.",
            ],
        },
        ensure_ascii=True,
    )


def _parse_provider_payload(text: str) -> dict[str, str]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object.")
    return {str(k): str(v) for k, v in parsed.items()}


def _provider_note(selection, warnings: list[str]) -> str:
    warning_text = " ".join(warnings)
    base = (
        f"Requested provider/model: {selection.requested_provider}/{selection.requested_model}. "
        f"Resolved provider/model: {selection.resolved_provider}/{selection.resolved_model}. "
        f"{selection.selection_reason}"
    )
    if selection.is_mock:
        base += " Offline structured reasoning is active."
    else:
        base += " Real provider-backed reasoning was requested for the agent narrative layer."
    return f"{base} {warning_text}".strip()
