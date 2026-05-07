from __future__ import annotations

import numpy as np
import pandas as pd

from .quant.pipeline import compute_metrics, compute_risk_series
from .schemas import AgentReasoningBundle, AllocationResult, EnvironmentResult, MarketGraph, WorkflowSettings


def _cap_weights(weights: pd.Series, cap: float) -> pd.Series:
    capped = weights.clip(lower=0.0).astype(float).copy()
    if capped.sum() <= 1e-12:
        return capped
    capped = capped / capped.sum()
    for _ in range(12):
        above = capped > cap
        if not above.any():
            break
        excess = float((capped[above] - cap).sum())
        capped[above] = cap
        room = cap - capped[~above]
        room = room[room > 0.0]
        if room.empty or excess <= 1e-12:
            break
        capped.loc[room.index] += excess * room / room.sum()
    return capped / capped.sum() if capped.sum() > 0 else capped


def _llm_weight_frame(
    signal_dates: list[pd.Timestamp],
    assets: list[str],
    agent_reasoning: AgentReasoningBundle,
    settings: WorkflowSettings,
) -> pd.DataFrame:
    signals = agent_reasoning.signal_frame().set_index("ticker")
    raw = signals["sentiment_score"].reindex(assets).fillna(0.0) * signals["confidence"].reindex(assets).fillna(0.5)
    raw = raw - raw.min()
    if raw.sum() <= 1e-12:
        raw = pd.Series(1.0, index=assets)
    weights = _cap_weights(raw, settings.profile.max_weight) * (1.0 - settings.profile.cash_buffer)
    return pd.DataFrame([weights.reindex(assets).values] * len(signal_dates), index=signal_dates, columns=assets)


def _portfolio_var(history_returns: pd.Series, confidence: float = 0.95) -> float:
    if history_returns.empty:
        return 0.0
    alpha = 1.0 - confidence
    return float(-np.quantile(history_returns.dropna(), alpha))


def _build_performance(strategy_returns: pd.Series, benchmark_returns: pd.Series, turnover: pd.Series, exposure: pd.Series, cash: pd.Series) -> pd.DataFrame:
    equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    frame = pd.DataFrame(
        {
            "strategy_return": strategy_returns,
            "benchmark_return": benchmark_returns,
            "equity": equity,
            "benchmark_equity": benchmark_equity,
            "drawdown": drawdown,
            "turnover": turnover,
            "exposure": exposure,
            "cash_weight": cash,
        }
    )
    frame.index.name = "date"
    return frame


def _suitability_checks(performance: pd.DataFrame, final_weights: pd.DataFrame, settings: WorkflowSettings) -> pd.DataFrame:
    latest_weights = final_weights.iloc[-1] if not final_weights.empty else pd.Series(dtype=float)
    latest_exposure = float(latest_weights.sum()) if not latest_weights.empty else 0.0
    max_weight = float(latest_weights.max()) if not latest_weights.empty else 0.0
    realized_vol = float(performance["strategy_return"].tail(60).std() * np.sqrt(252)) if len(performance) >= 20 else 0.0
    max_drawdown = float(performance["drawdown"].min()) if not performance.empty else 0.0
    return pd.DataFrame(
        [
            {
                "check": "research_use_only",
                "status": "Pass",
                "detail": "Recommendation is a decision-support simulation, not investment advice or live execution.",
            },
            {
                "check": "single_asset_cap",
                "status": "Pass" if max_weight <= settings.profile.max_weight + 1e-9 else "Review",
                "detail": f"Latest max position {max_weight:.1%}; cap {settings.profile.max_weight:.1%}.",
            },
            {
                "check": "cash_buffer",
                "status": "Pass" if latest_exposure <= 1.0 - settings.profile.cash_buffer + 1e-9 else "Review",
                "detail": f"Latest exposure {latest_exposure:.1%}; required cash buffer {settings.profile.cash_buffer:.1%}.",
            },
            {
                "check": "volatility_suitability",
                "status": "Pass" if realized_vol <= settings.profile.target_volatility * 1.35 else "Review",
                "detail": f"Trailing realized vol {realized_vol:.1%}; target {settings.profile.target_volatility:.1%}.",
            },
            {
                "check": "drawdown_suitability",
                "status": "Pass" if abs(max_drawdown) <= settings.profile.drawdown_limit * 1.75 else "Review",
                "detail": f"Backtest max drawdown {max_drawdown:.1%}; guard trigger {settings.profile.drawdown_limit:.1%}.",
            },
        ]
    )


def fuse_allocations(
    quant_result,
    agent_reasoning: AgentReasoningBundle,
    graph: MarketGraph,
    environment: EnvironmentResult,
    settings: WorkflowSettings,
) -> AllocationResult:
    assets = list(quant_result.returns.columns)
    signal_dates = sorted(pd.to_datetime(quant_result.predictions["date"].unique()))

    supervised = quant_result.signal_weights.reindex(index=signal_dates, columns=assets).fillna(0.0)
    llm_weights = _llm_weight_frame(signal_dates, assets, agent_reasoning, settings)
    rl_weights = environment.weights.reindex(index=signal_dates, columns=assets).fillna(0.0)

    fusion_total = settings.alpha_weight + settings.llm_weight + settings.rl_weight
    if fusion_total <= 1e-12:
        alpha_w, llm_w, rl_w = 0.45, 0.25, 0.30
    else:
        alpha_w = settings.alpha_weight / fusion_total
        llm_w = settings.llm_weight / fusion_total
        rl_w = settings.rl_weight / fusion_total

    raw_fused = alpha_w * supervised + llm_w * llm_weights + rl_w * rl_weights
    final_by_signal = pd.DataFrame(0.0, index=signal_dates, columns=assets)
    prev_weights = pd.Series(0.0, index=assets)
    equity = 1.0
    peak = 1.0
    perf_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for signal_date, day_frame in quant_result.predictions.groupby("date", sort=True):
        signal_ts = pd.Timestamp(signal_date)
        realized_date = pd.Timestamp(day_frame["realized_date"].iloc[0])
        target = raw_fused.loc[signal_ts].reindex(assets).fillna(0.0)
        if target.sum() > 1e-12:
            target = _cap_weights(target, settings.profile.max_weight)
        target = target * (1.0 - settings.profile.cash_buffer)

        history = quant_result.returns.loc[quant_result.returns.index < signal_ts, assets].tail(60).fillna(0.0)
        vol_scalar = 1.0
        var_scalar = 1.0
        est_vol = 0.0
        est_var = 0.0
        if len(history) >= 20 and target.sum() > 0:
            history_port = history.mul(target, axis=1).sum(axis=1)
            est_vol = float(history_port.std() * np.sqrt(252))
            est_var = _portfolio_var(history_port)
            if est_vol > 1e-12:
                vol_scalar = min(1.0, settings.profile.target_volatility / est_vol)
            if est_var > 1e-12:
                var_scalar = min(1.0, settings.profile.var_limit / est_var)

        current_drawdown = equity / peak - 1.0
        dd_scalar = settings.profile.drawdown_exposure if current_drawdown <= -settings.profile.drawdown_limit else 1.0
        target = target * min(vol_scalar, var_scalar, dd_scalar)

        turnover_before_control = float((target - prev_weights).abs().sum())
        turnover_scalar = 1.0
        if turnover_before_control > settings.profile.max_turnover > 0:
            turnover_scalar = settings.profile.max_turnover / turnover_before_control
            target = prev_weights + (target - prev_weights) * turnover_scalar

        exposure_cap = 1.0 - settings.profile.cash_buffer
        if target.sum() > exposure_cap:
            target = target * (exposure_cap / target.sum())
        target = target.clip(lower=0.0)
        final_by_signal.loc[signal_ts] = target

        realized_returns = day_frame.set_index("ticker")["target"].reindex(assets).fillna(0.0)
        gross_return = float((target * realized_returns).sum())
        turnover = float((target - prev_weights).abs().sum())
        transaction_cost = turnover * settings.transaction_cost_bps / 10_000.0
        strategy_return = gross_return - transaction_cost
        benchmark = settings.benchmark if settings.benchmark in quant_result.returns.columns else assets[0]
        benchmark_return = float(quant_result.returns.reindex([realized_date])[benchmark].fillna(0.0).iloc[0])

        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        realized_drawdown = equity / peak - 1.0
        cash_weight = float(max(0.0, 1.0 - target.sum()))

        perf_rows.append(
            {
                "date": realized_date,
                "signal_date": signal_ts,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "exposure": float(target.sum()),
                "cash_weight": cash_weight,
                "drawdown": realized_drawdown,
                "estimated_vol": est_vol,
                "estimated_var": est_var,
                "vol_scalar": vol_scalar,
                "var_scalar": var_scalar,
                "drawdown_scalar": dd_scalar,
                "turnover_scalar": turnover_scalar,
            }
        )

        day_alpha = supervised.loc[signal_ts].reindex(assets).fillna(0.0)
        day_llm = llm_weights.loc[signal_ts].reindex(assets).fillna(0.0)
        day_rl = rl_weights.loc[signal_ts].reindex(assets).fillna(0.0)
        for ticker in assets:
            decision_rows.append(
                {
                    "signal_date": signal_ts,
                    "realized_date": realized_date,
                    "ticker": ticker,
                    "supervised_weight": float(day_alpha[ticker]),
                    "llm_weight": float(day_llm[ticker]),
                    "rl_weight": float(day_rl[ticker]),
                    "raw_fused_weight": float(raw_fused.loc[signal_ts, ticker]),
                    "final_weight": float(target[ticker]),
                    "vol_scalar": vol_scalar,
                    "var_scalar": var_scalar,
                    "drawdown_scalar": dd_scalar,
                    "turnover_scalar": turnover_scalar,
                }
            )

        if vol_scalar < 0.999:
            event_rows.append({"date": realized_date, "event": "volatility_target", "detail": f"Exposure scaled by {vol_scalar:.0%}; estimated vol {est_vol:.2%}."})
        if var_scalar < 0.999:
            event_rows.append({"date": realized_date, "event": "var_throttle", "detail": f"Exposure scaled by {var_scalar:.0%}; estimated VaR {est_var:.2%}."})
        if dd_scalar < 0.999:
            event_rows.append({"date": realized_date, "event": "drawdown_guard", "detail": f"Exposure scaled by {dd_scalar:.0%}; prior drawdown {current_drawdown:.2%}."})
        if turnover_scalar < 0.999:
            event_rows.append({"date": realized_date, "event": "turnover_control", "detail": f"Target rebalance blended by {turnover_scalar:.0%} to respect turnover limit."})

        prev_weights = target

    raw_perf = pd.DataFrame(perf_rows).set_index("date").sort_index()
    performance = _build_performance(
        raw_perf["strategy_return"],
        raw_perf["benchmark_return"],
        raw_perf["turnover"],
        raw_perf["exposure"],
        raw_perf["cash_weight"],
    )
    for col in [
        "signal_date",
        "gross_return",
        "transaction_cost",
        "estimated_vol",
        "estimated_var",
        "vol_scalar",
        "var_scalar",
        "drawdown_scalar",
        "turnover_scalar",
    ]:
        performance[col] = raw_perf[col]

    metrics = compute_metrics(performance["strategy_return"], performance["benchmark_return"], performance["turnover"])
    metrics["avg_exposure"] = float(performance["exposure"].mean()) if not performance.empty else 0.0
    metrics["latest_exposure"] = float(performance["exposure"].iloc[-1]) if not performance.empty else 0.0
    metrics["benchmark_excess_total_return"] = metrics.get("total_return", 0.0) - metrics.get("benchmark_total_return", 0.0)

    risk_events = pd.DataFrame(event_rows, columns=["date", "event", "detail"])
    if not risk_events.empty:
        risk_events = risk_events.sort_values("date").reset_index(drop=True)

    decision_log = pd.DataFrame(decision_rows)
    suitability = _suitability_checks(performance, final_by_signal, settings)
    governance_warnings = [
        "This platform is for research and decision support only; it is not financial advice and does not place trades.",
        "Human confirmation is required before recommendations are accepted into an investment memo.",
    ]
    if (suitability["status"] == "Review").any():
        governance_warnings.append("One or more suitability checks require review before accepting the recommendation.")
    if not risk_events.empty:
        governance_warnings.append("Risk overlays were activated during the backtest; inspect the event ledger.")

    return AllocationResult(
        component_weights={
            "supervised_alpha": supervised,
            "llm_reasoning": llm_weights,
            "rl_policy": rl_weights,
            "raw_fused": raw_fused,
        },
        final_weights=final_by_signal,
        performance=performance,
        metrics=metrics,
        decision_log=decision_log,
        risk_events=risk_events,
        suitability_checks=suitability,
        governance_warnings=governance_warnings,
    )


def allocation_risk_series(allocation: AllocationResult) -> pd.DataFrame:
    return compute_risk_series(allocation.performance["strategy_return"], allocation.performance["benchmark_return"])
