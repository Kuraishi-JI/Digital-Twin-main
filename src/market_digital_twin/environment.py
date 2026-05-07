from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import AgentReasoningBundle, EnvironmentResult, MarketGraph, WorkflowSettings


def _normalize_positive(scores: pd.Series, max_weight: float) -> pd.Series:
    positive = scores.clip(lower=0.0)
    if positive.sum() <= 1e-12:
        positive = pd.Series(1.0, index=scores.index)
    weights = positive / positive.sum()
    for _ in range(12):
        above = weights > max_weight
        if not above.any():
            break
        excess = float((weights[above] - max_weight).sum())
        weights[above] = max_weight
        room = max_weight - weights[~above]
        room = room[room > 0.0]
        if room.empty or excess <= 1e-12:
            break
        weights.loc[room.index] += excess * room / room.sum()
    return weights / weights.sum()


def _standardize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        std = float(series.std())
        out[col] = 0.0 if not np.isfinite(std) or abs(std) < 1e-12 else (series - float(series.mean())) / std
    return out


def _policy_weights(day_frame: pd.DataFrame, text_scores: pd.Series, graph_scores: pd.Series, settings: WorkflowSettings) -> pd.Series:
    indexed = day_frame.set_index("ticker")
    z = _standardize(indexed, ["prediction", "mom_20d", "vol_20d", "drawdown_60d"])
    scores = (
        0.50 * z["prediction"]
        + 0.22 * z["mom_20d"]
        - 0.16 * z["vol_20d"]
        + 0.12 * z["drawdown_60d"]
        + 0.30 * text_scores.reindex(indexed.index).fillna(0.0)
        - 0.08 * graph_scores.reindex(indexed.index).fillna(0.0)
    )
    scores = scores - scores.min()
    return _normalize_positive(scores, settings.profile.max_weight)


def run_portfolio_environment(
    quant_result,
    agent_reasoning: AgentReasoningBundle,
    graph: MarketGraph,
    settings: WorkflowSettings,
) -> EnvironmentResult:
    assets = list(quant_result.returns.columns)
    signal_dates = sorted(pd.to_datetime(quant_result.predictions["date"].unique()))
    weights = pd.DataFrame(0.0, index=signal_dates, columns=assets)
    text_scores = agent_reasoning.signal_frame().set_index("ticker")["sentiment_score"]
    graph_scores = graph.node_features.set_index("ticker")["graph_centrality"]

    prev_weights = pd.Series(0.0, index=assets)
    equity = 1.0
    peak = 1.0
    rows: list[dict[str, object]] = []

    for signal_date, day_frame in quant_result.predictions.groupby("date", sort=True):
        signal_ts = pd.Timestamp(signal_date)
        realized_date = pd.Timestamp(day_frame["realized_date"].iloc[0])
        action = _policy_weights(day_frame, text_scores, graph_scores, settings).reindex(assets).fillna(0.0)

        exposure_cap = 1.0 - settings.profile.cash_buffer
        action = action * exposure_cap
        realized_returns = day_frame.set_index("ticker")["target"].reindex(assets).fillna(0.0)
        gross_return = float((action * realized_returns).sum())
        turnover = float((action - prev_weights).abs().sum())
        transaction_cost = turnover * settings.transaction_cost_bps / 10_000.0

        history = quant_result.returns.loc[quant_result.returns.index < signal_ts, assets].tail(60).fillna(0.0)
        realized_vol = float(history.mul(action, axis=1).sum(axis=1).std() * np.sqrt(252)) if len(history) >= 20 else 0.0
        current_drawdown = equity / peak - 1.0
        vol_penalty = settings.profile.volatility_penalty * max(0.0, realized_vol - settings.profile.target_volatility) / 252.0
        dd_penalty = settings.profile.drawdown_penalty * max(0.0, abs(current_drawdown) - settings.profile.drawdown_limit) / 252.0
        reward = gross_return - transaction_cost - vol_penalty - dd_penalty

        equity *= 1.0 + reward
        peak = max(peak, equity)
        weights.loc[signal_ts] = action
        rows.append(
            {
                "signal_date": signal_ts,
                "realized_date": realized_date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "volatility_penalty": vol_penalty,
                "drawdown_penalty": dd_penalty,
                "reward": reward,
                "turnover": turnover,
                "exposure": float(action.sum()),
                "equity": equity,
                "drawdown": equity / peak - 1.0,
            }
        )
        prev_weights = action

    transitions = pd.DataFrame(rows)
    reward_summary = {
        "average_reward": float(transitions["reward"].mean()) if not transitions.empty else 0.0,
        "reward_volatility": float(transitions["reward"].std()) if len(transitions) > 1 else 0.0,
        "average_turnover": float(transitions["turnover"].mean()) if not transitions.empty else 0.0,
        "average_exposure": float(transitions["exposure"].mean()) if not transitions.empty else 0.0,
    }
    return EnvironmentResult(weights=weights.sort_index(), transitions=transitions, reward_summary=reward_summary)
