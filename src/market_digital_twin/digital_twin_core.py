from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .data_foundation import sector_map_for_universe
from .schemas import AgentReasoningBundle, MarketGraph, MarketState, WorkflowSettings


def build_market_graph(returns: pd.DataFrame, settings: WorkflowSettings) -> MarketGraph:
    assets = list(returns.columns)
    sectors = sector_map_for_universe(assets)
    history = returns[assets].tail(settings.graph_lookback).fillna(0.0)
    corr = history.corr().fillna(0.0)
    edges: list[dict[str, object]] = []

    for left, right in itertools.combinations(assets, 2):
        corr_value = float(corr.loc[left, right])
        same_sector = sectors[left] == sectors[right]
        if abs(corr_value) >= settings.correlation_threshold or same_sector:
            edges.append(
                {
                    "source": left,
                    "target": right,
                    "correlation": corr_value,
                    "abs_correlation": abs(corr_value),
                    "relationship": "sector" if same_sector else "correlation",
                }
            )

    edge_frame = pd.DataFrame(
        edges,
        columns=["source", "target", "correlation", "abs_correlation", "relationship"],
    )
    if edge_frame.empty:
        centrality = pd.Series(0.0, index=assets)
        avg_abs_corr = corr.abs().replace(1.0, np.nan).mean().fillna(0.0)
    else:
        centrality = pd.Series(0.0, index=assets)
        for row in edge_frame.itertuples(index=False):
            centrality.loc[row.source] += float(row.abs_correlation)
            centrality.loc[row.target] += float(row.abs_correlation)
        if len(assets) > 1:
            centrality = centrality / (len(assets) - 1)
        avg_abs_corr = corr.abs().replace(1.0, np.nan).mean().fillna(0.0)

    node_features = pd.DataFrame(
        {
            "ticker": assets,
            "sector": [sectors[ticker] for ticker in assets],
            "graph_centrality": centrality.reindex(assets).fillna(0.0).values,
            "avg_abs_correlation": avg_abs_corr.reindex(assets).fillna(0.0).values,
            "degree": [
                int(((edge_frame["source"] == ticker) | (edge_frame["target"] == ticker)).sum())
                if not edge_frame.empty
                else 0
                for ticker in assets
            ],
        }
    )
    summary = {
        "node_count": len(assets),
        "edge_count": len(edge_frame),
        "average_abs_correlation": float(avg_abs_corr.mean()) if len(avg_abs_corr) else 0.0,
        "threshold": settings.correlation_threshold,
    }
    return MarketGraph(edges=edge_frame, node_features=node_features, sector_map=sectors, summary=summary)


def build_market_state(quant_result, agent_reasoning: AgentReasoningBundle, graph: MarketGraph) -> MarketState:
    latest_date = pd.Timestamp(quant_result.predictions["date"].max())
    latest_predictions = quant_result.predictions.loc[quant_result.predictions["date"] == latest_date].copy()
    text_frame = agent_reasoning.signal_frame()
    feature_frame = (
        latest_predictions.merge(text_frame, on="ticker", how="left")
        .merge(graph.node_features, on="ticker", how="left")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    numeric_cols = [
        "prediction",
        "mom_20d",
        "mom_60d",
        "vol_20d",
        "drawdown_60d",
        "beta_60d",
        "rsi_14",
        "sentiment_score",
        "confidence",
        "graph_centrality",
        "avg_abs_correlation",
    ]
    for col in numeric_cols:
        if col not in feature_frame:
            feature_frame[col] = 0.0
        feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce").fillna(0.0)

    state_vector = feature_frame[numeric_cols].mean()
    context = quant_result.market_context
    regime = str(context.get("market_state", "transition"))
    return MarketState(
        as_of_date=latest_date,
        feature_frame=feature_frame,
        graph=graph,
        regime=regime,
        state_vector=state_vector,
    )
