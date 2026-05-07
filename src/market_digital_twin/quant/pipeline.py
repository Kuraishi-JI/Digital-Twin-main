from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import compute_returns

FEATURE_COLUMNS = [
    "ret_1d",
    "mom_5d",
    "mom_20d",
    "mom_60d",
    "mean_reversion_10d",
    "dist_ma20",
    "dist_ma60",
    "vol_20d",
    "downside_20d",
    "drawdown_60d",
    "beta_60d",
    "corr_60d",
    "rsi_14",
    "market_mom_20d",
    "market_vol_20d",
    "market_drawdown_60d",
    "breadth_20d",
    "dispersion_20d",
]


@dataclass(frozen=True)
class QuantResearchConfig:
    benchmark: str = "SPY"
    train_ratio: float = 0.65
    retrain_every: int = 21
    top_k: int = 3
    prediction_horizon: int = 1
    model_alpha: float = 1.0
    max_weight: float = 0.40
    target_vol: float = 0.18
    var_limit: float = 0.025
    var_confidence: float = 0.95
    drawdown_limit: float = 0.12
    drawdown_exposure: float = 0.35
    transaction_cost_bps: float = 8.0
    covariance_lookback: int = 60


@dataclass
class QuantResearchResult:
    config: QuantResearchConfig
    prices: pd.DataFrame
    returns: pd.DataFrame
    panel: pd.DataFrame
    predictions: pd.DataFrame
    latest_model: Pipeline
    model_snapshots: dict[pd.Timestamp, Pipeline]
    feature_columns: list[str]
    signal_weights: pd.DataFrame
    applied_weights: pd.DataFrame
    decision_log: pd.DataFrame
    performance: pd.DataFrame
    metrics: dict[str, float]
    risk_series: pd.DataFrame
    risk_events: pd.DataFrame
    daily_ic: pd.DataFrame
    market_context: dict[str, Any]
    crisis_summary: dict[str, Any]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gains = delta.clip(lower=0.0).rolling(period).mean()
    losses = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gains / losses.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _build_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def build_feature_panel(prices: pd.DataFrame, benchmark: str, prediction_horizon: int = 1) -> pd.DataFrame:
    """Create an interpretable cross-sectional feature panel."""
    if benchmark not in prices.columns:
        benchmark = prices.columns[0]

    prices = prices.sort_index().ffill()
    returns = compute_returns(prices)

    benchmark_price = prices[benchmark]
    benchmark_ret = returns[benchmark]
    breadth_20d = prices.div(prices.rolling(20).mean()).gt(1.0).mean(axis=1)
    dispersion_20d = returns.std(axis=1).rolling(20).mean()
    market_mom_20d = benchmark_price.pct_change(20)
    market_vol_20d = benchmark_ret.rolling(20).std() * np.sqrt(252)
    market_drawdown_60d = benchmark_price.div(benchmark_price.rolling(60).max()) - 1.0

    frames = []
    for ticker in prices.columns:
        price = prices[ticker]
        ret = returns[ticker]
        frame = pd.DataFrame(
            {
                "date": prices.index,
                "ticker": ticker,
                "ret_1d": ret.reindex(prices.index),
                "mom_5d": price.pct_change(5),
                "mom_20d": price.pct_change(20),
                "mom_60d": price.pct_change(60),
                "mean_reversion_10d": -(price.pct_change(5) - price.pct_change(20)),
                "dist_ma20": price.div(price.rolling(20).mean()) - 1.0,
                "dist_ma60": price.div(price.rolling(60).mean()) - 1.0,
                "vol_20d": ret.rolling(20).std() * np.sqrt(252),
                "downside_20d": ret.clip(upper=0.0).rolling(20).std() * np.sqrt(252),
                "drawdown_60d": price.div(price.rolling(60).max()) - 1.0,
                "beta_60d": ret.rolling(60).cov(benchmark_ret) / benchmark_ret.rolling(60).var(),
                "corr_60d": ret.rolling(60).corr(benchmark_ret),
                "rsi_14": _compute_rsi(price),
                "market_mom_20d": market_mom_20d.reindex(prices.index),
                "market_vol_20d": market_vol_20d.reindex(prices.index),
                "market_drawdown_60d": market_drawdown_60d.reindex(prices.index),
                "breadth_20d": breadth_20d.reindex(prices.index),
                "dispersion_20d": dispersion_20d.reindex(prices.index),
                "target": ret.shift(-prediction_horizon).reindex(prices.index),
            }
        )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["target"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


def walk_forward_predictions(panel: pd.DataFrame, config: QuantResearchConfig) -> tuple[pd.DataFrame, dict[pd.Timestamp, Pipeline], Pipeline]:
    """Run a lightweight walk-forward training loop."""
    unique_dates = np.array(sorted(panel["date"].unique()))
    min_train_days = max(140, int(len(unique_dates) * config.train_ratio))
    if len(unique_dates) <= min_train_days + 5:
        raise ValueError("Not enough history to run the walk-forward research pipeline.")

    prediction_frames = []
    model_snapshots: dict[pd.Timestamp, Pipeline] = {}
    model: Pipeline | None = None
    latest_model: Pipeline | None = None

    for idx in range(min_train_days, len(unique_dates)):
        signal_date = pd.Timestamp(unique_dates[idx])
        should_refit = model is None or (idx - min_train_days) % config.retrain_every == 0
        if should_refit:
            train_mask = panel["date"] < signal_date
            x_train = panel.loc[train_mask, FEATURE_COLUMNS]
            y_train = panel.loc[train_mask, "target"]
            model = _build_model(config.model_alpha)
            model.fit(x_train, y_train)
            model_snapshots[signal_date] = model
            latest_model = model

        day_frame = panel.loc[panel["date"] == signal_date, ["date", "ticker", "target"] + FEATURE_COLUMNS].copy()
        if day_frame.empty:
            continue
        next_dates = unique_dates[idx + config.prediction_horizon : idx + config.prediction_horizon + 1]
        if len(next_dates) == 0:
            break
        day_frame["realized_date"] = pd.Timestamp(next_dates[0])
        day_frame["prediction"] = model.predict(day_frame[FEATURE_COLUMNS])
        prediction_frames.append(day_frame)

    if not prediction_frames or latest_model is None:
        raise ValueError("Walk-forward loop did not produce any predictions.")

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return predictions, model_snapshots, latest_model


def _cap_long_only_weights(weights: pd.Series, cap: float) -> pd.Series:
    if weights.empty or float(weights.sum()) <= 0.0:
        return weights.copy()

    capped = weights.astype(float).copy()
    capped = capped / capped.sum()
    for _ in range(10):
        above_cap = capped > cap
        if not above_cap.any():
            break
        excess = float((capped[above_cap] - cap).sum())
        capped[above_cap] = cap
        room = cap - capped[~above_cap]
        room = room[room > 0.0]
        if room.empty or excess <= 1e-12:
            break
        capped.loc[room.index] += excess * room / room.sum()
    total = float(capped.sum())
    return capped / total if total > 0 else capped


def generate_signal_weights(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    config: QuantResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Translate predictions into long-only research target weights plus risk overlays."""
    assets = sorted(predictions["ticker"].unique())
    signal_dates = sorted(predictions["date"].unique())
    weights_frame = pd.DataFrame(0.0, index=pd.to_datetime(signal_dates), columns=assets)
    decision_frames = []
    event_rows = []

    alpha = 1.0 - config.var_confidence
    for signal_date, day_frame in predictions.groupby("date", sort=True):
        ranked = day_frame.sort_values(["prediction", "ticker"], ascending=[False, True]).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        selected = ranked.head(config.top_k).copy()
        positive = selected[selected["prediction"] > 0.0].copy()

        raw_weights = pd.Series(dtype=float)
        capped_weights = pd.Series(dtype=float)
        if not positive.empty and positive["prediction"].sum() > 0:
            raw_weights = pd.Series(positive["prediction"].values, index=positive["ticker"].values, dtype=float)
            raw_weights = raw_weights / raw_weights.sum()
            capped_weights = _cap_long_only_weights(raw_weights, config.max_weight)

        base_weights = pd.Series(0.0, index=assets, dtype=float)
        if not capped_weights.empty:
            base_weights.loc[capped_weights.index] = capped_weights

        history = returns.loc[returns.index < pd.Timestamp(signal_date), assets].tail(config.covariance_lookback).fillna(0.0)
        est_vol = np.nan
        est_var = np.nan
        vol_scalar = 1.0
        var_scalar = 1.0
        if len(history) >= 20 and base_weights.sum() > 0:
            history_port = history.mul(base_weights, axis=1).sum(axis=1)
            est_vol = float(history_port.std() * np.sqrt(252))
            est_var = float(-np.quantile(history_port, alpha))
            if np.isfinite(est_vol) and est_vol > 1e-12:
                vol_scalar = min(1.0, config.target_vol / est_vol)
            if np.isfinite(est_var) and est_var > 1e-12:
                var_scalar = min(1.0, config.var_limit / est_var)

        scalar = min(vol_scalar, var_scalar)
        scaled_weights = base_weights * scalar
        weights_frame.loc[pd.Timestamp(signal_date)] = scaled_weights

        ranked["raw_weight"] = ranked["ticker"].map(raw_weights).fillna(0.0)
        ranked["capped_weight"] = ranked["ticker"].map(capped_weights).fillna(0.0)
        ranked["pre_drawdown_weight"] = ranked["ticker"].map(scaled_weights).fillna(0.0)
        ranked["selected"] = ranked["ticker"].isin(capped_weights.index)
        ranked["vol_scalar"] = vol_scalar
        ranked["var_scalar"] = var_scalar
        ranked["estimated_vol"] = est_vol
        ranked["estimated_var"] = est_var
        decision_frames.append(ranked)

        if vol_scalar < 0.999:
            event_rows.append(
                {
                    "date": pd.Timestamp(signal_date),
                    "event": "vol_target",
                    "detail": f"Exposure scaled to {vol_scalar:.0%} because forecast vol was {est_vol:.2%}.",
                }
            )
        if var_scalar < 0.999:
            event_rows.append(
                {
                    "date": pd.Timestamp(signal_date),
                    "event": "var_limit",
                    "detail": f"Exposure scaled to {var_scalar:.0%} because 1-day VaR was {est_var:.2%}.",
                }
            )

    decisions = pd.concat(decision_frames, ignore_index=True).sort_values(["date", "rank", "ticker"]).reset_index(drop=True)
    events = pd.DataFrame(event_rows)
    return weights_frame, decisions, events


def _build_performance_frame(strategy_returns: pd.Series, benchmark_returns: pd.Series, turnover: pd.Series, cash_weight: pd.Series) -> pd.DataFrame:
    equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    performance = pd.DataFrame(
        {
            "strategy_return": strategy_returns,
            "benchmark_return": benchmark_returns,
            "equity": equity,
            "benchmark_equity": benchmark_equity,
            "drawdown": drawdown,
            "turnover": turnover,
            "cash_weight": cash_weight,
        }
    )
    performance.index.name = "date"
    return performance


def compute_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series, turnover: pd.Series) -> dict[str, float]:
    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0.0)
    if strategy_returns.empty:
        return {}

    total_return = float((1.0 + strategy_returns).prod() - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / len(strategy_returns)) - 1.0)
    annual_vol = float(strategy_returns.std() * np.sqrt(252))
    downside = strategy_returns[strategy_returns < 0.0].std() * np.sqrt(252)
    sharpe = _safe_ratio(strategy_returns.mean() * np.sqrt(252), strategy_returns.std())
    sortino = _safe_ratio(annual_return, downside)
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    max_drawdown = float(drawdown.min())
    calmar = _safe_ratio(annual_return, abs(max_drawdown))
    var_95 = float(-np.quantile(strategy_returns, 0.05))
    es_95 = float(-strategy_returns[strategy_returns <= np.quantile(strategy_returns, 0.05)].mean())
    benchmark_total_return = float((1.0 + benchmark_returns).prod() - 1.0)

    beta = _safe_ratio(strategy_returns.cov(benchmark_returns), benchmark_returns.var())
    correlation = float(strategy_returns.corr(benchmark_returns))

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": float((strategy_returns > 0.0).mean()),
        "avg_turnover": float(turnover.mean()),
        "var_95": var_95,
        "es_95": es_95,
        "beta_to_benchmark": beta,
        "corr_to_benchmark": correlation,
        "benchmark_total_return": benchmark_total_return,
    }


def compute_risk_series(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    roll_vol = strategy_returns.rolling(20).std() * np.sqrt(252)
    roll_var = -strategy_returns.rolling(60).quantile(0.05)
    roll_es = strategy_returns.rolling(60).apply(
        lambda x: -x[x <= np.quantile(x, 0.05)].mean() if len(x) and np.isfinite(x).all() else np.nan,
        raw=False,
    )
    roll_corr = strategy_returns.rolling(60).corr(benchmark_returns)
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    return pd.DataFrame(
        {
            "rolling_vol_20d": roll_vol,
            "rolling_var_60d": roll_var,
            "rolling_es_60d": roll_es,
            "rolling_corr_60d": roll_corr,
            "drawdown": drawdown,
        }
    )


def compute_daily_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_date, day_frame in predictions.groupby("date", sort=True):
        if day_frame["prediction"].nunique() <= 1 or day_frame["target"].nunique() <= 1:
            ic = np.nan
        else:
            ic = spearmanr(day_frame["prediction"], day_frame["target"]).correlation
        rows.append({"date": pd.Timestamp(signal_date), "ic": ic})
    return pd.DataFrame(rows).set_index("date")


def summarize_market_context(panel: pd.DataFrame, benchmark: str) -> dict[str, Any]:
    if panel.empty:
        return {}

    latest_date = pd.Timestamp(panel["date"].max())
    latest_slice = panel.loc[panel["date"] == latest_date].copy()
    if latest_slice.empty:
        return {}

    benchmark_slice = latest_slice.loc[latest_slice["ticker"] == benchmark]
    benchmark_row = benchmark_slice.iloc[0] if not benchmark_slice.empty else latest_slice.iloc[0]

    market_mom = float(benchmark_row["market_mom_20d"])
    market_vol = float(benchmark_row["market_vol_20d"])
    market_drawdown = float(benchmark_row["market_drawdown_60d"])
    breadth = float(benchmark_row["breadth_20d"])
    dispersion = float(benchmark_row["dispersion_20d"])

    if market_drawdown <= -0.10 or market_vol >= 0.24:
        regime = "risk_off"
    elif market_mom >= 0.0 and breadth >= 0.50:
        regime = "risk_on"
    else:
        regime = "transition"

    return {
        "date": latest_date,
        "benchmark": benchmark_row["ticker"],
        "market_state": regime,
        "market_mom_20d": market_mom,
        "market_vol_20d": market_vol,
        "market_drawdown_60d": market_drawdown,
        "breadth_20d": breadth,
        "dispersion_20d": dispersion,
        "average_asset_vol_20d": float(latest_slice["vol_20d"].mean()),
    }


def find_crisis_window(performance: pd.DataFrame, window: int = 30) -> dict[str, Any]:
    if performance.empty or len(performance) < window:
        return {}
    benchmark_window = (1.0 + performance["benchmark_return"]).rolling(window).apply(np.prod, raw=True) - 1.0
    end_date = benchmark_window.idxmin()
    start_pos = max(performance.index.get_loc(end_date) - window + 1, 0)
    start_date = performance.index[start_pos]
    crisis_slice = performance.loc[start_date:end_date]
    return {
        "start": pd.Timestamp(start_date),
        "end": pd.Timestamp(end_date),
        "benchmark_window_return": float((1.0 + crisis_slice["benchmark_return"]).prod() - 1.0),
        "strategy_window_return": float((1.0 + crisis_slice["strategy_return"]).prod() - 1.0),
        "strategy_window_drawdown": float(crisis_slice["drawdown"].min()),
        "window_days": int(len(crisis_slice)),
    }


def run_backtest(
    predictions: pd.DataFrame,
    signal_weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: QuantResearchConfig,
    benchmark: str,
    overlay_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply one-day-ahead returns to the target weights."""
    assets = list(signal_weights.columns)
    benchmark = benchmark if benchmark in returns.columns else returns.columns[0]

    prev_weights = pd.Series(0.0, index=assets, dtype=float)
    equity = 1.0
    peak = 1.0
    perf_rows = []
    applied_rows = []
    risk_rows = []

    for signal_date, day_frame in predictions.groupby("date", sort=True):
        realized_date = pd.Timestamp(day_frame["realized_date"].iloc[0])
        target_weights = signal_weights.loc[pd.Timestamp(signal_date)].astype(float)
        current_drawdown = equity / peak - 1.0
        dd_scalar = config.drawdown_exposure if current_drawdown <= -config.drawdown_limit else 1.0
        applied_weights = target_weights * dd_scalar
        cash_weight = float(max(0.0, 1.0 - applied_weights.sum()))

        realized_returns = day_frame.set_index("ticker")["target"].reindex(assets).fillna(0.0)
        gross_return = float((applied_weights * realized_returns).sum())
        turnover = float((applied_weights - prev_weights).abs().sum())
        transaction_cost = turnover * (config.transaction_cost_bps / 10_000.0)
        strategy_return = gross_return - transaction_cost

        benchmark_return = float(returns.reindex([realized_date])[benchmark].fillna(0.0).iloc[0])

        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        realized_drawdown = equity / peak - 1.0

        perf_rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "date": realized_date,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "cash_weight": cash_weight,
                "exposure": float(applied_weights.sum()),
                "dd_scalar": dd_scalar,
                "drawdown": realized_drawdown,
            }
        )

        for ticker, weight in applied_weights.items():
            applied_rows.append({"date": realized_date, "ticker": ticker, "final_weight": float(weight)})

        if dd_scalar < 0.999:
            risk_rows.append(
                {
                    "date": realized_date,
                    "event": "drawdown_guard",
                    "detail": f"Exposure cut to {dd_scalar:.0%} after portfolio drawdown breached {config.drawdown_limit:.0%}.",
                }
            )

        prev_weights = applied_weights

    performance = pd.DataFrame(perf_rows).set_index("date").sort_index()
    applied = pd.DataFrame(applied_rows)
    applied_weights = applied.pivot(index="date", columns="ticker", values="final_weight").fillna(0.0).sort_index()

    events = pd.concat([overlay_events, pd.DataFrame(risk_rows)], ignore_index=True) if not overlay_events.empty or risk_rows else pd.DataFrame(columns=["date", "event", "detail"])
    if not events.empty:
        events["date"] = pd.to_datetime(events["date"])
        events = events.sort_values("date").reset_index(drop=True)

    strategy_returns = performance["strategy_return"]
    benchmark_returns = performance["benchmark_return"]
    turnover = performance["turnover"]
    cash_weight = performance["cash_weight"]
    base_performance = _build_performance_frame(strategy_returns, benchmark_returns, turnover, cash_weight)
    base_performance["signal_date"] = performance["signal_date"]
    base_performance["gross_return"] = performance["gross_return"]
    base_performance["transaction_cost"] = performance["transaction_cost"]
    base_performance["exposure"] = performance["exposure"]
    base_performance["dd_scalar"] = performance["dd_scalar"]
    return base_performance, applied_weights, events


def run_quant_research(prices: pd.DataFrame, config: QuantResearchConfig) -> QuantResearchResult:
    """End-to-end quantitative research orchestration."""
    clean_prices = prices.sort_index().ffill().dropna(how="all")
    benchmark = config.benchmark if config.benchmark in clean_prices.columns else clean_prices.columns[0]
    returns = compute_returns(clean_prices)
    panel = build_feature_panel(clean_prices, benchmark, config.prediction_horizon)
    predictions, model_snapshots, latest_model = walk_forward_predictions(panel, config)
    signal_weights, decisions, overlay_events = generate_signal_weights(predictions, returns, config)
    performance, applied_weights, risk_events = run_backtest(predictions, signal_weights, returns, config, benchmark, overlay_events)
    metrics = compute_metrics(performance["strategy_return"], performance["benchmark_return"], performance["turnover"])
    risk_series = compute_risk_series(performance["strategy_return"], performance["benchmark_return"])
    daily_ic = compute_daily_ic(predictions)
    market_context = summarize_market_context(panel, benchmark)
    crisis_summary = find_crisis_window(performance)

    final_weights_long = applied_weights.stack().rename("final_weight").reset_index()
    final_weights_long.columns = ["realized_date", "ticker", "final_weight"]
    decision_log = decisions.rename(columns={"date": "signal_date"}).merge(
        final_weights_long,
        left_on=["realized_date", "ticker"],
        right_on=["realized_date", "ticker"],
        how="left",
    )
    decision_log["final_weight"] = decision_log["final_weight"].fillna(0.0)
    decision_log = decision_log.merge(
        performance.reset_index()[["date", "cash_weight", "exposure", "dd_scalar"]],
        left_on="realized_date",
        right_on="date",
        how="left",
    ).drop(columns=["date"])

    return QuantResearchResult(
        config=config,
        prices=clean_prices,
        returns=returns,
        panel=panel,
        predictions=predictions,
        latest_model=latest_model,
        model_snapshots=model_snapshots,
        feature_columns=list(FEATURE_COLUMNS),
        signal_weights=signal_weights,
        applied_weights=applied_weights,
        decision_log=decision_log,
        performance=performance,
        metrics=metrics,
        risk_series=risk_series,
        risk_events=risk_events,
        daily_ic=daily_ic,
        market_context=market_context,
        crisis_summary=crisis_summary,
    )
