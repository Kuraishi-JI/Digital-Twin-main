# Algorithm Notes

The platform uses original, interpretable algorithmic components that are deliberately compact enough to audit and extend.

## Feature Construction

The feature stack includes short and medium momentum, mean reversion, moving-average distance, volatility, downside volatility, drawdown, beta, correlation, RSI, market momentum, market volatility, breadth, and dispersion.

Features are designed to support three goals:

- Explainable supervised alpha prediction.
- Risk-state awareness for portfolio controls.
- Market-state inputs for graph and simulation layers.

## Walk-Forward Prediction

The supervised alpha model is a standardized Ridge regression trained in a walk-forward loop. For each signal date, the model trains only on prior data, predicts next-period cross-sectional returns, and produces long-only alpha weights.

The walk-forward design reduces look-ahead bias and makes prediction quality reviewable through realized-return diagnostics and calibration buckets.

## Structured Reasoning

The reasoning engine organizes analysis as a Digital Twin Research Council:

- Market analyst
- News analyst
- Social analyst
- Fundamentals analyst
- Bull researcher
- Bear researcher
- Trader
- Risk analysts
- Portfolio manager

The final signal schema includes sentiment, confidence, risk flags, evidence, bias warnings, and investment rationale. Provider-backed LLM calls can enrich this layer, while offline structured reasoning keeps the workflow reproducible without external credentials.

## Market Graph

The graph connects assets by trailing correlation and shared sector relationships. Node features include degree, centrality proxy, and average absolute correlation.

This graph gives the allocator a lightweight dependency view so diversification and concentration can be evaluated beyond individual asset scores.

## Portfolio Simulation

State:
Structured market features, text insight signals, graph features, and current portfolio state.

Action:
Long-only portfolio weights.

Reward:

```text
portfolio return - transaction cost - volatility penalty - drawdown penalty
```

The environment is not a heavy training system by default. It provides a reproducible simulation ledger and a policy baseline that always runs locally.

## Fusion Allocator

The allocator blends:

- supervised alpha weights
- structured reasoning weights
- simulation policy weights

It then applies max single-asset weight, volatility targeting, VaR throttling, drawdown reduction, turnover control, and cash-buffer rules.

## Metrics

The platform computes total return, annual return, annual volatility, Sharpe, Sortino, maximum drawdown, Calmar, VaR 95, expected shortfall 95, turnover, exposure, benchmark comparison, and win rate.
