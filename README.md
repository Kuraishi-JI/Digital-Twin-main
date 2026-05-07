# AI-Driven Market Digital Twin Platform

A complete original AI investment research platform for human-in-the-loop fund decision support.

> AI-Driven Market Digital Twins: Toward the Next Generation of Fund Investment

This repository delivers a self-contained Market Digital Twin system that turns market data, quantitative features, structured LLM reasoning, portfolio simulation, risk controls, explainability, and governance review into one coherent Streamlit product.

The platform is intentionally designed for research, education, and investment committee discussion. It does not provide personalized financial advice, connect to brokerage systems, or execute trades.

## Quickstart

```bash
pip install -e ".[dev]"
python -m pytest
python -m streamlit run app.py
```

The default workflow runs in offline simulation mode without API keys. If you prefer a non-editable install, `pip install -r requirements.txt` also installs the runtime dependencies.

## Product Capabilities

- Command Center for universe, benchmark, date range, risk profile, model weights, and LLM settings.
- Digital Twin Workflow view showing the full research path from data validation to governance status.
- Agent Reasoning view with market, news, sentiment, fundamentals, bull/bear, trader, risk, and portfolio-manager perspectives.
- Portfolio Lab with final weights, component weights, equity curve, benchmark comparison, turnover, exposure, and cash behavior.
- Risk & Governance layer with volatility, VaR, expected shortfall, drawdown, suitability checks, warnings, override notes, and human confirmation.
- Explainability & Audit layer with feature importance, local contribution analysis, calibration buckets, what-if analysis, reasoning trace, and downloadable report.
- Sidebar Platform Assistant for context-aware help using its own OpenAI, Anthropic, or Google LLM settings.

## Repository Structure

```text
app.py
pyproject.toml
docs/
examples/
scripts/
src/market_digital_twin/
|-- data_foundation.py          # Sample/CSV data loading, validation, sector metadata
|-- agent_adapter.py            # Digital Twin Research Council reasoning engine
|-- digital_twin_core.py        # Market-state representation and dependency graph
|-- environment.py              # Portfolio simulation environment
|-- allocator.py                # Alpha + reasoning + simulation fusion and risk overlays
|-- workflow.py                 # End-to-end orchestration and audit report
|-- schemas.py                  # Typed workflow contracts
|-- quant/                      # Walk-forward quantitative research engine
|-- explainability/             # Model interpretation and audit helpers
|-- llm/                        # Provider registry and lightweight LLM client
`-- app.py                      # Streamlit six-tab product interface
tests/
```

See `docs/ARCHITECTURE.md`, `docs/ALGORITHMS.md`, `docs/GOVERNANCE.md`, `docs/LLM_CONFIGURATION.md`, and `docs/PLATFORM_ASSISTANT.md` for implementation notes.

## Platform Assistant

The Streamlit sidebar includes a context-aware **Platform Assistant** chatbot. It has its own sidebar **LLM Settings** module, separate from the Command Center workflow settings.

The assistant supports hosted LLM providers only:

- OpenAI
- Anthropic
- Google

Enter a password-masked session key in the sidebar, or set the matching environment variable before launching Streamlit. If no assistant API key is available, the sidebar asks for a real LLM key instead of generating offline assistant text. API keys are never written to prompts, chat history, logs, reports, or exported files.

The assistant supports multiple conversations. Use **New** to start another chat, **Clear** to clear the current chat, and **Delete** to remove the selected conversation after confirmation. Each message also has a hidden **...** menu for deleting a single message; user questions can be modified and regenerated.

## Algorithm Principles

Feature construction:
The platform uses an interpretable feature stack built for this product: return, momentum, mean reversion, moving-average distance, volatility, downside volatility, drawdown, beta, correlation, RSI, breadth, dispersion, and benchmark regime features.

Walk-forward prediction:
The supervised alpha model is a standardized Ridge regression trained through a walk-forward loop. It predicts next-period cross-sectional returns and converts positive forecasts into long-only target weights.

Structured reasoning:
The Digital Twin Research Council converts quantitative state into structured market, sentiment, risk, debate, and portfolio rationale signals. The reasoning schema includes `sentiment_score`, `confidence`, `risk_flags`, `evidence`, `bias_warning`, and `investment_rationale`.

Market graph:
The digital twin core builds a dependency graph using trailing correlations and sector relationships. Node features include sector, degree, graph centrality, and average absolute correlation.

Portfolio simulation:
The environment represents state as market features, text insight signals, graph features, and current portfolio state. Actions are long-only portfolio weights. Reward is:

```text
reward = portfolio_return
       - transaction_cost
       - volatility_penalty
       - drawdown_penalty
```

Fusion allocator:
The allocator blends supervised alpha weights, structured reasoning weights, and simulation policy weights. It then applies max single-asset weight, volatility targeting, VaR throttling, drawdown exposure reduction, turnover control, and cash-buffer rules.

Metrics:
The platform computes total return, annual return, annual volatility, Sharpe, Sortino, maximum drawdown, Calmar, VaR 95, expected shortfall 95, turnover, exposure, benchmark comparison, and win rate.

Governance:
The UI records suitability checks, abnormal-risk warnings, manual override notes, and explicit human confirmation before a research recommendation is accepted as memo input.

## Offline Research Mode

Offline research mode is the default workflow. It uses reproducible synthetic multi-asset data and offline structured reasoning, so the full product can run without:

- API keys
- internet access
- external market-data vendors
- live trading connections

This mode is intended for classroom use, reproducibility checks, UI review, and local development. Uploaded CSV data and provider-backed LLM reasoning can be enabled when available.

## LLM Configuration

The Command Center LLM Settings panel supports offline simulation reasoning plus OpenAI, Anthropic, and Google provider-backed reasoning for the main workflow. The Platform Assistant has a separate sidebar LLM Settings module and supports OpenAI, Anthropic, and Google only.

Command Center offline mode:

- provider id: `deterministic`
- model id: `offline-mock`

OpenAI:

- `gpt-5.5`
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-4.1`

Anthropic:

- `claude-opus-4-7`
- `claude-sonnet-4-6`
- `claude-haiku-4-5`

Google:

- `gemini-3.1-pro-preview`
- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite-preview`
- `gemini-2.5-pro`
- `gemini-2.5-flash`

Environment variable alternatives:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

UI-entered keys are used only for the current runtime session. They are not written to disk, `.env`, reports, audit trails, memory logs, checkpoints, or JSON outputs.

## Original Product Identity

This project is an original software system and product design. Its value is not a wrapper around another repository; it is the end-to-end research workflow: transparent feature engineering, structured financial reasoning, market-state digital twin construction, portfolio simulation, explainable allocation, and governance evidence in one auditable application.

The architecture is modular so future researchers can replace individual components such as feature builders, LLM prompts, graph construction, reward design, or risk overlays without changing the product boundary.

## Testing

```bash
python -m pytest
```

The test suite verifies that the full offline workflow runs without API keys and produces reasoning signals, graph features, portfolio-environment transitions, fused weights, risk metrics, governance checks, assistant behavior, LLM configuration handling, and an audit report.

## Disclaimer

This software is for research, education, and decision support only. It is not financial advice, does not recommend securities for real-money action, does not connect to a broker, and does not place trades. Any real investment process requires independent verification, suitability review, compliance approval, and human accountability.
