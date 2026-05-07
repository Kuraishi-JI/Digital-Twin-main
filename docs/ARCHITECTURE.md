# Architecture

The repository is organized as a standalone original Python application with a clean product boundary, typed workflow contracts, tests, documentation, examples, and a Streamlit user interface.

```text
.
|-- app.py                         # Streamlit entry point
|-- pyproject.toml                  # Package metadata, dependencies, tooling
|-- README.md                       # Product overview and quickstart
|-- docs/                           # Architecture, algorithms, governance, LLM, assistant notes
|-- examples/                       # User-facing walkthroughs
|-- scripts/                        # Developer convenience commands
|-- src/market_digital_twin/        # Product package
|   |-- llm/                        # Hosted-provider registry and lightweight client
|   |-- quant/                      # Walk-forward quantitative research engine
|   `-- explainability/             # Model interpretation and audit helpers
`-- tests/                          # Workflow, LLM, and assistant tests
```

## Runtime Layers

1. Data Foundation Layer
   Loads sample or uploaded prices, validates schema, computes returns, benchmark context, sector metadata, and asset profiles.

2. Structured Reasoning Layer
   Produces Digital Twin Research Council analysis through provider-aware LLM configuration. The Command Center supports offline structured reasoning plus OpenAI, Anthropic, and Google. The sidebar assistant uses its own hosted-LLM settings.

3. Market Digital Twin Core
   Combines quantitative features, structured reasoning signals, and graph features into a market-state representation.

4. Portfolio Simulation Layer
   Implements a long-only simulation environment with explicit state, action, reward, turnover, exposure, and risk accounting.

5. Fusion Allocator
   Blends supervised alpha, reasoning scores, and simulation policy weights before applying risk overlays.

6. Governance Layer
   Adds suitability checks, risk warnings, human confirmation, and an audit-ready report.

## Product Boundary

The platform is not a live-trading engine. It creates decision-support evidence for human review. The application boundary ends at research memo generation, allocation simulation, audit export, and governance confirmation.

## Original Design Position

The product is designed around one core idea: every research allocation should be traceable across market data, quantitative features, structured reasoning, graph state, simulated transitions, risk controls, and governance evidence.

Each layer can be extended independently while preserving the same end-to-end workflow contract.
