# Contributing

Thanks for helping improve the AI-Driven Market Digital Twin Platform.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest
python -m streamlit run app.py
```

## Contribution Guidelines

- Keep external APIs optional; offline research mode must run without keys.
- Add tests for any new workflow component, allocator rule, or governance behavior.
- Preserve the research/disclaimer framing. Do not add live brokerage execution.
- Prefer typed dataclasses or explicit schemas for cross-module contracts.
- Keep heavyweight LLM and RL dependencies behind optional extras.

## Pull Request Checklist

- Tests pass with `python -m pytest`.
- README or docs are updated when user-facing behavior changes.
- New risk logic includes an audit-trail entry or suitability check where relevant.
- The change does not require API keys for the default offline workflow.
