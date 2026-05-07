# Platform Assistant

The Platform Assistant is a context-aware chatbot in the Streamlit sidebar. It helps users understand controls, workflow stages, financial metrics, portfolio outputs, risk indicators, explainability views, and audit report fields.

## How To Use It

Open the Streamlit app:

```bash
python -m streamlit run app.py
```

The assistant appears in the sidebar as **Platform Assistant**.

The sidebar contains its own **LLM Settings** module, independent from the Command Center workflow LLM settings.

Assistant LLM settings:

- Provider: OpenAI, Anthropic, or Google
- Model: filtered by selected provider
- API key: password-masked session key first, then provider environment variable

The assistant uses hosted LLMs only. If no key is available, it asks the user to configure a real LLM key.

## Conversation Controls

- **New** starts a separate conversation.
- **Clear** removes messages from the current conversation only.
- **Delete** asks for confirmation before deleting the selected conversation.
- **...** under a message reveals per-message controls.
- **Modify** edits a user question; **Regenerate** sends the revised question and replaces the previous answer.

Type directly into the sidebar chat input. For example:

- `Explain Sharpe ratio`
- `What is VaR 95?`
- `Feature importance 是什么？`
- `How should I choose Start date?`

## Covered Concepts

The context registry covers core controls and outputs, including:

- Start date, End date, Benchmark, Universe, Risk profile, sample data mode
- LLM provider, LLM model, API key
- Walk-forward prediction, Prediction, Realized return
- Portfolio weights, Exposure, Cash buffer, Turnover
- Sharpe, Sortino, Max drawdown, VaR 95, Expected Shortfall 95
- 20d volatility, Drawdown guard, Volatility targeting, VaR throttling
- Feature importance, Local contribution, What-if analysis
- Agent reasoning, Bull/bear debate, Risk debate, Final recommendation, Audit trail

## LLM Behavior

The assistant uses its own sidebar LLM Settings rather than the Command Center's LLM Settings:

- Provider: OpenAI, Anthropic, or Google
- Model: filtered by selected provider
- API key: UI-entered session key first, then provider environment variable

If no hosted-provider key is available, the assistant shows a clear warning and waits for a real key. If a provider call fails, the assistant reports the provider issue and asks the user to retry after checking credentials and model access.

## Privacy

API keys are used only for provider authentication. They are not included in prompts, chat history, reports, audit trails, checkpoints, or exported files. The assistant also redacts key-like strings from chat messages where possible.

## Context Matching

Arbitrary browser text selection is fragile in Streamlit without custom components, so the current product uses direct chat for explanations. Users can paste selected text into chat; if the text does not match a registered concept, the assistant provides a cautious general explanation grounded in platform behavior.
