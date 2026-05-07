# LLM Configuration

The platform has two separate LLM configuration surfaces:

- Command Center LLM Settings for the main research workflow.
- Platform Assistant LLM Settings in the sidebar for user help.

This separation lets the workflow run in offline research mode while the assistant can use a hosted LLM, or vice versa.

## Command Center Providers

| Provider | Models | Environment fallback |
|---|---|---|
| `deterministic` | `offline-mock` | None |
| OpenAI | `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-4.1` | `OPENAI_API_KEY` |
| Anthropic | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| Google | `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` | `GOOGLE_API_KEY` |

The provider dropdown is shown first. The model dropdown is filtered to the selected provider, and invalid provider/model pairs are rejected by `validate_provider_model`.

The internal `deterministic` provider powers offline structured reasoning for reproducible local runs. In the UI, this is presented as an offline simulation option rather than a hosted LLM.

## Platform Assistant Providers

The sidebar assistant supports hosted LLM providers only:

- OpenAI
- Anthropic
- Google

It does not expose the internal offline provider. If no assistant key is available, the assistant asks for a real key and does not generate an offline assistant answer.

## API Key Handling

Hosted providers have a password-masked API key input in the Streamlit UI. Key resolution order is:

1. UI-entered key for the current Streamlit session.
2. Matching environment variable.
3. Offline structured reasoning for the Command Center workflow when fallback is enabled.

UI-entered keys are not written to `.env`, config files, memory logs, reports, checkpoints, chat history, or JSON outputs. Reports include provider/model metadata but never secrets.

## Offline Workflow Behavior

The full research workflow runs without API keys. If no key is entered and the environment variable is missing, the app shows a warning and uses offline structured reasoning for the narrative layer.

If a provider call fails at runtime, the workflow records a warning and uses offline structured reasoning for that run. Portfolio, risk, explainability, and audit-report generation continue.

## Examples

OpenAI with a UI key:

```powershell
python -m streamlit run app.py
```

Choose `OpenAI`, select an OpenAI model, and paste the key into the password field.

OpenAI with an environment variable:

```powershell
$env:OPENAI_API_KEY="sk-..."
python -m streamlit run app.py
```

Anthropic with an environment variable:

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
python -m streamlit run app.py
```

Google with an environment variable:

```powershell
$env:GOOGLE_API_KEY="..."
python -m streamlit run app.py
```

Offline workflow run:

```powershell
python -m streamlit run app.py
```

Select the offline simulation provider in the Command Center or leave hosted-provider credentials unset with fallback enabled.

## Updating The Static Registry

The model registry lives in:

```text
src/market_digital_twin/llm/catalog.py
```

To update supported models:

1. Edit `MODEL_CATALOG`.
2. Keep provider IDs limited to `deterministic`, `openai`, `anthropic`, and `google` unless the product scope changes.
3. Mark one model per provider with `is_default=True`.
4. Update `tests/test_llm_catalog.py`.
5. Run `python -m pytest`.
