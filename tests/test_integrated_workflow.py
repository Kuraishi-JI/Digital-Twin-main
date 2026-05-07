from __future__ import annotations

from market_digital_twin.data_foundation import load_demo_prices, validate_prices
from market_digital_twin.schemas import WorkflowSettings
from market_digital_twin.workflow import run_digital_twin_workflow


def test_offline_workflow_runs_without_api_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    prices = load_demo_prices(
        "2020-01-01",
        "2023-12-31",
        tickers=["SPY", "QQQ", "TLT", "GLD"],
        seed=7,
    )
    settings = WorkflowSettings(
        universe=tuple(prices.columns),
        benchmark="SPY",
        start_date="2020-01-01",
        end_date="2023-12-31",
        risk_profile="Balanced",
        demo_mode=True,
        llm_provider="OpenAI",
        llm_model="gpt-5.5",
        top_k=2,
        retrain_every=42,
    )

    result = run_digital_twin_workflow(prices, settings)

    assert result.allocation.performance.shape[0] > 100
    assert set(result.allocation.component_weights) == {
        "supervised_alpha",
        "llm_reasoning",
        "rl_policy",
        "raw_fused",
    }
    assert result.allocation.final_weights.sum(axis=1).max() <= 1.000001
    assert len(result.agent_reasoning.text_signals) == len(prices.columns)
    assert not result.market_state.graph.node_features.empty
    assert not result.environment.transitions.empty
    assert {"total_return", "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "var_95", "es_95", "win_rate"}.issubset(result.allocation.metrics)
    assert "decision-support simulation" in result.audit_report_markdown
    assert result.agent_reasoning.llm_selection.resolved_provider == "deterministic"


def test_price_schema_validation_reports_sample_shape():
    prices = load_demo_prices("2020-01-01", "2021-12-31", tickers=["SPY", "QQQ", "TLT"], seed=3)
    report = validate_prices(prices)

    assert report.valid
    assert report.assets == 3
    assert report.observations >= 260


def test_streamlit_app_imports():
    from market_digital_twin.app import APP_TITLE

    assert "Digital Twin" in APP_TITLE
