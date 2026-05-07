from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from .llm.catalog import ResolvedLLMSelection


RiskProfileName = Literal["Conservative", "Balanced", "Growth"]


@dataclass(frozen=True)
class RiskProfile:
    name: RiskProfileName
    max_weight: float
    target_volatility: float
    var_limit: float
    drawdown_limit: float
    drawdown_exposure: float
    cash_buffer: float
    max_turnover: float
    volatility_penalty: float
    drawdown_penalty: float


RISK_PROFILES: dict[RiskProfileName, RiskProfile] = {
    "Conservative": RiskProfile(
        name="Conservative",
        max_weight=0.28,
        target_volatility=0.12,
        var_limit=0.018,
        drawdown_limit=0.08,
        drawdown_exposure=0.25,
        cash_buffer=0.12,
        max_turnover=0.35,
        volatility_penalty=1.10,
        drawdown_penalty=1.35,
    ),
    "Balanced": RiskProfile(
        name="Balanced",
        max_weight=0.36,
        target_volatility=0.18,
        var_limit=0.026,
        drawdown_limit=0.12,
        drawdown_exposure=0.38,
        cash_buffer=0.06,
        max_turnover=0.55,
        volatility_penalty=0.75,
        drawdown_penalty=1.00,
    ),
    "Growth": RiskProfile(
        name="Growth",
        max_weight=0.48,
        target_volatility=0.24,
        var_limit=0.036,
        drawdown_limit=0.16,
        drawdown_exposure=0.55,
        cash_buffer=0.02,
        max_turnover=0.75,
        volatility_penalty=0.45,
        drawdown_penalty=0.70,
    ),
}


@dataclass(frozen=True)
class WorkflowSettings:
    universe: tuple[str, ...]
    benchmark: str
    start_date: str
    end_date: str
    risk_profile: RiskProfileName = "Balanced"
    demo_mode: bool = True
    llm_provider: str = "deterministic"
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False, compare=False)
    allow_demo_fallback: bool = True
    enable_checkpoint: bool = True
    alpha_weight: float = 0.45
    llm_weight: float = 0.25
    rl_weight: float = 0.30
    top_k: int = 3
    retrain_every: int = 21
    model_alpha: float = 1.2
    transaction_cost_bps: float = 8.0
    correlation_threshold: float = 0.35
    graph_lookback: int = 252

    @property
    def profile(self) -> RiskProfile:
        return RISK_PROFILES[self.risk_profile]


@dataclass(frozen=True)
class TextSignal:
    ticker: str
    sentiment_score: float
    confidence: float
    risk_flags: tuple[str, ...]
    evidence: tuple[str, ...]
    bias_warning: str
    investment_rationale: str


@dataclass
class AgentReasoningBundle:
    analyst_reports: dict[str, str]
    bull_case: str
    bear_case: str
    research_manager_view: str
    trader_plan: str
    risk_debate: dict[str, str]
    portfolio_manager_view: str
    text_signals: list[TextSignal]
    memory_log: pd.DataFrame
    checkpoint_note: str
    provider_note: str
    llm_selection: ResolvedLLMSelection

    def signal_frame(self) -> pd.DataFrame:
        rows = [
            {
                "ticker": signal.ticker,
                "sentiment_score": signal.sentiment_score,
                "confidence": signal.confidence,
                "risk_flags": ", ".join(signal.risk_flags),
                "evidence": " | ".join(signal.evidence),
                "bias_warning": signal.bias_warning,
                "investment_rationale": signal.investment_rationale,
            }
            for signal in self.text_signals
        ]
        return pd.DataFrame(rows)


@dataclass
class MarketGraph:
    edges: pd.DataFrame
    node_features: pd.DataFrame
    sector_map: dict[str, str]
    summary: dict[str, Any]


@dataclass
class MarketState:
    as_of_date: pd.Timestamp
    feature_frame: pd.DataFrame
    graph: MarketGraph
    regime: str
    state_vector: pd.Series


@dataclass
class EnvironmentResult:
    weights: pd.DataFrame
    transitions: pd.DataFrame
    reward_summary: dict[str, float]


@dataclass
class AllocationResult:
    component_weights: dict[str, pd.DataFrame]
    final_weights: pd.DataFrame
    performance: pd.DataFrame
    metrics: dict[str, float]
    decision_log: pd.DataFrame
    risk_events: pd.DataFrame
    suitability_checks: pd.DataFrame
    governance_warnings: list[str]


@dataclass
class ExplainabilityBundle:
    feature_importance: pd.DataFrame
    prediction_buckets: pd.DataFrame
    local_contributions: pd.DataFrame
    what_if: pd.DataFrame
    structured_trace: pd.DataFrame


@dataclass
class WorkflowResult:
    settings: WorkflowSettings
    quant_result: Any
    agent_reasoning: AgentReasoningBundle
    market_state: MarketState
    environment: EnvironmentResult
    allocation: AllocationResult
    explainability: ExplainabilityBundle
    audit_report_markdown: str
    reference_links: list[str] = field(default_factory=list)
