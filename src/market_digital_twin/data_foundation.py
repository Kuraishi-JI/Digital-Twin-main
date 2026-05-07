from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .quant.data import (
    DEFAULT_UNIVERSE,
    compute_returns,
    generate_demo_prices,
    load_prices_from_csv,
)


DEFAULT_SECTOR_MAP = {
    "SPY": "Broad Equity",
    "QQQ": "Technology Growth",
    "IWM": "Small Cap Equity",
    "TLT": "Rates Duration",
    "GLD": "Real Assets",
    "XLE": "Energy",
    "XLV": "Defensive Equity",
}


@dataclass(frozen=True)
class PriceSchemaReport:
    valid: bool
    warnings: tuple[str, ...]
    observations: int
    assets: int
    start: pd.Timestamp
    end: pd.Timestamp


def load_demo_prices(
    start_date: str,
    end_date: str,
    tickers: Iterable[str] | None = None,
    seed: int = 14,
) -> pd.DataFrame:
    universe = list(tickers or DEFAULT_UNIVERSE)
    return generate_demo_prices(start_date=start_date, end_date=end_date, seed=seed, tickers=universe)


def load_uploaded_prices(file_bytes: bytes) -> pd.DataFrame:
    return load_prices_from_csv(file_bytes)


def validate_prices(prices: pd.DataFrame, min_assets: int = 2, min_observations: int = 260) -> PriceSchemaReport:
    warnings: list[str] = []
    if prices.empty:
        raise ValueError("Price table is empty.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price table must use a DatetimeIndex.")

    numeric = prices.apply(pd.to_numeric, errors="coerce")
    valid_assets = numeric.dropna(axis=1, how="all")
    if valid_assets.shape[1] < min_assets:
        raise ValueError(f"At least {min_assets} assets are required.")
    if len(valid_assets) < min_observations:
        warnings.append(
            f"Only {len(valid_assets)} observations are available; walk-forward estimates may be unstable."
        )
    missing_ratio = float(valid_assets.isna().mean().mean())
    if missing_ratio > 0.05:
        warnings.append(f"Average missing-data ratio is {missing_ratio:.1%}; forward filling is applied.")

    non_positive = int((valid_assets <= 0.0).sum().sum())
    if non_positive:
        warnings.append(f"{non_positive} non-positive price cells were detected and ignored by percentage returns.")

    return PriceSchemaReport(
        valid=True,
        warnings=tuple(warnings),
        observations=int(len(valid_assets)),
        assets=int(valid_assets.shape[1]),
        start=pd.Timestamp(valid_assets.index.min()),
        end=pd.Timestamp(valid_assets.index.max()),
    )


def clean_price_frame(prices: pd.DataFrame, universe: Iterable[str] | None = None) -> pd.DataFrame:
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    if universe:
        requested = [ticker for ticker in universe if ticker in frame.columns]
        if len(requested) >= 2:
            frame = frame[requested]
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.ffill().dropna(axis=1, how="all").dropna(how="all")
    frame.index.name = "date"
    return frame


def benchmark_or_first(prices: pd.DataFrame, benchmark: str) -> str:
    return benchmark if benchmark in prices.columns else str(prices.columns[0])


def sector_map_for_universe(tickers: Iterable[str]) -> dict[str, str]:
    return {ticker: DEFAULT_SECTOR_MAP.get(ticker, "Custom Asset") for ticker in tickers}


def asset_profile(prices: pd.DataFrame) -> pd.DataFrame:
    returns = compute_returns(prices)
    latest = prices.iloc[-1]
    trailing_vol = returns.tail(20).std() * np.sqrt(252)
    drawdown = prices.div(prices.rolling(60).max()) - 1.0
    return pd.DataFrame(
        {
            "ticker": prices.columns,
            "latest_price": latest.reindex(prices.columns).values,
            "20d_volatility": trailing_vol.reindex(prices.columns).fillna(0.0).values,
            "60d_drawdown": drawdown.iloc[-1].reindex(prices.columns).fillna(0.0).values,
            "missing_ratio": prices.isna().mean().reindex(prices.columns).fillna(0.0).values,
        }
    )
