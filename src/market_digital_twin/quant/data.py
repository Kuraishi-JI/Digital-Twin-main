from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE", "XLV"]


def generate_demo_prices(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    seed: int = 14,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a reproducible multi-asset market with regime shifts for offline research."""
    tickers = list(tickers or DEFAULT_UNIVERSE)
    dates = pd.bdate_range(start_date, end_date)
    if len(dates) < 260:
        raise ValueError("Sample market generation needs at least 260 business days.")

    rng = np.random.default_rng(seed)
    n = len(dates)

    regime = np.zeros(n, dtype=int)
    for idx in range(1, n):
        stay_prob = 0.95 if regime[idx - 1] == 0 else 0.82
        if rng.random() > stay_prob:
            regime[idx] = 1 - regime[idx - 1]
        else:
            regime[idx] = regime[idx - 1]

    shock_mask = rng.random(n) < 0.025
    market = np.where(regime == 0, rng.normal(0.0005, 0.008, n), rng.normal(-0.0007, 0.017, n))
    rates = np.where(regime == 0, rng.normal(0.0001, 0.004, n), rng.normal(-0.0002, 0.006, n))
    inflation = rng.normal(0.00015, 0.007, n)
    defensive = np.where(regime == 0, rng.normal(-0.00005, 0.003, n), rng.normal(0.00045, 0.008, n))
    momentum = np.zeros(n)
    for idx in range(1, n):
        momentum[idx] = 0.35 * momentum[idx - 1] + 0.65 * market[idx - 1]

    market[shock_mask] += rng.normal(-0.055, 0.018, shock_mask.sum())
    defensive[shock_mask] += rng.normal(0.014, 0.006, shock_mask.sum())
    inflation[shock_mask] += rng.normal(0.01, 0.01, shock_mask.sum())

    factor_frame = pd.DataFrame(
        {
            "market": market,
            "rates": rates,
            "inflation": inflation,
            "defensive": defensive,
            "momentum": momentum,
            "regime": regime,
        },
        index=dates,
    )

    betas = {
        "SPY": {"market": 1.00, "rates": -0.15, "inflation": 0.05, "defensive": -0.10, "momentum": 0.20, "drift": 0.00025, "idio": 0.0045},
        "QQQ": {"market": 1.25, "rates": -0.45, "inflation": -0.08, "defensive": -0.25, "momentum": 0.45, "drift": 0.00035, "idio": 0.0065},
        "IWM": {"market": 1.10, "rates": -0.20, "inflation": 0.20, "defensive": -0.22, "momentum": 0.25, "drift": 0.00020, "idio": 0.0075},
        "TLT": {"market": -0.35, "rates": -1.00, "inflation": -0.28, "defensive": 0.55, "momentum": 0.10, "drift": 0.00010, "idio": 0.0050},
        "GLD": {"market": 0.05, "rates": -0.22, "inflation": 0.82, "defensive": 0.35, "momentum": 0.18, "drift": 0.00018, "idio": 0.0058},
        "XLE": {"market": 0.72, "rates": 0.08, "inflation": 1.15, "defensive": -0.32, "momentum": 0.22, "drift": 0.00028, "idio": 0.0085},
        "XLV": {"market": 0.62, "rates": -0.05, "inflation": 0.10, "defensive": 0.42, "momentum": 0.12, "drift": 0.00020, "idio": 0.0048},
    }

    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        spec = betas.get(
            ticker,
            {"market": 0.80, "rates": -0.10, "inflation": 0.10, "defensive": 0.00, "momentum": 0.20, "drift": 0.00020, "idio": 0.0060},
        )
        idio = rng.normal(0.0, spec["idio"], n)
        asset_returns = (
            spec["drift"]
            + factor_frame["market"] * spec["market"]
            + factor_frame["rates"] * spec["rates"]
            + factor_frame["inflation"] * spec["inflation"]
            + factor_frame["defensive"] * spec["defensive"]
            + factor_frame["momentum"] * spec["momentum"]
            + idio
        )
        asset_returns = asset_returns.clip(-0.22, 0.22)
        frames[ticker] = (100.0 * (1.0 + asset_returns).cumprod()).rename(ticker)

    prices = pd.concat(frames.values(), axis=1)
    prices.index.name = "date"
    return prices


def load_prices_from_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load a wide price table from CSV bytes."""
    frame = pd.read_csv(BytesIO(file_bytes))
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")

    first_col = frame.columns[0]
    if first_col.lower() in {"date", "datetime", "timestamp"}:
        date_col = first_col
    else:
        date_col = first_col

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").ffill().dropna(how="all")
    if numeric.shape[1] < 2:
        raise ValueError("The CSV needs at least two asset columns.")
    numeric.index.name = "date"
    return numeric


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage returns."""
    return prices.sort_index().pct_change().dropna(how="all")
