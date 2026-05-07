from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def standardized_importance(model, x_eval: pd.DataFrame, y_eval: pd.Series, feature_names: Iterable[str]) -> pd.DataFrame:
    """Combine standardized coefficients with permutation importance."""
    feature_names = list(feature_names)
    ridge = model.named_steps["ridge"]
    coef_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": ridge.coef_,
            "abs_coefficient": np.abs(ridge.coef_),
        }
    )

    perm = permutation_importance(
        model,
        x_eval[feature_names],
        y_eval,
        n_repeats=12,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    coef_frame["permutation_importance"] = perm.importances_mean
    return coef_frame.sort_values(["permutation_importance", "abs_coefficient"], ascending=False).reset_index(drop=True)


def local_contributions(model, row: pd.Series, feature_names: Iterable[str]) -> pd.DataFrame:
    """Explain a single prediction via standardized linear contributions."""
    feature_names = list(feature_names)
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]

    values = pd.DataFrame([row[feature_names].astype(float).to_dict()])
    scaled = scaler.transform(values)[0]
    contrib = scaled * ridge.coef_
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": values.iloc[0].values,
            "scaled_value": scaled,
            "contribution": contrib,
        }
    )
    frame["abs_contribution"] = frame["contribution"].abs()
    frame = frame.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    return frame


def prediction_bucket_table(predictions: pd.DataFrame, bucket_count: int = 5) -> pd.DataFrame:
    """Summarize realized returns by prediction quantile bucket."""
    frame = predictions[["prediction", "target"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "avg_prediction", "avg_realized_return", "hit_rate", "observations"])

    percentiles = frame["prediction"].rank(method="first", pct=True)
    frame["bucket"] = np.ceil(percentiles * bucket_count).clip(1, bucket_count).astype(int)
    summary = (
        frame.groupby("bucket", sort=True)
        .agg(
            avg_prediction=("prediction", "mean"),
            avg_realized_return=("target", "mean"),
            hit_rate=("target", lambda x: float((x > 0.0).mean())),
            observations=("target", "size"),
        )
        .reset_index()
    )
    return summary


def feature_percentile_table(
    reference_row: pd.Series,
    feature_frame: pd.DataFrame,
    feature_names: Iterable[str],
    focus_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Show where the current feature values sit relative to history."""
    feature_names = list(feature_names)
    focus = list(dict.fromkeys(focus_features or feature_names))
    records = []
    for feature in focus:
        if feature not in feature_names:
            continue
        history = feature_frame[feature].dropna()
        if history.empty:
            continue

        current_value = float(reference_row[feature])
        percentile = float((history <= current_value).mean())
        median = float(history.median())
        std = float(history.std())
        z_score = 0.0 if not np.isfinite(std) or abs(std) < 1e-12 else (current_value - median) / std
        records.append(
            {
                "feature": feature,
                "feature_value": current_value,
                "history_percentile": percentile,
                "z_score_vs_median": z_score,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["feature", "feature_value", "history_percentile", "z_score_vs_median"]
        )
    return pd.DataFrame(records).sort_values("history_percentile", ascending=False).reset_index(drop=True)


def build_signal_narrative(
    contributions: pd.DataFrame,
    predicted_return: float,
    realized_return: float,
    final_weight: float,
    top_n: int = 3,
) -> str:
    """Create a concise human-readable explanation for a single signal."""
    positive = contributions[contributions["contribution"] > 0.0].head(top_n)["feature"].tolist()
    negative = contributions[contributions["contribution"] < 0.0].head(top_n)["feature"].tolist()

    positive_text = ", ".join(positive) if positive else "no strong positive drivers"
    negative_text = ", ".join(negative) if negative else "no material negative drivers"

    direction = "bullish" if predicted_return >= 0.0 else "defensive"
    realized_text = "outperformed the signal" if realized_return >= predicted_return else "lagged the signal"
    return (
        f"The model issued a {direction} signal with a target weight of {final_weight:.1%}. "
        f"Main support came from {positive_text}, while {negative_text} held the score back. "
        f"The realized next-day move {realized_text}."
    )


def what_if_curve(
    model,
    reference_row: pd.Series,
    feature_name: str,
    feature_frame: pd.DataFrame,
    feature_names: Iterable[str],
    steps: int = 25,
) -> pd.DataFrame:
    """One-dimensional partial dependence style curve for a selected row."""
    feature_names = list(feature_names)
    lower = float(feature_frame[feature_name].quantile(0.05))
    upper = float(feature_frame[feature_name].quantile(0.95))
    grid = np.linspace(lower, upper, steps)
    records = []
    for value in grid:
        candidate = reference_row.copy()
        candidate[feature_name] = value
        pred = float(model.predict(pd.DataFrame([candidate[feature_names].astype(float).to_dict()]))[0])
        records.append({"feature_value": value, "predicted_return": pred})
    return pd.DataFrame(records)
