"""Churn prediction as a first-class service.

This module is the single entry point for scoring customers. Callers pass
raw records (a mapping, a list of mappings, or a DataFrame) and receive a
row-aligned DataFrame with the prediction, human-readable label, and churn
probability. Model loading, cleaning, and feature preparation stay here so
notebooks, APIs, and batch jobs share exactly one inference path.

Example:
    from src.services.churn import predict

    predict({"gender": "Female", "tenure": 1, ...})
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import pandas as pd

from main import prepare_inference_data

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

PREDICTION_COLUMN = "churn_prediction"
LABEL_COLUMN = "churn_label"
PROBABILITY_COLUMN = "churn_probability"

Records = Mapping[str, Any] | pd.DataFrame | Iterable[Mapping[str, Any]]


@lru_cache(maxsize=4)
def load_model(model_path: str = str(DEFAULT_MODEL_PATH)):
    """Load and cache the fitted churn pipeline for this process."""
    path = Path(model_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict(
    records: Records,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score raw customer records for churn.

    Args:
        records: A single record, an iterable of records, or a DataFrame.
            Target and identifier columns are ignored when present.
        model_path: Path to the fitted pipeline. Defaults to the project
            model and is resolved relative to the project root.
        threshold: Probability cutoff used for the binary prediction.

    Returns:
        A DataFrame with one row per input record and the columns
        ``churn_prediction`` (1/0), ``churn_label`` ("Yes"/"No"), and
        ``churn_probability`` (0-1), preserving input order.

    Raises:
        ValueError: If ``threshold`` is outside ``[0, 1]``.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    frame = _as_frame(records)
    if frame.empty:
        return pd.DataFrame(columns=[PREDICTION_COLUMN, LABEL_COLUMN, PROBABILITY_COLUMN])

    features = prepare_inference_data(frame)
    model = load_model(str(Path(model_path)))

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            PREDICTION_COLUMN: predictions,
            LABEL_COLUMN: pd.Series(predictions).map({1: "Yes", 0: "No"}),
            PROBABILITY_COLUMN: probabilities,
        }
    )


def _as_frame(records: Records) -> pd.DataFrame:
    """Normalise supported inputs into a DataFrame with a clean index."""
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Mapping):
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(records)
    return frame.reset_index(drop=True)
