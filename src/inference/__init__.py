"""Realtime inference helpers for Boston Bus Equity."""

from .mbta_v3_client import MBTAV3Client, normalize_prediction_payload
from .runtime import DelayPredictorRuntime, PredictionInputError


def create_app(*args, **kwargs):
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "DelayPredictorRuntime",
    "MBTAV3Client",
    "PredictionInputError",
    "create_app",
    "normalize_prediction_payload",
]
