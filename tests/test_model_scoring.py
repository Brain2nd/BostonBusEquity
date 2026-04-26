from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.score_v4_model_sweep import score_model_sweep


def test_score_model_sweep_outputs_ranked_scores(tmp_path: Path) -> None:
    sweep = pd.DataFrame(
        [
            {
                "model_kind_requested": "lightgbm",
                "feature_profile": "v2_core",
                "feature_count": 18,
                "status": "ok",
                "error": "",
                "train_MAE": 3.7,
                "validation_MAE": 4.1,
                "test_MAE": 4.2,
                "test_RMSE": 6.0,
                "test_R2": 0.02,
                "model_kind": "lightgbm",
                "fit_seconds": 5.0,
                "final_2024_2025_to_2026_MAE": 3.8,
                "final_2024_2025_to_2026_RMSE": 5.9,
                "final_2024_2025_to_2026_R2": 0.04,
                "final_2024_2025_to_2026_early_f1": 0.35,
                "final_2024_2025_to_2026_early_MAE": 2.5,
                "final_2024_2025_to_2026_negative_prediction_rate": 0.12,
            },
            {
                "model_kind_requested": "dummy",
                "feature_profile": "v2_core",
                "feature_count": 18,
                "status": "ok",
                "error": "",
                "train_MAE": 4.2,
                "validation_MAE": 4.2,
                "test_MAE": 4.2,
                "test_RMSE": 6.2,
                "test_R2": -0.01,
                "model_kind": "dummy_median",
                "fit_seconds": 0.1,
                "final_2024_2025_to_2026_MAE": 4.1,
                "final_2024_2025_to_2026_RMSE": 6.2,
                "final_2024_2025_to_2026_R2": -0.01,
                "final_2024_2025_to_2026_early_f1": 0.0,
                "final_2024_2025_to_2026_early_MAE": 3.5,
                "final_2024_2025_to_2026_negative_prediction_rate": 0.0,
            },
        ]
    )
    sweep_path = tmp_path / "sweep.csv"
    score_path = tmp_path / "scores.csv"
    report_path = tmp_path / "guide.md"
    figure_path = tmp_path / "scores.png"
    sweep.to_csv(sweep_path, index=False)

    scored = score_model_sweep(
        sweep_csv=sweep_path,
        output_csv=score_path,
        output_report=report_path,
        output_figure=figure_path,
    )

    assert score_path.exists()
    assert report_path.exists()
    assert figure_path.exists()
    assert scored.iloc[0]["composite_score"] >= scored.iloc[-1]["composite_score"]
    assert {"accuracy_score", "stability_score", "online_readiness_score"} <= set(
        scored.columns
    )
