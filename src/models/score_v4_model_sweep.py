"""Score V4 model sweep candidates with a deployability rubric."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR

DEFAULT_SWEEP_CSV = REPORTS_DIR / "delay_prediction_v4_model_sweep.csv"
DEFAULT_SCORE_CSV = REPORTS_DIR / "delay_prediction_v4_model_scores.csv"
DEFAULT_REPORT = REPORTS_DIR / "MODEL_SCORING_GUIDE.md"
DEFAULT_FIGURE = FIGURES_DIR / "v4_model_deployability_scores.png"

PROFILE_ONLINE_SCORE = {
    "v2_core": 100.0,
    "stats_time": 90.0,
    "no_ids": 78.0,
    "all": 62.0,
}

WEIGHTS = {
    "accuracy_score": 0.40,
    "stability_score": 0.15,
    "online_readiness_score": 0.15,
    "early_delay_score": 0.20,
    "cost_score": 0.10,
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _higher_is_better(values: pd.Series) -> pd.Series:
    numeric = _safe_numeric(values)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(50.0, index=values.index)
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        return pd.Series(100.0, index=values.index)
    return ((numeric - low) / (high - low) * 100.0).clip(0, 100).fillna(0.0)


def _lower_is_better(values: pd.Series) -> pd.Series:
    return 100.0 - _higher_is_better(values)


def _metric_column(frame: pd.DataFrame, metric: str) -> str:
    final = f"final_2024_2025_to_2026_{metric}"
    test = f"test_{metric}"
    if final in frame.columns and _safe_numeric(frame[final]).notna().any():
        return final
    return test


def score_model_sweep(
    sweep_csv: Path = DEFAULT_SWEEP_CSV,
    output_csv: Path = DEFAULT_SCORE_CSV,
    output_report: Path = DEFAULT_REPORT,
    output_figure: Path = DEFAULT_FIGURE,
) -> pd.DataFrame:
    records = pd.read_csv(sweep_csv)
    scored = records[records["status"].eq("ok")].copy()
    if scored.empty:
        raise ValueError(f"No successful models found in {sweep_csv}")

    mae_column = _metric_column(scored, "MAE")
    rmse_column = _metric_column(scored, "RMSE")
    r2_column = _metric_column(scored, "R2")

    scored["primary_mae"] = _safe_numeric(scored[mae_column])
    scored["primary_rmse"] = _safe_numeric(scored[rmse_column])
    scored["primary_r2"] = _safe_numeric(scored[r2_column])

    train_gap = (
        _safe_numeric(scored["test_MAE"]) - _safe_numeric(scored["train_MAE"])
    ).clip(lower=0)
    validation_gap = (
        _safe_numeric(scored["test_MAE"]) - _safe_numeric(scored["validation_MAE"])
    ).abs()
    scored["generalization_gap"] = train_gap + validation_gap

    scored["accuracy_score"] = _lower_is_better(scored["primary_mae"])
    scored["rmse_score"] = _lower_is_better(scored["primary_rmse"])
    scored["r2_score"] = _higher_is_better(scored["primary_r2"])
    scored["stability_score"] = _lower_is_better(scored["generalization_gap"])
    scored["online_readiness_score"] = (
        scored["feature_profile"].map(PROFILE_ONLINE_SCORE).fillna(55.0)
    )

    early_f1_column = _metric_column(scored, "early_f1")
    early_mae_column = _metric_column(scored, "early_MAE")
    negative_rate_column = _metric_column(scored, "negative_prediction_rate")
    early_share_column = _metric_column(scored, "early_share")
    if early_f1_column in scored.columns:
        early_f1 = _safe_numeric(scored[early_f1_column]).fillna(0.0).clip(0, 1) * 100.0
        if early_mae_column in scored.columns:
            early_mae_score = _lower_is_better(scored[early_mae_column])
        else:
            early_mae_score = pd.Series(50.0, index=scored.index)
        scored["early_delay_score"] = (0.7 * early_f1 + 0.3 * early_mae_score).clip(
            0,
            100,
        )
        early_note = (
            "Early-delay score uses measured negative-delay F1 with an early-event "
            "MAE tie-breaker."
        )
    elif "test_early_recall" in scored.columns:
        scored["early_delay_score"] = _safe_numeric(scored["test_early_recall"]).fillna(0.0).clip(0, 1) * 100.0
        early_note = "Early-delay score uses measured negative-delay recall."
    else:
        scored["early_delay_score"] = 50.0
        early_note = (
            "Early-delay recall was not stored in the current sweep CSV, so this "
            "component is neutral. Future sweeps should add test_early_recall, "
            "negative_prediction_rate, and early_delay_MAE."
        )

    fit_penalty = _higher_is_better(_safe_numeric(scored["fit_seconds"]))
    feature_penalty = _higher_is_better(_safe_numeric(scored["feature_count"]))
    scored["cost_score"] = (100.0 - 0.65 * fit_penalty - 0.35 * feature_penalty).clip(
        0,
        100,
    )

    scored["composite_score"] = sum(
        scored[column] * weight for column, weight in WEIGHTS.items()
    )
    if negative_rate_column in scored.columns and early_share_column in scored.columns:
        negative_rate = _safe_numeric(scored[negative_rate_column]).fillna(0.0)
        early_share = _safe_numeric(scored[early_share_column]).fillna(0.0)
        no_negative_gate = (early_share >= 0.05) & (negative_rate < 0.01)
        scored["early_viability_penalty"] = np.where(no_negative_gate, 25.0, 0.0)
        scored["composite_score"] = (
            scored["composite_score"] - scored["early_viability_penalty"]
        ).clip(lower=0.0)
        scored["early_viability_gate"] = np.where(no_negative_gate, "capped", "pass")
        scored["composite_score"] = np.where(
            no_negative_gate,
            np.minimum(scored["composite_score"], 55.0),
            scored["composite_score"],
        )
    else:
        scored["early_viability_penalty"] = 0.0
        scored["early_viability_gate"] = "not_measured"
    scored["rank"] = scored["composite_score"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    scored["model_label"] = (
        scored["model_kind"].astype(str).str.replace("_", " ", regex=False)
        + " / "
        + scored["feature_profile"].astype(str)
    )
    scored = scored.sort_values(["composite_score", "primary_mae"], ascending=[False, True])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    _plot_scores(scored, output_figure)
    _write_report(scored, output_report, output_csv, output_figure, early_note)
    return scored


def _plot_scores(scored: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = scored.head(14).copy().sort_values("composite_score")

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    fig.patch.set_facecolor("#f2eee6")
    ax.set_facecolor("#fffaf4")
    colors = ["#2f6f4e" if rank == 1 else "#1f5f8b" for rank in top["rank"]]
    bars = ax.barh(top["model_label"], top["composite_score"], color=colors)
    for bar, mae in zip(bars, top["primary_mae"], strict=False):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f} | MAE {mae:.2f}",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlim(0, max(100, float(top["composite_score"].max()) + 8))
    ax.set_xlabel("Composite score (higher is better)")
    ax.set_title("Model Deployability Score", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    fig.text(
        0.01,
        0.015,
        "Score = 40% MAE + 15% generalization + 15% online readiness + 20% early-delay behavior + 10% cost; no-negative models lose 25 points.",
        fontsize=9.5,
        color="#4a5568",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_report(
    scored: pd.DataFrame,
    output_path: Path,
    csv_path: Path,
    figure_path: Path,
    early_note: str,
) -> Path:
    top_columns = [
        "rank",
        "model_kind",
        "feature_profile",
        "composite_score",
        "primary_mae",
        "primary_rmse",
        "primary_r2",
        "accuracy_score",
        "stability_score",
        "online_readiness_score",
        "early_delay_score",
        "early_viability_penalty",
        "early_viability_gate",
        "cost_score",
    ]
    optional_columns = [
        "final_2024_2025_to_2026_early_f1",
        "test_early_f1",
        "final_2024_2025_to_2026_negative_prediction_rate",
        "test_negative_prediction_rate",
    ]
    table_columns = top_columns + [
        column for column in optional_columns if column in scored.columns
    ]
    lines = [
        "# Model Scoring Guide",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Purpose",
        "",
        "The model sweep should not be ranked only by MAE. For a realtime bus-delay dashboard, a model also needs to generalize across years, use features that are available online, detect early/negative delay when measured, and remain cheap enough to retrain.",
        "",
        "## Composite Score",
        "",
        "- Accuracy score, 40%: lower 2026 true-label MAE is better. If final 2024+2025 -> 2026 metrics exist, they are used; otherwise test metrics are used.",
        "- Stability score, 15%: lower train/test overfit and validation/test gap is better.",
        "- Online readiness score, 15%: online-safe feature profiles receive more credit than feature-heavy profiles that depend on fields often missing at inference time.",
        "- Early-delay score, 20%: rewards negative-delay F1 and early-event MAE. This matters because the user noticed the model under-predicts early arrivals.",
        "- Cost score, 10%: faster and simpler models receive a small bonus.",
        "- Early viability gate: subtracts 25 points and caps the final score at 55 when the test data has meaningful early arrivals but the model almost never predicts negative delay.",
        "",
        f"Early-delay note: {early_note}",
        "",
        "## Top Models",
        "",
        "```text",
        scored[table_columns].head(20).to_string(index=False),
        "```",
        "",
        f"CSV: `{csv_path}`",
        f"Figure: `{figure_path}`",
        "",
        "## Baseline Policy",
        "",
        "The sweep already includes a `dummy_median` model as a statistical baseline. The dashboard also exposes a route-stop-hour historical baseline for realtime comparison. Future matched-live evaluation should add MBTA official MAE and V5 residual-correction MAE once actual labels are available.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score V4 model sweep candidates")
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    scored = score_model_sweep(
        sweep_csv=args.sweep_csv,
        output_csv=args.output_csv,
        output_report=args.output_report,
        output_figure=args.output_figure,
    )
    best = scored.iloc[0]
    print(
        {
            "output_csv": str(args.output_csv),
            "output_report": str(args.output_report),
            "output_figure": str(args.output_figure),
            "best_model": str(best["model_kind"]),
            "best_feature_profile": str(best["feature_profile"]),
            "best_score": float(best["composite_score"]),
            "best_mae": float(best["primary_mae"]),
        }
    )


if __name__ == "__main__":
    main()
