"""Create TA-facing April check-in figures.

These figures are designed for presentation and rubric defense rather than
exploratory notebook work. They use existing report CSVs and do not perturb
model outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


COLORS = {
    "ink": "#17212b",
    "muted": "#657386",
    "paper": "#fffaf2",
    "blue": "#1f5f8b",
    "red": "#bc4749",
    "green": "#2f6f4e",
    "amber": "#c47f2c",
    "grid": "#d8cec0",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": "#fffaf2",
            "axes.edgecolor": "#d8cec0",
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def create_model_scorecard() -> Path:
    scores_path = REPORTS / "delay_prediction_v4_model_scores.csv"
    scores = pd.read_csv(scores_path)
    scores = scores[scores["status"].eq("ok")].copy()
    scores = scores.sort_values("rank").head(8)
    scores["short_label"] = scores["model_label"].str.replace(" / v2_core", "", regex=False)

    best = scores.iloc[0]

    fig = plt.figure(figsize=(14, 8.5), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.25, 1],
        left=0.20,
        right=0.98,
        top=0.84,
        bottom=0.13,
        hspace=0.42,
        wspace=0.25,
    )
    ax_score = fig.add_subplot(grid[0, :])
    ax_mae = fig.add_subplot(grid[1, 0])
    ax_early = fig.add_subplot(grid[1, 1])

    labels = scores["short_label"].tolist()[::-1]
    composite = scores["composite_score"].tolist()[::-1]
    bar_colors = [COLORS["green"] if label == best["short_label"] else "#aeb8a6" for label in labels]
    ax_score.barh(labels, composite, color=bar_colors)
    ax_score.set_title("Model selection: score-best V4 is deployable, not just low-MAE")
    ax_score.set_xlabel("Composite deployability score (higher is better)")
    ax_score.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax_score.set_xlim(0, max(100, max(composite) + 5))
    for y, value in enumerate(composite):
        ax_score.text(value + 1, y, f"{value:.1f}", va="center", fontsize=9)

    mae_labels = scores["short_label"].tolist()[::-1]
    mae_values = scores["primary_mae"].tolist()[::-1]
    ax_mae.barh(mae_labels, mae_values, color="#d59a50")
    ax_mae.set_title("True-delay test MAE")
    ax_mae.set_xlabel("MAE minutes (lower is better)")
    ax_mae.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    for y, value in enumerate(mae_values):
        ax_mae.text(value + 0.03, y, f"{value:.2f}", va="center", fontsize=8)

    ax_early.scatter(
        scores["final_2024_2025_to_2026_negative_prediction_rate"] * 100,
        scores["final_2024_2025_to_2026_early_f1"],
        s=scores["composite_score"] * 5,
        c=[COLORS["green"] if rank == 1 else COLORS["blue"] for rank in scores["rank"]],
        alpha=0.82,
        edgecolors="white",
        linewidths=1.0,
    )
    for _, row in scores.iterrows():
        ax_early.annotate(
            str(int(row["rank"])),
            (
                row["final_2024_2025_to_2026_negative_prediction_rate"] * 100,
                row["final_2024_2025_to_2026_early_f1"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax_early.set_title("Why we do not pick MAE alone")
    ax_early.set_xlabel("Negative predictions (%)")
    ax_early.set_ylabel("Early-arrival F1")
    ax_early.grid(color=COLORS["grid"], linewidth=0.8)

    fig.suptitle(
        "Boston Bus Equity April Check-In: V4 Model Evaluation",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.02,
        0.035,
        "Target is true delay = actual - scheduled. MBTA official predictions are a comparison baseline, not the training label.",
        fontsize=10,
        color=COLORS["muted"],
    )
    out = FIGURES / "april_checkin_model_scorecard.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def create_live_estimates() -> Path:
    live_path = REPORTS / "mbta_realtime_official_vs_model.csv"
    live = pd.read_csv(live_path)
    live["observed_at"] = (
        pd.to_datetime(live["observed_at"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    live["scheduled_time"] = (
        pd.to_datetime(live["scheduled_time"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    latest_at = live["observed_at"].max()
    latest = live[live["observed_at"].eq(latest_at)].sort_values("scheduled_time").head(12).copy()
    local = latest["official_delay_minutes"] + (
        latest["model_predicted_delay_minutes"] - latest["historical_baseline_delay_minutes"]
    )

    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.plot(
        latest["scheduled_time"],
        latest["official_delay_minutes"],
        marker="o",
        linewidth=2.7,
        color=COLORS["blue"],
        label="MBTA official live",
    )
    ax.plot(
        latest["scheduled_time"],
        local,
        marker="^",
        linewidth=2.7,
        color=COLORS["red"],
        label="Local realtime forecast",
    )
    ax.plot(
        latest["scheduled_time"],
        latest["historical_baseline_delay_minutes"],
        linewidth=2.2,
        linestyle="--",
        color=COLORS["amber"],
        label="Route-stop-hour baseline",
    )
    ax.axhline(0, color=COLORS["ink"], linewidth=1, alpha=0.55)
    ax.set_title("Realtime demo: official, local, and baseline delay estimates")
    ax.set_ylabel("Delay estimate (minutes)")
    ax.set_xlabel("Scheduled trip time")
    ax.grid(color=COLORS["grid"], linewidth=0.8)
    ax.legend(loc="upper left", ncols=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    route = latest["route_id"].iloc[0]
    stop = latest["stop_id"].iloc[0]
    mean_gap = (latest["official_delay_minutes"] - local).abs().mean()
    ax.text(
        0.01,
        -0.22,
        (
            f"Latest snapshot: route {route} / stop {stop}. "
            f"Mean official-local disagreement: {mean_gap:.2f} min. "
            "This is not true error until later actual arrivals are matched."
        ),
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["muted"],
    )
    out = FIGURES / "april_checkin_live_estimates_readable.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _card(ax, xy, title: str, points: str, lines: list[str], color: str) -> None:
    x, y = xy
    width, height = 0.46, 0.38
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#e0d2bf",
        facecolor="#fff8ed",
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + height - 0.075, title, fontsize=14, fontweight="bold", color=color)
    ax.text(x + width - 0.11, y + height - 0.075, points, fontsize=13, fontweight="bold", color=color)
    for i, line in enumerate(lines):
        ax.text(x + 0.03, y + height - 0.13 - i * 0.055, f"- {line}", fontsize=10, color=COLORS["ink"])


def create_rubric_evidence_map() -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.02,
        0.94,
        "April Check-In Rubric Evidence Map",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.02,
        0.895,
        "What we will show the TA, why it matters, and where it appears in the repository.",
        fontsize=11,
        color=COLORS["muted"],
    )

    _card(
        ax,
        (0.03, 0.50),
        "Data Visualizations",
        "15 pts",
        [
            "Delay distribution and route-level reliability",
            "Equity context via demographic/service figures",
            "Model scorecard and realtime comparison are labeled",
            "Charts support claims, not decoration",
        ],
        COLORS["blue"],
    )
    _card(
        ax,
        (0.52, 0.50),
        "Data Processing",
        "15 pts",
        [
            "MBTA historical arrival/departure data",
            "2024-2026 processed into parquet",
            "62.3M rows, 158 routes, 1,220 stops",
            "Time split prevents future leakage",
        ],
        COLORS["green"],
    )
    _card(
        ax,
        (0.03, 0.08),
        "Modeling Methods",
        "15 pts",
        [
            "Baselines: dummy, historical, V2 MLP",
            "V4 sweep: LightGBM, CatBoost, XGBoost, sklearn",
            "Current dashboard model: LightGBM-q35 / v2_core",
            "Online-safe features match realtime inputs",
        ],
        COLORS["amber"],
    )
    _card(
        ax,
        (0.52, 0.08),
        "Results & Interpretation",
        "5 pts",
        [
            "Score-best V4 final MAE: 3.94 min",
            "V2 sample MAE: 4.18 min",
            "Realtime chart is disagreement, not accuracy",
            "V5 will need matched live actual arrivals",
        ],
        COLORS["red"],
    )

    ax.text(
        0.03,
        0.015,
        "Main claim: we have a defensible data pipeline, final-quality visualizations, tested models, and honest interpretation of realtime limits.",
        fontsize=11,
        color=COLORS["muted"],
    )
    out = FIGURES / "april_checkin_rubric_evidence_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = [
        create_model_scorecard(),
        create_live_estimates(),
        create_rubric_evidence_map(),
    ]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
