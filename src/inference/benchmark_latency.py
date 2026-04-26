"""Benchmark realtime inference latency and generate a report figure."""

from __future__ import annotations

import asyncio
import argparse
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.inference.build_bundle import build_realtime_bundle_from_dataframe
from src.inference.runtime import DelayPredictorRuntime
from src.models.v2_delay_predictor import V2_CHECKPOINT_NAME

DEFAULT_FIGURE_PATH = FIGURES_DIR / "realtime_inference_latency_baseline.png"
DEFAULT_REPORT_PATH = REPORTS_DIR / "REALTIME_INFERENCE_BASELINE.md"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / V2_CHECKPOINT_NAME


def _make_synthetic_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in [2024, 2025]:
        for route_id, route_bias in [("1", 0.5), ("2", 2.0)]:
            for stop_id, stop_bias in [("A", 0.2), ("B", 0.7)]:
                for day in [3, 4]:
                    for hour in [8, 17]:
                        scheduled = pd.Timestamp(
                            year=year,
                            month=1,
                            day=day,
                            hour=hour,
                            minute=0,
                            tz="UTC",
                        )
                        delay_minutes = route_bias + stop_bias + (0.5 if hour == 17 else 0.0)
                        actual = scheduled + pd.Timedelta(minutes=delay_minutes)
                        rows.append(
                            {
                                "service_date": scheduled.date().isoformat(),
                                "route_id": route_id,
                                "stop_id": stop_id,
                                "direction_id": "0",
                                "scheduled": scheduled.isoformat(),
                                "actual": actual.isoformat(),
                                "scheduled_headway": 12,
                                "year": year,
                            }
                        )

    return pd.DataFrame(rows)


def _measure_latency_ms(
    callable_obj: Callable[[], Any],
    iterations: int,
    warmup: int,
) -> list[float]:
    for _ in range(warmup):
        callable_obj()

    samples_ms: list[float] = []
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        callable_obj()
        samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

    return samples_ms


def _summarize_samples(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    array = np.asarray(ordered, dtype=float)
    return {
        "count": float(len(array)),
        "min_ms": float(array.min()),
        "avg_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


def _plot_latency(
    runtime_samples_ms: list[float],
    runtime_summary: dict[str, float],
    output_path: Path,
    api_samples_ms: list[float] | None = None,
    api_summary: dict[str, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#f3f1ea")

    histogram_ax, cdf_ax = axes
    runtime_array = np.asarray(runtime_samples_ms, dtype=float)
    bins = min(30, max(10, len(runtime_array) // 10))

    histogram_ax.hist(
        runtime_array,
        bins=bins,
        color="#1f5f8b",
        alpha=0.85,
        edgecolor="white",
        label="runtime.predict",
    )
    histogram_ax.axvline(
        runtime_summary["avg_ms"],
        color="#d1495b",
        linewidth=2,
        linestyle="--",
        label=f"avg {runtime_summary['avg_ms']:.3f} ms",
    )
    histogram_ax.axvline(
        runtime_summary["p95_ms"],
        color="#edae49",
        linewidth=2,
        linestyle=":",
        label=f"p95 {runtime_summary['p95_ms']:.3f} ms",
    )

    if api_samples_ms is not None and api_summary is not None:
        api_array = np.asarray(api_samples_ms, dtype=float)
        histogram_ax.hist(
            api_array,
            bins=bins,
            color="#6a994e",
            alpha=0.55,
            edgecolor="white",
            label="api /predict",
        )

    histogram_ax.set_title("Latency Distribution")
    histogram_ax.set_xlabel("Latency (ms)")
    histogram_ax.set_ylabel("Calls")
    histogram_ax.legend(frameon=False)

    runtime_sorted = np.sort(runtime_array)
    runtime_cdf = np.arange(1, len(runtime_sorted) + 1) / len(runtime_sorted)
    cdf_ax.plot(runtime_sorted, runtime_cdf, color="#1f5f8b", linewidth=2.2, label="runtime.predict")

    if api_samples_ms is not None and api_summary is not None:
        api_sorted = np.sort(np.asarray(api_samples_ms, dtype=float))
        api_cdf = np.arange(1, len(api_sorted) + 1) / len(api_sorted)
        cdf_ax.plot(api_sorted, api_cdf, color="#6a994e", linewidth=2.2, label="api /predict")

    cdf_ax.set_title("Empirical CDF")
    cdf_ax.set_xlabel("Latency (ms)")
    cdf_ax.set_ylabel("CDF")
    cdf_ax.set_ylim(0, 1.02)
    cdf_ax.legend(frameon=False)

    fig.suptitle("Realtime Inference Latency Baseline", fontsize=15, fontweight="bold")
    fig.text(
        0.02,
        0.02,
        (
            f"runtime avg={runtime_summary['avg_ms']:.3f} ms, "
            f"p95={runtime_summary['p95_ms']:.3f} ms"
            + (
                ""
                if api_summary is None
                else f" | api avg={api_summary['avg_ms']:.3f} ms, p95={api_summary['p95_ms']:.3f} ms"
            )
        ),
        fontsize=10,
        color="#3d405b",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.94))
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _benchmark_runtime(
    bundle_path: Path,
    iterations: int,
    warmup: int,
) -> tuple[list[float], dict[str, float]]:
    runtime = DelayPredictorRuntime.from_bundle_path(bundle_path)

    def _call_predict() -> dict[str, Any]:
        return runtime.predict(
            route_id="1",
            stop_id="A",
            scheduled_time=datetime(2026, 1, 5, 8, 0, 0),
        )

    samples_ms = _measure_latency_ms(_call_predict, iterations=iterations, warmup=warmup)
    return samples_ms, _summarize_samples(samples_ms)


def _benchmark_api(
    bundle_path: Path,
    iterations: int,
    warmup: int,
) -> tuple[list[float] | None, dict[str, float] | None, str | None]:
    try:
        import httpx

        from src.inference.api import create_app
    except Exception as exc:
        return None, None, str(exc)

    app = create_app(bundle_path)
    payload = {"route_id": "1", "stop_id": "A", "scheduled_time": "2026-01-05T08:00:00"}

    async def _run_benchmark() -> list[float]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            for _ in range(warmup):
                response = await client.post("/predict", json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"/predict returned {response.status_code}: {response.text}"
                    )

            samples_ms: list[float] = []
            for _ in range(iterations):
                start_ns = time.perf_counter_ns()
                response = await client.post("/predict", json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"/predict returned {response.status_code}: {response.text}"
                    )
                samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

        return samples_ms

    samples_ms = asyncio.run(_run_benchmark())
    return samples_ms, _summarize_samples(samples_ms), None


def _write_report(
    output_path: Path,
    figure_path: Path,
    checkpoint_path: Path,
    iterations: int,
    warmup: int,
    runtime_summary: dict[str, float],
    api_summary: dict[str, float] | None,
    api_skip_reason: str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _summary_lines(name: str, summary: dict[str, float]) -> str:
        return (
            f"| {name} | {int(summary['count'])} | {summary['min_ms']:.3f} | "
            f"{summary['avg_ms']:.3f} | {summary['p50_ms']:.3f} | "
            f"{summary['p95_ms']:.3f} | {summary['max_ms']:.3f} |"
        )

    lines = [
        "# Realtime Inference Baseline Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Iterations: `{iterations}`",
        f"- Warmup: `{warmup}`",
        f"- Platform: `{platform.platform()}`",
        f"- Figure: `{figure_path}`",
        "",
        "## Latency Summary",
        "",
        "| Target | Calls | Min (ms) | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) |",
        "|--------|------:|---------:|---------:|---------:|---------:|---------:|",
        _summary_lines("runtime.predict", runtime_summary),
    ]

    if api_summary is not None:
        lines.append(_summary_lines("api /predict", api_summary))
    else:
        lines.extend(
            [
                "",
                "## API Benchmark",
                "",
                f"Skipped: `{api_skip_reason}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- Runtime path verified with avg latency `{runtime_summary['avg_ms']:.3f} ms`.",
            f"- Runtime path verified with p95 latency `{runtime_summary['p95_ms']:.3f} ms`.",
        ]
    )

    if api_summary is not None:
        lines.extend(
            [
                f"- API path verified with avg latency `{api_summary['avg_ms']:.3f} ms`.",
                f"- API path verified with p95 latency `{api_summary['p95_ms']:.3f} ms`.",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_latency(
    checkpoint_path: Path,
    figure_path: Path,
    report_path: Path,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    temp_dir_path = PROJECT_ROOT / "tmp_benchmark_latency"
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path, ignore_errors=True)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    bundle_path = temp_dir_path / "realtime_bundle.pt"

    try:
        build_realtime_bundle_from_dataframe(
            dataframe=_make_synthetic_dataframe(),
            checkpoint_path=checkpoint_path,
            output_path=bundle_path,
        )

        runtime_samples_ms, runtime_summary = _benchmark_runtime(
            bundle_path=bundle_path,
            iterations=iterations,
            warmup=warmup,
        )
        api_samples_ms, api_summary, api_skip_reason = _benchmark_api(
            bundle_path=bundle_path,
            iterations=max(20, iterations // 2),
            warmup=max(5, warmup // 2),
        )
    finally:
        shutil.rmtree(temp_dir_path, ignore_errors=True)

    _plot_latency(
        runtime_samples_ms=runtime_samples_ms,
        runtime_summary=runtime_summary,
        output_path=figure_path,
        api_samples_ms=api_samples_ms,
        api_summary=api_summary,
    )
    _write_report(
        output_path=report_path,
        figure_path=figure_path,
        checkpoint_path=checkpoint_path,
        iterations=iterations,
        warmup=warmup,
        runtime_summary=runtime_summary,
        api_summary=api_summary,
        api_skip_reason=api_skip_reason,
    )

    return {
        "figure_path": str(figure_path),
        "report_path": str(report_path),
        "runtime_summary": runtime_summary,
        "api_summary": api_summary,
        "api_skip_reason": api_skip_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark realtime inference latency")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the V2 MLP checkpoint",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Path to save the latency figure",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to save the benchmark markdown report",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of measured iterations per benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup calls before measuring",
    )
    args = parser.parse_args()

    metrics = benchmark_latency(
        checkpoint_path=args.checkpoint.resolve(),
        figure_path=args.output_figure.resolve(),
        report_path=args.output_report.resolve(),
        iterations=args.iterations,
        warmup=args.warmup,
    )
    print(metrics)


if __name__ == "__main__":
    main()
