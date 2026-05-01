"""
Matched-actuals streaming daemon.

Periodically queries the MBTA V3 API for upcoming-trip predictions on a set
of routes/stops, persists each snapshot, and later matches each prediction
against the trip's *actual* arrival when it lands. This produces ground-
truth labels for live model evaluation - the missing piece that lets us
say whether V3 GRU or V6 Transformer is *truly* more accurate, rather
than just whether one or the other agrees more closely with MBTA's own
predictions.

Output structure:
    reports/matched_actuals/
        YYYY-MM-DD.predictions.parquet  -- one row per (trip_id, stop_id, observed_at)
        YYYY-MM-DD.actuals.parquet      -- one row per (trip_id, stop_id) with actual_arrival
        matched_actuals.parquet          -- joined view; predictions + matched actuals
        meta.json                        -- daemon runtime stats

Usage:
    # Foreground (Ctrl+C to stop)
    python -m src.inference.matched_actuals_daemon

    # Background, 5 minute polling cadence on default popular routes
    nohup python -m src.inference.matched_actuals_daemon --poll-seconds 300 \\
        > matched_actuals.log 2>&1 &

After 2-4 weeks of accumulated data, run the matched-actuals evaluation
script (separate, future) to compute true MAE/RMSE/R^2 per model on
matched_actuals.parquet rather than on offline test split.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.log_mbta_live_snapshots import collect_live_snapshot
from src.inference.mbta_v3_client import MBTAV3Client


OUTPUT_DIR = PROJECT_ROOT / "reports" / "matched_actuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRACKED_ROUTES = [
    # Equity-priority routes (Livable Streets target list)
    ("22", "0"), ("29", "0"), ("15", "0"), ("28", "0"), ("44", "0"),
    ("17", "0"), ("23", "0"), ("31", "0"), ("111", "0"),
    # High-frequency routes (good signal density)
    ("1", "0"), ("39", "0"), ("57", "0"), ("66", "0"),
]


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _append_parquet(path: Path, df: pd.DataFrame) -> None:
    """Append DataFrame to a parquet file (creates if missing)."""
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


def collect_one_pass(client: MBTAV3Client, tracked: list[tuple[str, str]]) -> int:
    """Pull one snapshot of upcoming-trip predictions for each tracked route.

    Returns the number of prediction rows captured.
    """
    rows: list[dict] = []
    observed_at = datetime.now(timezone.utc)

    for route_id, direction_id in tracked:
        try:
            snapshot = collect_live_snapshot(
                client=client,
                route_id=route_id,
                stop_id=None,  # all stops on the route
                direction_id=direction_id,
                prediction_limit=50,
                vehicle_limit=30,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] route {route_id}: {exc}")
            continue

        merged = snapshot.get("merged", pd.DataFrame())
        if merged.empty:
            continue

        # Keep only fields useful for matched-actuals analysis. pandas NA
        # values can't be coerced via `or ""`, so guard with pd.isna.
        def _opt_str(v: object, fallback: str = "") -> str:
            return fallback if pd.isna(v) else str(v)

        for _, row in merged.iterrows():
            rows.append(
                {
                    "observed_at": observed_at.isoformat(),
                    "route_id": _opt_str(row.get("route_id"), route_id),
                    "direction_id": _opt_str(row.get("direction_id"), direction_id),
                    "trip_id": _opt_str(row.get("trip_id")),
                    "vehicle_id": _opt_str(row.get("vehicle_id")),
                    "stop_id": _opt_str(row.get("stop_id")),
                    "scheduled_time": _opt_str(row.get("scheduled_time")),
                    "predicted_time": _opt_str(row.get("predicted_time")),
                    "official_delay_minutes": row.get("official_delay_minutes"),
                    "current_stop_sequence": row.get("current_stop_sequence"),
                    "speed": row.get("speed"),
                    "status": _opt_str(row.get("status")),
                }
            )

    if not rows:
        return 0
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / f"{_today_str()}.predictions.parquet"
    _append_parquet(out_path, df)
    return len(rows)


def match_to_actuals(predictions_path: Path, actuals_parquet: Path | None = None) -> int:
    """Match captured predictions against arrival_departure actuals.

    For each prediction whose scheduled_time has already passed (>15 min ago),
    look up the actual arrival in the historical parquet (if available) and
    record the realized delay. Returns count of newly matched rows.

    The historical parquet is the offline arrival-departure source. Once
    enough days accumulate, MBTA's own historical updates will land trips
    that we predicted while they were still upcoming.
    """
    if not predictions_path.exists():
        return 0

    actuals_parquet = actuals_parquet or (
        PROJECT_ROOT / "data" / "processed" / "arrival_departure.parquet"
    )
    if not actuals_parquet.exists():
        print(f"  [warn] no historical actuals parquet at {actuals_parquet}; skipping matching")
        return 0

    preds = pd.read_parquet(predictions_path)
    preds["scheduled_time"] = pd.to_datetime(preds["scheduled_time"], errors="coerce", utc=True)
    now = datetime.now(timezone.utc)
    horizon_passed = preds["scheduled_time"] < pd.Timestamp(now) - pd.Timedelta(minutes=15)
    candidates = preds[horizon_passed].copy()
    if candidates.empty:
        return 0

    # Load only the trip-level columns we need from actuals; row-group filter
    # avoids loading the full 18 GB. We accept that newly-arrived trips may
    # not be in the snapshot until MBTA refreshes the historical archive.
    cols = ["trip_id", "stop_id", "scheduled", "actual"]
    actuals = pd.read_parquet(actuals_parquet, columns=cols)
    actuals["scheduled_t"] = pd.to_datetime(actuals["scheduled"], format="mixed", errors="coerce", utc=True)
    actuals["actual_t"] = pd.to_datetime(actuals["actual"], format="mixed", errors="coerce", utc=True)
    actuals["actual_delay_minutes"] = (actuals["actual_t"] - actuals["scheduled_t"]).dt.total_seconds() / 60.0

    # Match on (trip_id, stop_id) — the canonical key
    merged = candidates.merge(
        actuals[["trip_id", "stop_id", "actual_delay_minutes", "actual_t"]],
        on=["trip_id", "stop_id"],
        how="left",
    )
    matched = merged.dropna(subset=["actual_delay_minutes"])

    if matched.empty:
        return 0

    out = OUTPUT_DIR / "matched_actuals.parquet"
    if out.exists():
        existing = pd.read_parquet(out)
        # Avoid re-matching already-recorded rows
        merge_keys = ["observed_at", "trip_id", "stop_id"]
        new_rows = matched.merge(existing[merge_keys], on=merge_keys, how="left", indicator=True)
        new_rows = new_rows[new_rows["_merge"] == "left_only"].drop(columns=["_merge"])
    else:
        new_rows = matched

    if new_rows.empty:
        return 0
    _append_parquet(out, new_rows)
    return len(new_rows)


def run_daemon(poll_seconds: int, tracked: list[tuple[str, str]]) -> None:
    client = MBTAV3Client()
    stats = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "passes": 0,
        "predictions_captured": 0,
        "actuals_matched": 0,
    }

    stop = {"flag": False}

    def _handle_signal(signum, _frame):  # noqa: ARG001
        stop["flag"] = True
        print("\n[daemon] received signal, exiting cleanly...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[daemon] tracking {len(tracked)} (route, direction) pairs; poll={poll_seconds}s")
    print(f"[daemon] output dir: {OUTPUT_DIR}")

    while not stop["flag"]:
        t0 = time.time()
        captured = collect_one_pass(client, tracked)
        stats["passes"] += 1
        stats["predictions_captured"] += captured

        # Try to match older predictions against actuals once an hour
        if stats["passes"] % max(1, 3600 // max(poll_seconds, 60)) == 0:
            today_path = OUTPUT_DIR / f"{_today_str()}.predictions.parquet"
            new_matches = match_to_actuals(today_path)
            stats["actuals_matched"] += new_matches
            if new_matches:
                print(f"[daemon] matched {new_matches} new actuals")

        # Persist runtime stats so external tools can read progress
        (OUTPUT_DIR / "meta.json").write_text(json.dumps(stats, indent=2))

        elapsed = time.time() - t0
        print(
            f"[daemon] pass {stats['passes']:>4d}: "
            f"{captured:>4d} predictions in {elapsed:.1f}s  "
            f"(total preds={stats['predictions_captured']}, matched={stats['actuals_matched']})"
        )

        sleep_for = max(0.0, poll_seconds - elapsed)
        for _ in range(int(sleep_for)):
            if stop["flag"]:
                break
            time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poll-seconds", type=int, default=300,
                        help="Seconds between snapshots (default 300 = 5 min)")
    parser.add_argument("--routes", nargs="*",
                        help="Optional override list, e.g. --routes 1 22 28 (uses direction 0)")
    args = parser.parse_args()

    if args.routes:
        tracked = [(r, "0") for r in args.routes]
    else:
        tracked = DEFAULT_TRACKED_ROUTES

    run_daemon(poll_seconds=args.poll_seconds, tracked=tracked)


if __name__ == "__main__":
    main()
