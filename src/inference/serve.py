"""CLI entrypoint for the local realtime FastAPI service."""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import create_app
from .dashboard import choose_default_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the realtime delay inference API")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to the realtime inference bundle. Defaults to score-best V4, then MAE-best V4, then V2.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
        raise SystemExit(
            "uvicorn is required to run the realtime API. Install `uvicorn` first."
        ) from exc

    app = create_app(args.bundle or choose_default_bundle())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
