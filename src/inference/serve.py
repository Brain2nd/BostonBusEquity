"""
CLI entrypoint for the realtime FastAPI service.
"""

from __future__ import annotations

import argparse

import uvicorn

from src.inference.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Boston Bus Equity realtime API")
    parser.add_argument("--bundle", required=True, help="Path to the realtime bundle .pt file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.bundle)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
