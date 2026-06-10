from __future__ import annotations

import argparse

import uvicorn

from .config import get_settings
from .providers.llm.codex_subscription import run_codex_device_login


def main() -> None:
    parser = argparse.ArgumentParser(prog="kotomka")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Run the local web service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    sub.add_parser("codex-login", help="Login to the Codex subscription route")
    args = parser.parse_args()

    if args.command == "codex-login":
        path = run_codex_device_login()
        print(f"Saved Codex subscription auth state: {path}")
        return

    settings = get_settings()
    uvicorn.run(
        "kotomka.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

