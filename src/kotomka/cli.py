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
    cleanup = sub.add_parser("cleanup-frames", help="Prune unused PNG candidates from completed jobs")
    cleanup.add_argument("--dry-run", action="store_true", help="Count candidates without deleting them")
    args = parser.parse_args()

    if args.command == "codex-login":
        path = run_codex_device_login()
        print(f"Saved Codex subscription auth state: {path}")
        return

    settings = get_settings()
    if args.command == "cleanup-frames":
        from .storage import JobStore

        store = JobStore(settings.db_path, settings.jobs_dir)
        files = size = jobs = skipped = 0
        for job_id in store.completed_job_ids():
            try:
                result = store.cleanup_frames(job_id, dry_run=args.dry_run)
            except (OSError, ValueError):
                skipped += 1
                print(f"Skipped {job_id}: report or frame references could not be validated")
                continue
            if result is not None:
                jobs += 1
                files += result.files
                size += result.bytes
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {files} PNGs ({size} bytes) across {jobs} completed jobs; skipped {skipped}")
        return
    uvicorn.run(
        "kotomka.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
