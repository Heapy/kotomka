from __future__ import annotations

from pathlib import Path

from .models import Report
from .utils import read_json, write_json


def save_report(report: Report, path: Path) -> None:
    write_json(path, report.model_dump())


def load_report(path: Path) -> Report:
    return Report.model_validate(read_json(path))

