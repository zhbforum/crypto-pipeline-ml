from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


class CsvSink:
    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        is_new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            if is_new:
                w.writeheader()
            w.writerows(rows)
        return len(rows)
