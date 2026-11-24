"""Small I/O helpers shared by the dataset and method layers."""

from __future__ import annotations

from pathlib import Path


def read_whitespace_rows(path: str | Path) -> list[list[str]]:
    """Return non-empty whitespace-separated rows of a text file."""
    rows: list[list[str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens:
            rows.append(tokens)
    return rows


def write_lines(path: str | Path, lines: list[str] | tuple[str, ...]) -> Path:
    """Write one line per item and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target
