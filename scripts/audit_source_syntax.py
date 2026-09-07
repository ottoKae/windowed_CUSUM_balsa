#!/usr/bin/env python3
"""Parse release Python sources without importing them or creating bytecode."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    files = [ROOT / name for name in result.stdout.splitlines() if name]
    failures: list[tuple[str, str]] = []
    for path in files:
        if not path.is_file():
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append((path.relative_to(ROOT).as_posix(), str(exc)))
    print(f"Python files checked: {len(files)}")
    for name, error in failures:
        print(f"FAIL {name}: {error}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
