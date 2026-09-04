"""Build deterministic SHA-256 manifests for the complete public package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
MANIFEST = PROVENANCE / "RELEASE_MANIFEST.json"
CHECKSUMS = PROVENANCE / "CHECKSUMS.sha256"
EXCLUDED = {MANIFEST.resolve(), CHECKSUMS.resolve()}
EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".pytest_tmp",
    ".pytest_tmp_root",
    ".qa",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "build",
    "dist",
}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".abs",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".synctex.gz",
    ".toc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    """Return whether *path* belongs to the distributable release."""

    if not path.is_file() or path.resolve() in EXCLUDED:
        return False
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
        return False
    name = path.name.lower()
    return not any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if included(path))
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": "windowed-cusum-balsa.release-manifest.v2",
        "release_date": "2026-09-04",
        "scope": ["complete repository release, excluding generated build and QA files"],
        "excluded_self_referential_files": [
            "provenance/RELEASE_MANIFEST.json",
            "provenance/CHECKSUMS.sha256"
        ],
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    CHECKSUMS.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records ({manifest['total_bytes']} bytes).")


if __name__ == "__main__":
    main()
