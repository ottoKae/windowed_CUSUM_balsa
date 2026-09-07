"""Build deterministic SHA-256 manifests for the complete public package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
MANIFEST = PROVENANCE / "RELEASE_MANIFEST.json"
CHECKSUMS = PROVENANCE / "CHECKSUMS.sha256"
EXCLUDED = {MANIFEST.resolve(), CHECKSUMS.resolve()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = release_files()
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
        "release_date": "2026-09-07",
        "scope": ["Git tracked plus untracked non-ignored files; no ignored B/C archives"],
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


def release_files() -> list[Path]:
    """Use Git's actual allow/ignore rules, including already tracked files."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    paths = {ROOT / x.decode("utf-8") for x in result.stdout.split(b"\0") if x}
    return sorted(p for p in paths if p.is_file() and p.resolve() not in EXCLUDED)


if __name__ == "__main__":
    main()
