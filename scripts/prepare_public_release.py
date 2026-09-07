"""Stage a software-only source tree and licensed B/C copies; never upload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "provenance"))
from lr_cusum.capsule import digest, seal, verify, write_json
from build_release_manifest import release_files


def software_files() -> list[Path]:
    """Positive selection: no manuscript images, literature or raw spatial data."""
    top = {"README.md", "LICENSE", "DATA_LICENSE", "CITATION.cff",
           "LICENSE_PENDING.md", "pyproject.toml", ".gitignore", ".gitattributes"}
    text = {".py", ".md", ".json", ".csv", ".txt", ".toml", ".yml", ".yaml", ".cff"}
    roots = {"src", "scripts", "configs", "docs", "provenance", ".github"}
    aggregates = {s.strip() for s in (ROOT / "data/manifests/PUBLIC_AGGREGATE_FILES.txt").read_text().splitlines()
                  if s.strip() and not s.startswith("#")}
    chosen = []
    for path in release_files():
        rel = path.relative_to(ROOT)
        if rel.as_posix() in top:
            chosen.append(path)
        elif (rel.parts[0] in roots or rel.as_posix() in aggregates
              or rel.as_posix() in {"data/README.md", "data/CITATION.cff"}
              or rel.parts[:2] in {("data", "demo"), ("data", "schemas"), ("data", "manifests")}):
            if path.suffix.lower() in text and not path.name.endswith("_CN.md"):
                chosen.append(path)
    return chosen


def licensed_copy(source: Path, target: Path) -> dict:
    verify(source)
    baseline = json.loads((source / "MANIFEST.json").read_text())
    target.mkdir()
    for row in baseline["files"]:
        rel = Path(row["path"])
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dest)
    authorization = json.loads((target / "AUTHORIZATION.json").read_text())
    authorization.update(code_license="MIT", data_license="CC-BY-4.0",
                         publication_authorized=True, publication_performed=False)
    write_json(target / "AUTHORIZATION.json", authorization)
    for src, dest in [("docs/REAL_CAPSULES.md", "README.md"),
                      ("DATA_LICENSE", "DATA_LICENSE"), ("data/CITATION.cff", "CITATION.cff")]:
        shutil.copyfile(ROOT / src, target / dest)
    changed_metadata = {"README.md", "AUTHORIZATION.json", "DATA_LICENSE", "CITATION.cff"}
    unchanged = [row for row in baseline["files"] if row["path"] not in changed_metadata]
    for row in unchanged:
        if digest(target / row["path"]) != row["sha256"]:
            raise ValueError("scientific payload changed: " + row["path"])
    metadata = {k: v for k, v in baseline.items() if k != "files"}
    metadata.update(data_license="CC-BY-4.0", packaging_revision="licensed-1",
                    baseline_manifest_sha256=digest(source / "MANIFEST.json"))
    result = seal(target, metadata)
    return {**result, "unchanged_baseline_files": len(unchanged),
            "baseline_manifest_sha256": metadata["baseline_manifest_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsules", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    out = args.destination.resolve()
    if not out.is_relative_to((ROOT / "dist").resolve()):
        parser.error("destination must be a new directory inside this repository's dist/")
    out.mkdir(parents=True, exist_ok=False)
    source_tree = out / "software"
    records = []
    for path in software_files():
        rel = path.relative_to(ROOT)
        dest = source_tree / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        records.append({"path": rel.as_posix(), "bytes": dest.stat().st_size, "sha256": digest(dest)})
    write_json(source_tree / "SOURCE_MANIFEST.json", {"files": records,
        "excluded": ["manuscripts and rendered assets", "papers", "D spatial/raw layer", "Git history"],
        "externally_published": False})
    archive = out / "windowed_cusum_balsa-1.0.0-source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source_tree.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(source_tree).as_posix())
    archive.with_suffix(".zip.sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="ascii")
    result = {"software": {"files": len(records), "bytes": archive.stat().st_size,
                           "sha256": digest(archive)}}
    for profile in ("science", "pixels"):
        result[profile] = licensed_copy(args.capsules / profile, out / profile)
    write_json(out / "PREPARATION.json", {"externally_published": False, "archives": result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
