"""Export explicitly allowlisted files into a new, clean GitHub candidate directory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/public_export.json"
PRIVATE_PARTS = {"runs", "archive", "private", "local", "capsules", ".git", ".env", "__pycache__"}
RAW_SUFFIXES = {".xlsx", ".xls", ".parquet", ".npz", ".npy", ".tif", ".tiff", ".gpkg", ".shp", ".geojson", ".pkl"}
SECRET = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b|\bAIza[0-9A-Za-z_-]{30,}\b|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])|/mnt/|/home/")
PRIVATE_RECORD = re.compile(r"balsa:mag|mag_fc001|r-\d{4,}_c-\d{4,}|\bBPR\d{4}\b", re.I)
PRIVATE_COLUMNS = {"prediction_id", "verified_event_id", "sample_id", "polygon_id", "longitude", "latitude", "geometry", "evidence_notes", "reviewer"}


def relative_file(root: Path, name: str) -> Path:
    path = root / name
    if ":" in name or "\\" in name or Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path leaves source root: " + name)
    if not path.is_file():
        raise FileNotFoundError(name)
    return path


def collect_files(root: Path, config: dict) -> dict[str, Path]:
    entries = dict(config["files"])
    for folder, pattern in config.get("trees", {}).items():
        directory = root / folder
        if not directory.is_dir() or not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("Missing or external source tree: " + folder)
        for path in sorted(directory.rglob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                name = path.relative_to(root).as_posix()
                if name in entries and entries[name] != name:
                    raise ValueError("Conflicting export mapping: " + name)
                entries[name] = name
    files = {}
    for destination, source in entries.items():
        parts = Path(destination).parts
        if ":" in destination or "\\" in destination or Path(destination).is_absolute() or ".." in parts or PRIVATE_PARTS.intersection(parts):
            raise ValueError("Unsafe export destination: " + destination)
        if Path(destination).suffix.lower() in RAW_SUFFIXES:
            raise ValueError("Raw/private data type: " + destination)
        files[destination] = relative_file(root, source)
    return files


def privacy_findings(files: dict[str, Path], path_exemptions=(), approved_pdfs=None, approved_binaries=None) -> dict[str, list[str]]:
    findings = {}
    for name, path in files.items():
        # Explicit upstream template archives/icons only; changing bytes requires
        # a new reviewed hash. Other binary files still fail the text scanner.
        if name in (approved_binaries or {}):
            if path.suffix.lower() not in {'.zip', '.jpeg'} or approved_binaries[name] != hashlib.sha256(path.read_bytes()).hexdigest():
                findings[name] = ['Third-party binary not approved at this content hash']
            continue
        # Only the author's reviewed publication PDFs bypass the text scanner.
        if path.suffix.lower() == '.pdf':
            if (approved_pdfs or {}).get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
                findings[name] = ['PDF not approved at this content hash']
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            findings[name] = ["unapproved binary or non-UTF-8 content"]
            continue
        problems = []
        if SECRET.search(text):
            problems.append("credential-like content")
        # The privacy validators themselves contain search expressions, not records.
        if name != "tooling/release/export_public.py":
            if ABSOLUTE_PATH.search(text) and name not in path_exemptions:
                problems.append("machine-specific absolute path")
            if PRIVATE_RECORD.search(text):
                problems.append("private record identifier")
        if path.suffix.lower() == ".csv" and not name.startswith(("data/demo/", "data/schemas/")):
            header = next(csv.reader(text.splitlines()), [])
            if PRIVATE_COLUMNS.intersection(c.strip().lower() for c in header):
                problems.append("individual-record columns")
        if problems:
            findings[name] = problems
    return findings


def export(root: Path, output: Path, config: dict) -> dict:
    root = root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root / "dist"):
        raise ValueError("Export must be inside the repository dist directory")
    if output.exists():
        raise FileExistsError("Refusing stale or nonempty export; choose a new output directory")
    files = collect_files(root, config)
    findings = privacy_findings(files, config.get("absolute_path_exemptions", []), config.get("approved_pdfs", {}), config.get("approved_binaries", {}))
    if findings:
        # Never print matched contents, credentials, or private notes.
        raise ValueError("Public export blocked; file names and categories only: " + json.dumps(findings))
    output.mkdir(parents=True)
    manifest = {}
    for name, source in sorted(files.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        if manifest[name] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise AssertionError("Source changed during export: " + name)
    (output / "PUBLIC_FILES.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"file_count": len(manifest), "output": output.as_posix(), "privacy_findings": findings,
            "copied_bytes_verified": True, "manuscript_text_modified": False, "pushed": False}


def verify(directory: Path) -> dict:
    """Verify a public checkout without the author's private source mappings."""
    directory = directory.resolve()
    manifest = json.loads((directory / "PUBLIC_FILES.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Expected a nonempty public-file manifest")
    if (directory / ".git").exists():
        tracked = subprocess.run(["git", "-C", str(directory), "ls-files", "-z"],
                                 check=True, capture_output=True).stdout.decode("utf-8").split("\0")
        extra = sorted(set(tracked) - set(manifest) - {"", "PUBLIC_FILES.json"})
        if extra:
            raise ValueError("Tracked files outside public manifest: " + json.dumps(extra))
    files = collect_files(directory, {"files": {name: name for name in manifest}})
    changed = [name for name, path in files.items()
               if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]]
    if changed:
        raise ValueError("Manifest hash mismatch: " + json.dumps(changed))
    config_path = directory / 'configs/public_export.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    findings = privacy_findings(files, approved_pdfs=config.get('approved_pdfs', {}), approved_binaries=config.get('approved_binaries', {}))
    if findings:
        raise ValueError("Privacy findings (file names and categories only): " + json.dumps(findings))
    return {"file_count": len(files), "manifest_verified": True, "privacy_findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path, help="New author-workspace directory beneath dist; never overwrite an earlier export")
    operation.add_argument("--verify", type=Path, metavar="DIRECTORY", help="Verify PUBLIC_FILES.json in a public checkout")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = verify(args.verify) if args.verify is not None else export(
        ROOT, args.output, json.loads(args.config.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
