"""Index actual TeX figure use, source routes and hashes without rendering or editing TeX."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "provenance/figures"
MANUSCRIPT = ROOT / "manuscript_JAG_v2"
TOKEN = re.compile(r"\\(input|include|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}")
AUTHOR_ONLY_CODE = {
    "scripts/figures/preview_reference_v2.py": "Depends on unshipped rerun_reference_v2 author review workflow; not a current TeX figure producer."
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def uses(path: Path, entry: str, stack: tuple = ()) -> list[dict]:
    if path in stack:
        raise ValueError("Cyclic TeX input")
    text = path.read_text(encoding="utf-8-sig")
    # Preserve line counts while excluding TeX comments, including commented-out figures.
    text = "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())
    rows = []
    for item in TOKEN.finditer(text):
        kind, name = item.groups()
        target = MANUSCRIPT / name
        if kind != "includegraphics":
            rows.extend(uses(target.with_suffix(".tex"), entry, (*stack, path)))
        else:
            if not target.suffix:
                target = target.with_suffix(".pdf")
            if not target.is_file():
                raise FileNotFoundError(f"Unresolved graphic: {name}")
            rows.append({"entry_point": entry, "tex_file": relative(path),
                         "tex_line": text.count("\n", 0, item.start()) + 1,
                         "asset": relative(target), "asset_sha256": sha(target)})
    return rows


def file_record(path: str) -> dict:
    value = ROOT / path
    return {"path": path, "exists": value.exists(), "kind": "file" if value.is_file() else "directory" if value.is_dir() else "missing",
            "sha256": sha(value) if value.is_file() else "",
            "public_export_path": path.removeprefix("archive/legacy_20261004/") if path.startswith("archive/legacy_20261004/") and value.suffix == ".py" else ""}


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    protected = sorted(MANUSCRIPT.rglob("*.tex")) + sorted((MANUSCRIPT / "figures").glob("*.pdf"))
    before = {relative(p): sha(p) for p in protected}
    specs = json.loads((OUT / "source_spec.json").read_text(encoding="utf-8"))["figures"]
    bindings, dependencies, details = [], [], []
    per_entry = {}
    for entry in ("main.tex", "supplementary.tex"):
        entry_rows = uses(MANUSCRIPT / entry, entry)
        per_entry[entry] = len(entry_rows)
        for order, row in enumerate(entry_rows, 1):
            name = Path(row["asset"]).name
            spec = specs[name]
            matches = [p for p in spec["candidate_outputs"] if (ROOT / p).is_file() and sha(ROOT / p) == row["asset_sha256"]]
            receipt_match = False
            if "receipt_output_hash" in spec:
                receipt = spec["receipt_output_hash"]
                expected = json.loads((ROOT / receipt["path"]).read_text(encoding="utf-8-sig"))
                for key in receipt["keys"]:
                    expected = expected[key]
                receipt_match = expected.lower() == row["asset_sha256"]
            status = "output_and_receipt_hash_verified" if matches and receipt_match else "output_hash_verified" if matches else "source_route_only"
            record = {**row, "tex_encounter_order": order, "role": spec["role"], "version": spec["version"],
                      "binding_status": status, "source_scripts": ";".join(spec["builders"]),
                      "matched_output": ";".join(matches), "rebuild_status": spec["rebuild_status"], "rights": spec["rights"]}
            bindings.append(record)
            for role in ("builders", "helpers", "inputs", "receipts", "candidate_outputs"):
                for path in spec[role]:
                    dependencies.append({"asset": row["asset"], "dependency_role": role, **file_record(path)})
            details.append({**record, "receipt_output_hash_verified": receipt_match,
                            "dependencies": [d for d in dependencies if d["asset"] == row["asset"]]})
    assert len(bindings) == 8 and len({b["asset"] for b in bindings}) == 8
    assert per_entry == {"main.tex": 8, "supplementary.tex": 0}
    assert {relative(p): sha(p) for p in protected} == before, "Manuscript changed during read-only indexing; rerun against the updated version."
    assert not any(re.search(r"(?:[A-Za-z]:[/\\]|/Users/)", str(d)) for d in dependencies)
    write_csv("figure_bindings.csv", bindings)
    write_csv("figure_dependencies.csv", dependencies)
    code_paths = set((ROOT / "scripts/figures").glob("*.py")) | set((ROOT / "analysis/workflows").glob("build_*.py"))
    code_paths.update(ROOT / d["path"] for d in dependencies if d["dependency_role"] in ("builders", "helpers") and d["path"].endswith(".py"))
    inventory = []
    for path in sorted(code_paths):
        name = relative(path)
        linked = [d for d in dependencies if d["path"] == name]
        inventory.append({**file_record(name), "linked_assets": ";".join(sorted({d["asset"] for d in linked})),
                          "role": "tex_source_or_helper" if linked else "preview" if path.name.startswith("preview_") else "orchestrator" if path.name == "build_all.py" else "alternate_or_additional_builder_not_bound_to_current_tex",
                          "public_export_eligible": name not in AUTHOR_ONLY_CODE,
                          "export_exclusion_reason": AUTHOR_ONLY_CODE.get(name, "")})
    write_csv("figure_code_inventory.csv", inventory)
    public_code = {"schema_version": 1, "scope": "Explicit source-only figure code inventory; no rasters, PDF assets or data are implied",
                   "files": [{"source": row["path"], "destination": row["public_export_path"] or row["path"],
                              "sha256": row["sha256"], "role": row["role"]} for row in inventory if row["exists"] and row["public_export_eligible"]],
                   "excluded_author_only": AUTHOR_ONLY_CODE}
    assert len({r["destination"] for r in public_code["files"]}) == len(public_code["files"])
    (OUT / "public_code_files.json").write_text(json.dumps(public_code, indent=2) + "\n", encoding="utf-8")
    payload = {"schema_version": 1, "scope": "Current TeX asset inventory, not a new render or scientific-result update",
               "entry_point_figure_counts": per_entry, "protected_file_hashes": before,
               "tex_and_figures_unchanged": True, "figures": details}
    (OUT / "figure_provenance.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"figure_count": len(bindings), "dependency_rows": len(dependencies),
                      "hash_bound_assets": sum(b["binding_status"] != "source_route_only" for b in bindings),
                      "code_files": len(inventory), "public_code_files": len(public_code["files"]),
                      "protected_files": len(protected), "unchanged": True}))


if __name__ == "__main__":
    main()
