"""Keep private records and stale files out of the explicit public export."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("public_export", Path(__file__).resolve().parents[1] / "tooling/release/export_public.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_export_refuses_existing_destination_and_preserves_source(tmp_path):
    source = tmp_path / "README.md"
    source.write_text("Public description", encoding="utf-8")
    output = tmp_path / "dist/public"
    config = {"files": {"README.md": "README.md"}}
    result = module.export(tmp_path, output, config)
    assert result["file_count"] == 1
    assert (output / "README.md").read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        module.export(tmp_path, output, config)


@pytest.mark.parametrize("destination", ["runs/records.csv", "../outside.csv", "raw.xlsx", "C:outside.csv"])
def test_private_or_escaping_destination_rejected(tmp_path, destination):
    (tmp_path / "source.txt").write_text("safe")
    with pytest.raises(ValueError):
        module.collect_files(tmp_path, {"files": {destination: "source.txt"}})


def test_individual_review_rows_rejected_before_copy(tmp_path):
    source = tmp_path / "records.csv"
    source.write_text("prediction_id,judgement\np1,T\n", encoding="utf-8")
    with pytest.raises(ValueError, match="individual-record columns"):
        module.export(tmp_path, tmp_path / "dist/public", {"files": {"results/records.csv": "records.csv"}})
    assert not (tmp_path / "dist/public").exists()


def test_archived_code_can_be_mapped_without_archiving_output(tmp_path):
    folder = tmp_path / "archive"
    folder.mkdir()
    (folder / "builder.py").write_text("print('public')")
    result = module.export(tmp_path, tmp_path / "dist/public", {"files": {"scripts/builder.py": "archive/builder.py"}})
    assert result["file_count"] == 1


def test_public_checkout_verification_needs_no_archived_sources(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("public material", encoding="utf-8")
    output = tmp_path / "dist/public"
    module.export(tmp_path, output, {"files": {"README.md": "source.txt"}})
    source.unlink()
    assert module.verify(output)["manifest_verified"] is True
    (output / "README.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        module.verify(output)


@pytest.mark.parametrize("name", ["CITATION.cff", ".github/workflows/test.yml", "LICENSE", "template.sty"])
def test_privacy_checks_include_metadata_workflows_and_suffix_free_text(tmp_path, name):
    source = tmp_path / "source.txt"
    source.write_text("token: " + "ghp_" + "a" * 30, encoding="utf-8")
    assert module.privacy_findings({name: source}) == {name: ["credential-like content"]}


def test_only_reviewed_pdf_bytes_are_exported_and_verified(tmp_path):
    pdf = tmp_path / 'figure.pdf'
    pdf.write_bytes(b'%PDF-1.4\nsynthetic publication fixture\xff')
    config = {'files': {'figure.pdf': 'figure.pdf', 'configs/public_export.json': 'config.json'},
              'approved_pdfs': {'figure.pdf': hashlib.sha256(pdf.read_bytes()).hexdigest()}}
    (tmp_path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    output = tmp_path / 'dist/public'
    module.export(tmp_path, output, config)
    assert module.verify(output)['manifest_verified']
    pdf.write_bytes(pdf.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='PDF not approved'):
        module.export(tmp_path, tmp_path / 'dist/changed', config)
    with pytest.raises(ValueError, match='PDF not approved'):
        module.export(tmp_path, tmp_path / 'dist/unreviewed', {'files': {'figure.pdf': 'figure.pdf'}})


def test_editable_binary_is_not_covered_by_pdf_approval(tmp_path):
    editable = tmp_path / 'layout.pptx'
    editable.write_bytes(b'PK\x03\x04\xff')
    assert module.privacy_findings({'layout.pptx': editable})


def test_git_checkout_rejects_unlisted_tracked_files_but_allows_local_outputs(tmp_path):
    (tmp_path / "source.txt").write_text("public")
    output = tmp_path / "dist/public"
    module.export(tmp_path, output, {"files": {"README.md": "source.txt"}})
    subprocess.run(["git", "init", str(output)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(output), "add", "README.md", "PUBLIC_FILES.json"], check=True)
    (output / "local_output.txt").write_text("invented local output")
    assert module.verify(output)["manifest_verified"]
    subprocess.run(["git", "-C", str(output), "add", "local_output.txt"], check=True)
    with pytest.raises(ValueError, match="Tracked files outside public manifest"):
        module.verify(output)
