"""Load experiment entries in isolated processes, without private data or old imports."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = json.loads((ROOT / "provenance/figures/source_spec.json").read_text(encoding="utf-8"))["figures"]


@pytest.mark.parametrize("relative", [
    "scripts/fc001_validation_detection.py",
    "scripts/run_pixel_b20_benchmark_window.py",
    "scripts/run_pixel_b20_bfast_monitor.py",
    "scripts/run_pixel_b20_ewmacd_monitor.py",
    "tooling/diagnostics/balsa-results-ablation-20261101/polarization/run_polarization.py",
    "tooling/diagnostics/balsa-results-optimization-20261105/audit_polarization.py",
    "tooling/diagnostics/balsa-research-v3-20261031/controls/recusum_adapter.py",
    "tooling/diagnostics/cumulative-components-20261111/compute.py",
    "tooling/diagnostics/configuration-extension-20261111/geometry.py",
    "tooling/diagnostics/block-length-sensitivity-20261111/run_blocks.py",
    "tooling/diagnostics/detector-tda-benchmark-20261109/cloud.py",
    "tooling/diagnostics/range-primary-20261111/benchmark.py",
    "tooling/diagnostics/results-refresh-20261111/signatures.py",
] + sorted({path for figure in FIGURES.values() for path in figure["builders"]}))
def test_public_entry_has_its_transitive_imports(relative):
    code = """
import importlib.util
from pathlib import Path
import sys
root, path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(root / 'src'), str(path.parent)]
spec = importlib.util.spec_from_file_location('entry_under_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert not any(n == 'lr_cusum' or n.startswith('lr_cusum.') for n in sys.modules)
if hasattr(module, 'CONFIG') and isinstance(module.CONFIG, Path):
    assert module.CONFIG.is_file(), module.CONFIG
if hasattr(module, 'DETECTOR_CONFIG'):
    assert module.DETECTOR_CONFIG.is_file(), module.DETECTOR_CONFIG
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT), str(ROOT / relative)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
