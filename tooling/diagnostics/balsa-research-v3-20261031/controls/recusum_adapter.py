"""Notebook-derived ReCuSum with first-post boundaries throughout recursion."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import time

import numpy as np

from run_controls import (ROOT, OUT, SOURCE, DATES, freeze, sha, save, stable_seed)
from lr_cusum.baselines import cusum_single, recusum_detect


def firstpost_recusum(values, *, seed, confidence=.9, bootstrap=800,
                     minimum_segment=12, maximum_depth=10):
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2*minimum_segment or maximum_depth <= 0:
        return []
    peak, _ = cusum_single(values, confidence=confidence, bootstrap=bootstrap, seed=seed)
    if peak is None:
        return []
    after = valid[valid > peak]
    if not len(after):
        return []
    point = int(after[0])
    # The corrected split assigns the peak observation to the left segment.
    left = firstpost_recusum(values[:point], confidence=confidence, bootstrap=bootstrap,
        seed=seed+point, minimum_segment=minimum_segment, maximum_depth=maximum_depth-1)
    right = firstpost_recusum(values[point:], confidence=confidence, bootstrap=bootstrap,
        seed=seed+point+1, minimum_segment=minimum_segment, maximum_depth=maximum_depth-1)
    return sorted(set(left+[point]+[point+p for p in right]))


def pilot():
    units, _ = freeze()
    notebook = ROOT.parent / "LR-CuSum_v100/notebooks/LR-CuSum.ipynb"
    source = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][122]["source"])
    tree = ast.parse(source)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"cusum_single", "recusum_detect"}]
    assert len(funcs) == 2
    namespace = {"np": np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(notebook)+":cell122", "exec"), namespace)
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")
    selected = units.iloc[np.linspace(0, len(units)-1, 32, dtype=int)]
    checks = []; started = time.monotonic(); adapted_seconds = 0
    for row in selected.itertuples():
        vv = values[row.series_index]; seed = stable_seed(row.pixel_track_id, 20260820)
        original = namespace["recusum_detect"](vv, tc=.9, n_boot=800, seed=seed, min_segment=12, max_depth=10)
        current = recusum_detect(vv, confidence=.9, bootstrap=800, seed=seed, minimum_segment=12, maximum_depth=10)
        assert sorted(set(original)) == current
        began = time.monotonic()
        adapted = firstpost_recusum(vv, seed=seed)
        adapted_seconds += time.monotonic()-began
        valid = np.flatnonzero(np.isfinite(vv))
        shifted = sorted(set(int(valid[valid > p][0]) for p in current if np.any(valid > p)))
        checks.append(dict(unit_id=row.unit_id, pixel_track_id=row.pixel_track_id,
            notebook_peak_points=[int(p) for p in original], legacy_wrapper_peak_points=current,
            final_date_shift_only=shifted, firstpost_recursive_points=adapted,
            recursive_change_vs_final_shift=adapted != shifted))
    step = np.where(DATES < "2020-07-01", -8., -12.)
    assert firstpost_recusum(step, seed=20) == [54]
    missing = step.copy(); missing[53:55] = np.nan
    assert firstpost_recusum(missing, seed=20) == [55]
    repeat = np.repeat([-8., -12., -8., -12.], [30, 30, 30, 36])
    assert firstpost_recusum(repeat, seed=20) == [30, 60, 90]
    assert firstpost_recusum(np.full(126, -10.), seed=20) == []
    save(OUT / "recusum/pilot.json", dict(checks_passed=True, notebook_cell=122,
        notebook_sha256=sha(notebook), code_sha256=sha(Path(__file__)),
        legacy_notebook_set_parity_pixels=len(checks), cases=checks,
        ideal_cases=["constant", "step", "missing_firstpost_month", "three_boundaries"],
        pilot_adapted_cpu_seconds=adapted_seconds, pilot_total_seconds=time.monotonic()-started,
        estimated_full_seconds_four_workers=adapted_seconds/len(checks)*len(units)/4,
        status="adapted_not_official", original_paper_or_author_code_validated=False,
        changes=["peak→next finite first-post boundary at every recursion",
            "recursive split and seed offsets use corrected boundary",
            "historical selected c0.9/m12/bootstrap800 rather than notebook default c0.95",
            "fixed monthly calendar/native pixel seeds and radius6 decrease filter",
            "frozen B20 replaces notebook temporal DBSCAN"],
        unchanged=["absolute cumulative-sum amplitude", "strict permutation exceedance count",
            "CL >= 0.9", "parent >=24 finite observations", "maximum recursion depth 10",
            "800 permutations", "multiple candidate months per pixel-year permitted"]))
    print(json.dumps(dict(pilot_adapted_cpu_seconds=adapted_seconds,
        full_four_worker_seconds=adapted_seconds/len(checks)*len(units)/4,
        recursion_changes=sum(x["recursive_change_vs_final_shift"] for x in checks))), flush=True)


if __name__ == "__main__":
    pilot()
