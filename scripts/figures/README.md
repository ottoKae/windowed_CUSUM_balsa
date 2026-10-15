# Figure generation

For the current manuscript, start with
[the actual TeX figure catalogue](../../provenance/figures/README.md). It follows
current inclusions and asset hashes, corrects the old printed-order builder list,
and identifies archived sources where the simplified entry point is not the
accepted renderer. `build_all.py` remains a historical workflow, not a claim that
one command reproduces the current manuscript byte for byte.

`build_all.py` is the single local figure entry point. Default output goes to `runs/figures`, not the manuscript.

These builders retain the existing manuscript's data sources and include legacy W48/polygon-first results. They have **not** been regenerated for the frozen W36+B20 release. Use `docs/REPRODUCTION_STATUS.md` to check the result version before using a figure. No figure or TeX file was changed by the W36 repository cleanup.

Shared existing templates under `analysis/workflows` remain compatibility dependencies; moving those independently would break the retained builders. `preview_*` scripts are author-side layout tools, not scientific entry points.
