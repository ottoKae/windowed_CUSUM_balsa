# Configuration entry point

`standard_w36.json` is the current W36+B20 standard. `requirements-reproduce.txt` records the tested environment.

Its original +/-2-month evaluation field is retained for historical reproduction.
The final evaluation overlay is
`../results/final_reference_score_20261015/evaluation.json`, with +/-3-month
primary display selected after inspecting earlier scores and +/-1 and +/-2
sensitivity retained. This overlay does not change detection or B20 aggregation.
`public_export.json` is the explicit public file-selection manifest.

Other retained files are dependencies of old author workflows or tests, not alternative defaults. Unreferenced historical configurations were moved to `archive/legacy_20261004/configs`. The public software export includes only the standard configuration and its dependency constraints. Never silently substitute a W48/polygon-first configuration for W36.
