# Figure manifest

The active manuscript references eight frozen PDF assets. Each has exactly one
canonical builder in `scripts/figures/`; `build_all.py` is the sole orchestrator.
Rebuilds are written outside the manuscript by default so that a diagnostic run
cannot silently replace publication assets.

| Printed figure | Asset | Role | Canonical builder |
|---:|---|---|---|
| 1 | `fig01_national_recurrence.pdf` | National recurrence map and distributions | `build_national_recurrence.py` |
| 2 | `fig02_study_area.pdf` | Study area and official BALSA inventory | `build_study_area.py` |
| 3 | `fig03_workflow.pdf` | End-to-end methodological workflow | `build_workflow.py` |
| 4 | `fig04_windowed_cusum_concept.pdf` | Windowed CUSUM concept and event-admission semantics | `build_windowed_cusum_concept.py` |
| 5 | `fig05_population_s1_signatures.pdf` | Population-level Sentinel-1 disturbance signatures | `build_population_s1_signatures.py` |
| 6 | `fig06_representative_timeseries.pdf` | Representative polygon time series and image interpretation | `build_representative_timeseries.py` |
| 7 | `fig07_representation_configuration.pdf` | Stand representation, temporal geometry, and polarization sensitivity | `build_representation_configuration.py` |
| 8 | `fig08_polarization_intervals.pdf` | Cross-polarization interval sensitivity | `build_polarization_intervals.py` |

From the repository root, rebuild all figures without overwriting the frozen
assets:

```bash
python scripts/figures/build_all.py --output-dir runs/figures/rebuilt
```

Use `--figure N` to build one figure. Only an intentional publication step
should add `--publish`. Figure 6 needs the ignored licensed-source configuration
described in `../../configs/README.md`; raw third-party imagery is not included.

Use `SHA256SUMS.txt` to verify the eight released PDFs. Figures 1 and 2 use the
authorized Natural Earth context, and Figure 6 embeds a derived PlanetScope
comparison. Their public use remains subject to the permission and attribution
checks in `../README.md`.
