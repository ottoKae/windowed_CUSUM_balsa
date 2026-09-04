# Figure manifest

The table lists the eight figure assets referenced by the active manuscript.
Of these, only Figure 3 has a public builder in this repository. Figures 1, 2, and
4--8 are frozen, checksum-verified publication assets whose reconstruction
requires authorized private or geospatial inputs. For those figures, the paths
in the final column record their source provenance in the private authoring
workspace; they are not public repository paths or runnable public commands.

| Printed figure | Asset | Role | Public rebuild status or private source provenance |
|---:|---|---|---|
| 1 | `fig01_national_recurrence.pdf` | National recurrence map and distributions | Frozen PDF; private source: `analysis/workflows/build_figure6_recurrence_10km.py` (not included) |
| 2 | `fig02_study_area.pdf` | Study area and official BALSA inventory | Frozen PDF; private sources: `scripts/build_study_area_green_yellow.py` and `scripts/build_study_area_figure.py` (not included) |
| 3 | `fig03_workflow.pdf` | End-to-end methodological workflow | Public builder: `scripts/figures/build_workflow.py` |
| 4 | `fig04_windowed_cusum_concept.pdf` | Windowed CUSUM concept and event-admission semantics | Frozen PDF; private source: `analysis/workflows/build_windowed_cusum_concept.py` (not included) |
| 5 | `fig05_population_s1_signatures.pdf` | Population-level Sentinel-1 disturbance signatures | Frozen PDF; private source: `analysis/workflows/build_population_s1_signature_figure_v11.py` (not included) |
| 6 | `fig06_representative_timeseries.pdf` | Representative polygon time series and image interpretation | Frozen PDF; private source: `analysis/workflows/build_timeseries_signature_polygon_iqr.py` (not included) |
| 7 | `fig07_representation_configuration.pdf` | Stand representation, temporal geometry, and polarization sensitivity | Frozen PDF; private source: `analysis/workflows/build_figure5_abcde_2col3row.py` (not included) |
| 8 | `fig08_polarization_intervals.pdf` | Cross-polarization interval sensitivity | Frozen PDF; private source: `scripts/summarize_n1_polarization_interval_sensitivity.py` (not included) |

To rebuild the public workflow figure without overwriting the frozen manuscript
asset, run from the repository root:

```bash
python scripts/figures/build_workflow.py manuscript/figures/workflow_rebuilt.pdf
```

Use `SHA256SUMS.txt` to verify all eight released PDFs. Rebuilding the other
seven figures belongs to the authorized Level 3 replay described in
`../../docs/REPRODUCIBILITY.md`.

Figures 1 and 2 use an Esri World Hillshade background with Natural
Earth boundaries. Figure 6 embeds a derived PlanetScope visual comparison.
Their final public use remains subject to the attribution and permission check
listed in `../README.md`; raw third-party imagery is not included here.
