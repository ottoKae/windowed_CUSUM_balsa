# Data sources and access

The numerical study dataset is published as [V1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.24223182). It contains the retained FC001 balsa inventory, dated references, monthly VV/VH arrays and frozen pixel/TDA/STP products. GitHub contains the code, documentation and aggregate results. See the [data licence and attribution record](../dataset/LICENSES.md) for source credits and processing modifications.

| Source | Role | Where to obtain the study material |
|---|---|---|
| Ecuador MAG/SPF FC001 forest-plantation inventory | Fixed balsa stand boundaries and attributes | Retained study inventory in Zenodo; CC BY-NC-SA 4.0 |
| NASA/JPL/OPERA RTC-S1 products distributed by ASF DAAC | Sentinel-1 radar observations | Monthly numerical derivatives in Zenodo; raw bursts are not mirrored |
| S1-GRiTS | Organization of track-specific monthly radar observations | Extracted fixed-pixel VV/VH trajectories in Zenodo |
| Dated harvest records | Stand-level reference evaluation | Released reference table in Zenodo, with each record's date basis retained |
| PlanetScope | Optical interpretation and illustrated cases | Approved paper figures are included; original imagery and private interpretation files are not distributed |
| Administrative, climate and relief layers | National maps and regional summaries | Source requirements are listed in the figure provenance; GitHub supplies aggregate canton and climate statistics, not the original map layers |

The [dataset guide](../dataset/README.md) explains identifiers, array ordering and replay commands. [Results](../results/README.md) links the current reference evaluation, component comparisons and national summaries. [Figure provenance](figures/README.md) connects each approved PDF with its rendering scripts and input records.

Original imagery, extraction caches, earlier sample collections and detailed interpretation ledgers remain in the ignored author workspace. Historical paths document those experiments; they are not additional public downloads or alternative current reference sets. The public reference evaluation uses 459 stands and 1,049 dated records.

The released research compilation and numerical outputs use CC BY-NC-SA 4.0; software uses MIT. Underlying sensor products and other third-party materials retain their own notices. Providing an approved figure does not redistribute its source imagery.
