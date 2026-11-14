# Public release scope

The GitHub release contains the current windowed CuSum ? TDA ? STP implementation, its required helpers, synthetic examples, selected tests, aggregate results, manuscript and seven approved figures. Exact file selection is maintained in `configs/public_export.json`; the exported `PUBLIC_FILES.json` records the distributed file hashes.

Real stand boundaries, monthly VV/VH arrays, harvest references and frozen detection/TDA/STP products are published separately as [Zenodo V1.0.0](https://doi.org/10.5281/zenodo.24223182), under CC BY-NC-SA 4.0. They are not duplicated in GitHub. The code uses MIT; third-party notices remain in their respective files.

Original PlanetScope imagery, raw SAR bursts, private interpretation notes, credentials, local caches, editable artwork and historical archives remain local. The approved paper and figures include author information and illustrated study locations. Their publication does not release the underlying commercial imagery.

Older paths needed by current imports or recorded experiments remain intact. Files are selected by their role and dependencies, not by whether their names contain a date or the word `diagnostic`.

The author workspace exports a clean snapshot without its `.git` history. Ignore rules alone do not retract files that were already published. The release procedure verifies the selected files and approved PDF hashes before publication; see [the release guide](../docs/PUBLIC_RELEASE.md).
