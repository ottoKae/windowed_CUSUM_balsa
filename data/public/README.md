# Public data package

This directory defines the dataset that may accompany the public code
repository. It contains no real plantation geometry, coordinates, polygon
identifiers, reference-event rows, spatial-block identities, reviewer records,
commercial imagery, Esri tiles, or S1-GRiTS pixels.

The public archive is built with:

```bash
python scripts/build_data_release.py public
```

It contains the synthetic example, schemas, aggregate paper tables, the frozen
detector configuration, and release documentation. The generated ZIP and its
SHA-256 sidecar are written under ignored `dist/` and should be deposited in a
versioned research repository only after the code license and table-level data
license are approved.

The GitHub repository is the public companion for code review. The external
repository record should be the citable, immutable home of the versioned data
archive; the two records should link to each other. A DOI must not be added to
the manuscript until that deposit exists.

