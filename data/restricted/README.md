# Restricted reproducibility capsule

This directory is documentation only. The real capsule is assembled from the
ignored `data/local/paper_v1` and `runs/` trees and must never be committed to
Git. It contains exact plantation geometries, polygon-interior Sentinel-1
samples, dated events, spatial-block identities, OOF predictions, and detailed
detector outputs.

Build it only for controlled transfer:

```bash
python scripts/build_data_release.py restricted --acknowledge-sensitive-data
```

The command deliberately excludes the multi-terabyte S1-GRiTS Zarr archive,
Planet imagery, Esri basemap tiles, reviewer identities, and local source-path
receipts. S1-GRiTS is referenced by a catalog and should be obtained through
its own authorized distribution route.

Recommended release decision:

1. Keep the complete capsule private while redistribution rights and location
   disclosure are unresolved.
2. If rights permit, deposit a reviewed, de-identified derivative in a
   controlled or embargoed research-data repository, not in GitHub.
3. Publish the aggregate public archive separately. Never apply the software
   license automatically to third-party data.

