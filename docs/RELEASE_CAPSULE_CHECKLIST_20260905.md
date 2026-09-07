# B/C release checklist

The owner authorized B/C preparation, not the D spatial/raw-data layer.
This document lists the decisions still needed before external deposition.

- Completed: the owner approved MIT for original code and CC BY 4.0 for the
  processed B/C data and study-generated aggregates on 2026-09-05.
- Finalize citation authors, title, version and permanent deposit identifiers.
- Confirm which manuscript version is the release entry point. Both existing
  manuscript trees are preserved; neither has been silently designated final.
- Confirm redistribution permission for the already rendered third-party image
  content in manuscript figures, or distribute the corresponding lawful subset.
  Source imagery and basemap caches are not in B/C.
- Review linkage risk from rich real time series and local pixel shapes. Opaque
  IDs and removed georeferencing reduce direct disclosure but are not a formal
  anonymity guarantee.
  Original deterministic permutation seeds are also retained for exact replay;
  they may be linkable if an outsider knows the source-ID construction scheme.
- For a public download, transfer the exact ZIPs and SHA-256 sidecars from the
  final local package, then test the external download. No remote URL is invented.
- Verify the minimum supported Python version and target platforms separately.
  The recorded local runtime is Python 3.12.13 on Windows; no broader environment
  validation is claimed.

Full geographic figures cannot be rebuilt solely from B/C. Already rendered
figures remain usable in the supplied LaTeX source, subject to figure rights.
Fresh cloud CCDC execution is opt-in and is different from the verified offline
official-response replay.
