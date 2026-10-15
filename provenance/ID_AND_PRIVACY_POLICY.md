# Identifier and privacy controls

The private analysis tables contain fields that can identify exact inventory
records or recover locations. Public row-level exports must remove, at minimum:

- `polygon_id`, `sample_id`, `point_track_id`, original `ID`, `OBJECTID`, and
  source-row indices unless replaced with stable public IDs;
- longitude, latitude, projected x/y, WKT/WKB/GeoJSON geometry, raster row/col,
  tile/track identifiers where they create a practical linkage route;
- province, canton, parish, property, owner, manager, and other administrative
  or operational labels;
- historical-point/reference identifiers;
- annotator/reviewer names, workstation usernames, evidence filenames, optical
  image filenames, and local/UNC filesystem paths;
- fine spatial-block IDs used in evaluation.

Public IDs must be assigned by a stored private lookup, not by exposing or
plain-hashing the source identifier. All public tables from the same release
must use the same lookup. The lookup and randomization secret are never released.

The current package avoids this risk by publishing aggregate tables and wholly
synthetic demonstration rows only.

