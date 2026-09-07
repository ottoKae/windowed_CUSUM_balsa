# September 5 monthly NRT / official CCDC comparison

These aggregates supplement, rather than overwrite, the earlier benchmark
tables in results/formal_validation/. Their source hashes are in SOURCES.json.

The six-row method_metrics.csv includes the held-out-selected Windowed CUSUM,
ReCuSum decrease, fixed 48/12 Windowed control, source-default monthly NRT,
spatially selected monthly NRT, and official VV CCDC. The two primary contrasts
use the same 5,000 spatial-block bootstrap draws; familywise columns use the
predeclared two-comparison 97.5% intervals.

direction_control_metrics.csv is a separate diagnostic family. Its additional
negative-direction constraint and VV/VH settings must not be relabelled as
the unmodified NRT or common-VV CCDC benchmark.

The reference denominator is 1,093 dated events in 462 polygons and 57 blocks,
with five frozen outer folds and one-to-one event matching within two months.
MAE concerns matched events. This is reference-catalogue-conditional event
evaluation, not national land-cover accuracy or a verified national negative
denominator.

ccdc_grid_status.csv explicitly separates completed and failed parameter
configurations. Failure is not a zero-event detection result. The B capsule
retains the official response arrays; the offline replay command does not
contact Google Earth Engine.
