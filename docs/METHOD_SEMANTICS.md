# Frozen method semantics

This document fixes details that can otherwise produce numerically different
implementations of Windowed CUSUM. `configs/paper_v1.json` is the corresponding
machine-readable protocol.

## Monthly sequence and missing observations

Each trajectory has one calendar position per month. Missing SAR observations
remain NaN at that position; zero is never used as an imputation value. The
public table entry point rejects an omitted internal month so that a shortened
array cannot silently shift all subsequent dates.

The paper used January 2016--June 2026 as context and admitted reported events
only from January 2017 through December 2025.

## Window and decision interval

For sequence length `T`, window length `W`, and displacement `S`, logical starts
are `-S, 0, S, ...`. Each context is clipped to
`[max(0, start), min(T, start + W))`. Its decision interval is
`[max(0, start + S), min(T, start + W - S))`.

The frozen values `W=48` and `S=12` therefore define a 24-month central decision
interval for an untrimmed window. Edge windows may be clipped; their available
months and decision bounds remain explicit in the candidate ledger.

## Candidate boundary and direction

CUSUM is calculated on the finite observations within a window, then its index
is mapped back to the original calendar position. In the implementation, the
reported index `t*` is included in the post segment:

```text
pre  = window[:t*]
post = window[t*:]
```

A candidate must have finite observations on both sides and satisfy
`mean(post) < mean(pre)`. The reported `change_time` is the calendar month at
`t*`. This is an implementation boundary, not a claim that physical change
occurred at a known day within that month.

## N1 permutation admission

Only candidates that pass the calendar, validity, central-decision, and negative
shift gates are scored under the final N1 null. Finite observations in the
window are independently permuted and the CUSUM amplitude is recomputed.

For `k` randomized amplitudes at least as large as the observed amplitude and
`B` permutations, the p-value is

```text
p_perm = (1 + k) / (B + 1)
```

The initial test uses `B=3999`. Candidates with
`0.015 <= p_perm <= 0.035` are recomputed from the same deterministic candidate
seed with `B=19999`. Admission requires `p_perm <= 0.025`. This is a p-value for
the stated within-window null; it is neither a posterior disturbance probability
nor global false-discovery control.

## Consolidation and polarization

The detector first retains auditable overlapping-window candidates. In the paper
workflow:

1. same-band candidate rows and orbit-specific detections are consolidated
   within one calendar month at the polygon level;
2. polygon-first VV is the primary event chronology;
3. each VV event is matched one-to-one to a VH detection within two months for a
   sensitivity subset.

VH support does not replace the primary VV chronology.

## Validation

Reference events and detections are matched one-to-one within plus or minus two
calendar months. Train/validation/test or outer-fold roles are assigned at the
polygon and spatial-block level. Confidence intervals resample complete spatial
blocks, never individual events. Reference-silent polygons are audit controls,
not verified negative samples.

## Interpretation boundary

The national product is a chronology of SAR-detected stand disturbances. A
negative departure can be compatible with harvesting or another severe canopy
disturbance, but cause attribution requires independent evidence.
