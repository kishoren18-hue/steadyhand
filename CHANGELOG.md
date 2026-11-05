# Changelog

## 0.2.0 (2026-11-05)

- `steadyhand` is now an installable package with a small public API:
  `ConformalCutoff` and `AdaptiveCutoff`.
- New: adaptive cutoff with a per-user misfire guarantee on any data sequence,
  tested under electrode shift (21 people) and limb-position change (12 people).
- New: motion-level controller ("act on what's certain") on 3DC.
- New datasets: CIIL electrode shift, Fougner limb position.
- 190 tests, including checks against LibEMG's own rejection and voting code
  and worst-case sequences for the adaptive guarantee; continuous testing on
  Python 3.10 to 3.12.

## 0.1.0 (2026-11-04)

- Conformal cutoff across people, personal check, majority-vote calibration.
- UCI (36 people) and 3DC (22 people), shared and user-specific models.
- Interactive demo.
