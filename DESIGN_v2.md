# Wimbledon Models — Design (Phase D)

## Agreed in kickoff conversation (2026-07-08)
- Two models: (1) predictive match-winner ML; (2) prescriptive Bayesian serve placement.
- Architecture (non-negotiable): sampling offline in scripts/, versioned posterior artifact
  (.nc), FastAPI serves cached posteriors only. /predict p95 < 300ms.
- Serve placement = tendency model (hierarchical multinomial, non-centered, coords/dims)
  + outcome model (P(point won | direction, server, returner)) + decision layer
  (best-response mix, minimax gap with credible intervals).
- Data: TML atp_matches (predictive) + MCP ServeDirection & points-2020s (prescriptive).
  Both CC BY-NC-SA — non-commercial only.
- Language: Python end-to-end. Stack: PyMC + FastAPI (+ Railway), frontend Next.js 14
  (+ Vercel) vibe-coded against frozen JSON schemas.

## Decided (2026-07-14) — Phase D closed, proceed to Phase P

- **Definition of done, v1**: tendency model fit on real grass-court MCP data, served via
  the existing FastAPI `/tendency` endpoint, one dashboard view (server search -> direction
  probabilities + 80% CI, per side/serve-number), and a calibration report for the
  predictive model (Brier score + reliability curve vs tennis-data.co.uk closing odds).
  Outcome model and decision layer (best-response mix, minimax gap) are v1.1, not v1 —
  tendency-only ships first.
- **Non-goals, v1**: no WTA (men's tour only), no live in-match updating (batch-refresh
  posteriors, not streaming), no betting/staking layer. Explicitly protects evenings from
  Flashpoint/gk-posterior/Career Companion collisions.
- **Returner anticipation**: latent — equilibrium assumption only. No return-position
  features in v1. Keeps the decision layer's minimax framing clean; observed anticipation
  is a v1.1+ extension once the return-position data is validated for coverage.
- **Predictive model form**: gradient boosting baseline (LightGBM or sklearn
  HistGradientBoosting) on the surface-Elo + rolling serve/return features already scoped
  in `features.py`. Fastest to iterate and to calibration-check; Bayesian hierarchical or
  point-level O'Malley are explicitly deferred, not ruled out, if GBM calibration disappoints.

## Immediate next steps (Phase P entry)
- [x] Finish `features.randomize_winner_framing` (leakage guard) — was already done pre-Phase P
- [x] GBM predictive baseline (`predictive.py` + `scripts/fit_predictive.py`) — chronological
      split, gated on beating the base-rate Brier score. Validated against synthetic data.
- [x] Calibration script (`calibration.py` + `scripts/calibration_report.py`) — de-vigged
      market probability, Brier score, reliability curve. Validated against synthetic
      closing odds (`scripts/make_synthetic_data.py` now also emits `data/odds/closing_odds.csv`).
- [x] Dashboard: single view, server search -> `/tendency` call -> chart (`web/`, Next.js 14)
- [ ] Fit `fit_tendency.py` on real MCP grass data (currently only run on synthetic) — blocked
      on the real wimbledon-datasets.zip bundle
- [ ] Re-run `calibration_report.py` against real tennis-data.co.uk closing odds once the
      bundle + odds files are in `data/` — same blocker
