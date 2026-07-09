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

## Still to decide before Phase P — DO NOT skip
- [ ] Definition of done (v1 scope — suggestion: tendency model live for grass, one
      dashboard view, calibration report for the predictive model)
- [ ] Non-goals (protect evenings from Flashpoint/gk-posterior collisions: no in-match
      live updating? no WTA in v1? no betting layer?)
- [ ] Returner anticipation: latent (equilibrium assumption) vs observed (return position)
- [ ] Predictive model form: GBM baseline vs point-level O'Malley vs Bayesian hierarchical
