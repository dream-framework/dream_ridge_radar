# DREAM Index Ridge Failure Radar

Single-screen GitHub Pages app for testing DREAM/S2 structural-market diagnostics on major equity indices.

## What changed in this build

- Reworked the UI into a no-page-scroll desktop dashboard.
- Added meta-tabs: Overview, Ridge Theater, Dust + lambda_q, Scorecard, Audit.
- Kept the always-visible top layer: controls, global state, selected-index KPIs, chart, current read, narrative, matrix/scorecard panels.
- Reduced visual weight: finer typography, lower font weights, denser panels, higher contrast, shorter narrative blocks.
- Retained the research discipline: no h1 trading, no live orders, no hidden dummy rows. Demo data is explicitly marked when live fetch is unavailable.

## Data flow

GitHub Actions runs `scripts/build_index_radar.py` and writes:

```text
data/derived/market_ridge_radar.json
```

The static frontend reads that JSON only.

## Local preview

Serve the folder instead of opening `index.html` directly, because the browser fetches JSON:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Interpretation

The app is a structural-health radar, not a price predictor. It tests whether retained ridge health, operational residual-cloud thickening, and rolling S2 coherence-scale instability improve event detection for major drawdowns and bull runs versus simpler baselines.
