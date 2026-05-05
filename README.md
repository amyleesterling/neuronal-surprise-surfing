# Neuronal Surprise Surfing

A discovery-mode entry into the [FlyWire](https://flywire.ai) whole-brain connectome. Instead of starting with a search box, you start with a neuron that **breaks the pattern** — and step through the brain by following anomalies.

Built on top of [murthylab/codex](https://github.com/murthylab/codex) (the official Connectome Data Explorer), with three additions:

1. **`/surprise` — seed feed.** A pre-computed feed of statistical outliers across all 139,255 cells in FlyWire 783: connectivity-degree outliers (z ≥ 3 vs. cell-type peers) and cross-region neuropil bridges (rare input/output neuropil pair combinations). Each card is a door, not a search result.

2. **`/surprise?path=…` — journey mode.** Click any card, drop into a persistent 3D canvas. Your trail of past cells stays visible (in fading violet), your current cell glows cyan, and yellow lines connect every pair in your trail that actually synapses. A right-side pane surfaces *contextually-related* surprises — partners, cells sharing your rare neuropil pairs, NBLAST-similar shapes — that themselves break a pattern. Click any to step forward. The journey generates itself.

3. **Compatibility shims** so the public-source codebase can load the GCS-published v2 pickle on Python 3.10 with the public `Connections` class.

## Why this exists

The blank-canvas problem in connectome explorers: people search for what they already know, confirm priors, and leave without discovering anything. Surprise surfing turns that around — the algorithm finds what's anomalous, surfaces it as an entry point, and lets the user follow curiosity through the brain itself in 3D.

Conceptually: **surprise is the origin of curiosity.** A model violation creates an information gap; the gap pulls the brain toward an explanation. *Why?* opens the question. *How?* opens the path. The tool is designed to manufacture that loop reliably.

## Run it locally (Windows)

```powershell
git clone https://github.com/amyleesterling/neuronal-surprise-surfing.git
cd neuronal-surprise-surfing
.\run_local.ps1
```

Then visit **http://localhost:8080/surprise**.

First run downloads the FlyWire 783 NeuronDB pickle (~120 MB) from `storage.googleapis.com/flywire-data/codex/data/fafb/783/`. Subsequent launches are instant.

## Run it locally (macOS / Linux)

```bash
pip install Flask==3.0.3 requests==2.31.0 user-agents==2.2.0 nglui==2.7.2
export FLASK_SECRET_KEY=dev-only-not-secret
export PORT=8080
python -m codex.main
```

## What changed vs. upstream codex

| File | Change |
|---|---|
| `codex/service/surprise_scoring.py` | **NEW** — anomaly engine: `compute_feed`, `compute_related`, partner-index from `connected_pairs` |
| `codex/templates/surprise.html` | **NEW** — single template, two modes (seed feed + journey 3D canvas) |
| `codex/blueprints/base.py` | Added `/surprise` route (state-aware on `?path=`) + homepage card |
| `codex/utils/nglui.py` | Added `url_for_journey()` (current cell cyan, trail violet) and inline line-annotation layer for synaptic links between journey cells |
| `codex/data/connections_v2.py` | **NEW** — module shim; the published v2 pickle imports `codex.data.connections_v2.ConnectionsV2` |
| `codex/data/connections.py` | `num_synapses()` / `num_connections()` made defensive against the v2 pickle's `connected_pairs`-only instance shape |
| `codex/data/local_data_loader.py` | `datetime.UTC` shim for Python 3.10; relaxed strict schema check (the GCS pickle has 3 extra fields the public source doesn't declare) |
| `run_local.ps1` | **NEW** — Windows launcher |

## Real biology surfaced (validation)

First run on FlyWire 783 caught:
- **LO.5423** at z = 26.6σ — a lobula visual neuron wired far beyond its peers
- **OA-VUMa7** — known octopaminergic ventral unpaired neuron with extreme cross-region reach
- **DNpe053** — a descending neuron, biologically a cross-region bridge by definition

The algorithm independently rediscovered known wide-projecting cell classes. The signal isn't noise — it's where to look.

## Status

Prototype. The synaptic-links visualization connects soma-to-soma (the v2 pickle records pair existence, not synapse coordinates) — a real synapse-level layer would require wiring in the FlyWire CAVE synapse table.

Pull requests welcome. The original [codex](https://github.com/murthylab/codex) is the upstream.

## License

Same as upstream codex — see [LICENSE](LICENSE).
