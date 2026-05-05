"""Surprise surfing — algorithmically detect statistical outliers in the
connectome and present them as entry points, not search results.

Surprise is the origin of curiosity. The user does not arrive with a question;
they arrive at a neuron that breaks the pattern, and the question forms.

Two signals in v1:
  1. Degree outliers — neurons whose total partner count is anomalously high
     or low compared to peers of the same cell_type / super_class. Z-scored
     against the peer group; |z| >= 3 surfaces as surprising.
  2. Cross-region bridges — neurons whose input/output neuropil pairs span
     region combinations that are rare across the connectome.

Motif frequency anomalies are v2.

Scores are computed once per (neuron_db_id, version) and cached in process.
"""

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from math import log, sqrt
from typing import Dict, List, Optional, Tuple

from codex import logger
from codex.service.neuropil_meanings import role as np_role, short_role as np_short


# ---------------------------------------------------------------------------
# Tunables. Conservative defaults; surface the genuinely weird, not the noise.
# ---------------------------------------------------------------------------
MIN_PEER_GROUP_SIZE = 12          # below this, z-scores are not meaningful
DEGREE_Z_THRESHOLD = 3.0          # 3-sigma for degree outliers
TOP_N_PER_SIGNAL = 3              # how many cards to surface per anomaly type on the landing
TOP_N_HERO = 1                    # number of hero anomalies on the landing
MIN_PARTNERS_FOR_BRIDGE = 50      # ignore tiny neurons for bridge scoring


@dataclass
class SurpriseCard:
    root_id: int
    name: str
    cell_type: str
    super_class: str
    nt_type: str
    side: str
    signal: str                # "degree_outlier" | "cross_region_bridge"
    signal_label: str          # human label, e.g. "Connectivity outlier"
    signal_explainer: str      # one-line definition of the signal
    headline: str              # one-sentence why — the door (short)
    description: str           # 2-3 sentence prose explaining what's surprising
    detail: str                # raw stats for experts
    score: float               # surprise magnitude (sortable)
    peer_group: str            # what we compared against
    input_neuropils: List[str]
    output_neuropils: List[str]
    input_cells: int
    output_cells: int
    # Set by compute_related when a card is suggested from a current cell:
    relation_reason: str = ""

    def as_template_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_cell_type(nd: dict) -> Optional[str]:
    ct = nd.get("cell_type")
    if isinstance(ct, list) and ct:
        return ct[0]
    if isinstance(ct, str) and ct:
        return ct
    return None


def _peer_key(nd: dict) -> Optional[str]:
    """Most-specific reasonable peer group for a neuron."""
    return _primary_cell_type(nd) or nd.get("super_class") or nd.get("class")


def _format_ratio(observed: float, peer_mean: float) -> str:
    if peer_mean <= 0:
        return ""
    ratio = observed / peer_mean
    if ratio >= 2:
        return f"{ratio:.1f}× more"
    if ratio <= 0.5 and ratio > 0:
        return f"{1/ratio:.1f}× fewer"
    return f"{ratio:.2f}×"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, sqrt(var)


# ---------------------------------------------------------------------------
# Signal 1 — degree outliers
# ---------------------------------------------------------------------------

DEGREE_EXPLAINER = (
    "A connectivity outlier is a neuron whose total number of synaptic "
    "partners is much higher or lower than typical for its cell type. "
    "When wiring breaks the pattern, the function often does too."
)
BRIDGE_EXPLAINER = (
    "A cross-region bridge is a neuron whose input region and output region "
    "form a pair almost no other cell makes. If a circuit links those two "
    "regions, this neuron is one of just a few cells carrying it."
)


def score_degree_outliers(neuron_db) -> List[SurpriseCard]:
    """Z-score each neuron's total partner count (input_cells + output_cells)
    against peers of the same cell_type (or super_class)."""

    by_peer: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for rid, nd in neuron_db.neuron_data.items():
        peer = _peer_key(nd)
        if not peer:
            continue
        degree = (nd.get("input_cells") or 0) + (nd.get("output_cells") or 0)
        by_peer[peer].append((rid, degree))

    cards: List[SurpriseCard] = []
    for peer, entries in by_peer.items():
        if len(entries) < MIN_PEER_GROUP_SIZE:
            continue
        degrees = [d for _, d in entries]
        mean, std = _mean_std(degrees)
        if std <= 0:
            continue
        for rid, degree in entries:
            z = (degree - mean) / std
            if abs(z) < DEGREE_Z_THRESHOLD:
                continue
            nd = neuron_db.neuron_data[rid]
            direction = "more" if z > 0 else "fewer"
            ratio_str = _format_ratio(degree, mean)
            headline = (
                f"Wired to {ratio_str} partners than typical {peer} cells."
                if ratio_str
                else f"Wired {abs(z):.1f}σ above the typical {peer} cell."
            )
            description = (
                f"Most {peer} cells in the FlyWire connectome have around "
                f"{int(mean)} synaptic partners. This one has {int(degree)} — "
                f"a {direction}-than-typical wiring pattern that is "
                f"{abs(z):.1f} standard deviations from the peer average. "
                f"Cells this far from their cohort are usually doing something "
                f"unusual: integrating across more inputs, fanning out wider, "
                f"or sitting at a circuit bottleneck."
            )
            cards.append(SurpriseCard(
                root_id=rid,
                name=nd.get("name") or str(rid),
                cell_type=_primary_cell_type(nd) or "",
                super_class=nd.get("super_class") or "",
                nt_type=nd.get("nt_type") or "",
                side=nd.get("side") or "",
                signal="degree_outlier",
                signal_label="Connectivity outlier",
                signal_explainer=DEGREE_EXPLAINER,
                headline=headline,
                description=description,
                detail=(
                    f"z = {z:+.2f}, total partners = {int(degree)}, "
                    f"peer mean = {mean:.1f}, σ = {std:.1f}, "
                    f"n = {len(entries)} peers, group = {peer}."
                ),
                score=abs(z),
                peer_group=peer,
                input_neuropils=list(nd.get("input_neuropils") or []),
                output_neuropils=list(nd.get("output_neuropils") or []),
                input_cells=int(nd.get("input_cells") or 0),
                output_cells=int(nd.get("output_cells") or 0),
            ))

    cards.sort(key=lambda c: c.score, reverse=True)
    return cards[:TOP_N_PER_SIGNAL]


# ---------------------------------------------------------------------------
# Signal 2 — cross-region bridges
# ---------------------------------------------------------------------------

def score_cross_region_bridges(neuron_db) -> List[SurpriseCard]:
    """Surface neurons that connect input/output neuropil pairs which are
    statistically rare across the connectome."""

    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    neuron_pairs: Dict[int, List[Tuple[str, str]]] = {}

    for rid, nd in neuron_db.neuron_data.items():
        ins = nd.get("input_neuropils") or []
        outs = nd.get("output_neuropils") or []
        if not ins or not outs:
            continue
        if (nd.get("input_cells") or 0) + (nd.get("output_cells") or 0) < MIN_PARTNERS_FOR_BRIDGE:
            continue
        pairs = []
        for i_np in ins:
            for o_np in outs:
                if i_np == o_np:
                    continue
                pair = (i_np, o_np)
                pairs.append(pair)
                pair_counts[pair] += 1
        if pairs:
            neuron_pairs[rid] = pairs

    if not neuron_pairs:
        return []

    total_neurons = max(1, len(neuron_pairs))

    cards: List[SurpriseCard] = []
    for rid, pairs in neuron_pairs.items():
        # Surprise = sum of -log(p) over the pairs this neuron bridges,
        # divided by number of pairs (so it doesn't simply reward big neurons).
        score = 0.0
        rarest_pair: Optional[Tuple[str, str]] = None
        rarest_p = 1.0
        for p in set(pairs):
            n_with = pair_counts[p]
            prob = n_with / total_neurons
            if prob > 0:
                score += -log(prob)
                if prob < rarest_p:
                    rarest_p = prob
                    rarest_pair = p
        if not rarest_pair:
            continue
        score /= sqrt(len(set(pairs)))  # damped average — avoid trivially-large neurons winning

        nd = neuron_db.neuron_data[rid]
        in_np, out_np = rarest_pair
        n_with = pair_counts[rarest_pair]
        # Headline stays tight (just the codes); rich roles go in the description.
        headline = f"Connects {in_np} → {out_np}."
        if n_with == 0:
            rarity_phrase = "No other neuron in the connectome makes this same input→output pair."
        elif n_with == 1:
            rarity_phrase = "Only one other neuron in the connectome makes this same input→output pair."
        else:
            rarity_phrase = f"Only {n_with} other neurons in the connectome make this same input→output pair."
        description = (
            f"This cell takes input from {np_role(in_np)} and projects "
            f"output to {np_role(out_np)}. {rarity_phrase} "
            f"If a functional pathway links those regions, this neuron is "
            f"one of the few cells carrying it."
        )
        cards.append(SurpriseCard(
            root_id=rid,
            name=nd.get("name") or str(rid),
            cell_type=_primary_cell_type(nd) or "",
            super_class=nd.get("super_class") or "",
            nt_type=nd.get("nt_type") or "",
            side=nd.get("side") or "",
            signal="cross_region_bridge",
            signal_label="Cross-region bridge",
            signal_explainer=BRIDGE_EXPLAINER,
            headline=headline,
            description=description,
            detail=(
                f"bridge score = {score:.2f}, rarest pair = {in_np} → {out_np} "
                f"(seen in {n_with}/{total_neurons} neurons, p = {rarest_p:.4f}); "
                f"{len(set(pairs))} bridge pairs total."
            ),
            score=score,
            peer_group="connectome-wide",
            input_neuropils=list(nd.get("input_neuropils") or []),
            output_neuropils=list(nd.get("output_neuropils") or []),
            input_cells=int(nd.get("input_cells") or 0),
            output_cells=int(nd.get("output_cells") or 0),
        ))

    cards.sort(key=lambda c: c.score, reverse=True)
    return cards[:TOP_N_PER_SIGNAL]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class SurpriseFeed:
    hero: Optional[SurpriseCard]
    degree_outliers: List[SurpriseCard]
    cross_region_bridges: List[SurpriseCard]
    version: str

    def as_template_dict(self):
        return {
            "hero": self.hero.as_template_dict() if self.hero else None,
            "degree_outliers": [c.as_template_dict() for c in self.degree_outliers],
            "cross_region_bridges": [c.as_template_dict() for c in self.cross_region_bridges],
            "version": self.version,
        }


_FEED_CACHE: Dict[str, SurpriseFeed] = {}


def compute_feed(neuron_db, version: str) -> SurpriseFeed:
    logger.info(f"Computing surprise feed for version={version} "
                f"(n_cells={neuron_db.num_cells()})")
    degrees = score_degree_outliers(neuron_db)
    bridges = score_cross_region_bridges(neuron_db)

    # Hero = whichever single card has the most striking story.
    candidates = []
    if degrees:
        candidates.append(degrees[0])
    if bridges:
        candidates.append(bridges[0])
    hero = max(candidates, key=lambda c: c.score) if candidates else None

    # Avoid showing the hero twice in the grid below.
    if hero is not None:
        degrees = [c for c in degrees if c.root_id != hero.root_id]
        bridges = [c for c in bridges if c.root_id != hero.root_id]

    return SurpriseFeed(
        hero=hero,
        degree_outliers=degrees,
        cross_region_bridges=bridges,
        version=version,
    )


def get_feed(neuron_db, version: str) -> SurpriseFeed:
    """Cache-aware accessor. First call computes; later calls return cached."""
    cached = _FEED_CACHE.get(version)
    if cached is not None:
        return cached
    feed = compute_feed(neuron_db, version)
    _FEED_CACHE[version] = feed
    return feed


def invalidate_cache(version: Optional[str] = None) -> None:
    if version is None:
        _FEED_CACHE.clear()
    else:
        _FEED_CACHE.pop(version, None)


# ---------------------------------------------------------------------------
# Journey: contextual surprises related to the user's current cell.
# ---------------------------------------------------------------------------

_PARTNER_INDEX_CACHE: Dict[int, Dict[int, set]] = {}


def _partners_from_connected_pairs(neuron_db, rid: int) -> set:
    """Build a one-time partner index from ``connections_.connected_pairs`` and
    cache it on the NeuronDB instance. Returns the partner set for ``rid``.

    The GCS v2 pickle stores all connectivity as a single ``connected_pairs``
    dict keyed by ``(from_rid, to_rid)``. We invert it once into a sparse
    {rid -> set(partners)} index. ~5.6M entries; one pass takes a few seconds.
    """
    db_id = id(neuron_db)
    index = _PARTNER_INDEX_CACHE.get(db_id)
    if index is None:
        cp = getattr(neuron_db.connections_, "connected_pairs", None)
        if not cp:
            _PARTNER_INDEX_CACHE[db_id] = {}
            return set()
        logger.info(f"Building partner index from {len(cp)} connected pairs (one-time)…")
        index = {}
        for (a, b) in cp.keys():
            s = index.get(a)
            if s is None:
                s = set(); index[a] = s
            s.add(b)
            s = index.get(b)
            if s is None:
                s = set(); index[b] = s
            s.add(a)
        _PARTNER_INDEX_CACHE[db_id] = index
        logger.info(f"Partner index built: {len(index)} cells indexed.")
    return index.get(rid, set())


def _score_lookup(feed: SurpriseFeed) -> Dict[int, SurpriseCard]:
    out: Dict[int, SurpriseCard] = {}
    for c in feed.degree_outliers:
        out[c.root_id] = c
    for c in feed.cross_region_bridges:
        # If a neuron qualifies for both signals, prefer the higher-score one.
        if c.root_id not in out or out[c.root_id].score < c.score:
            out[c.root_id] = c
    return out


def compute_related(neuron_db, root_id: int, version: str, n: int = 12) -> List[SurpriseCard]:
    """For the current cell on the user's journey, return up to n related
    surprise cards. We rank candidates by:
      - synaptic partners that are themselves anomalous
      - neurons sharing rare neuropil-pair bridges with this cell
      - NBLAST-similar shape cells with anomalous connectivity
    Each returned card is shaped exactly like a SurpriseCard so the same
    visual contract applies."""

    feed = get_feed(neuron_db, version)
    surprise_index = _score_lookup(feed)
    if not surprise_index:
        return []

    nd_self = neuron_db.neuron_data.get(int(root_id))
    if not nd_self:
        return []

    # 1) Partners — try the public API first, then fall back to a partner
    # index built directly from ``connected_pairs`` (which is the only
    # connectivity attribute carried by the GCS-published v2 pickle).
    candidate_scores: Dict[int, float] = {}
    rid_int = int(root_id)
    partner_ids = set()
    try:
        downstream, upstream = neuron_db.connections_up_down(rid_int)
        partner_ids = set(downstream) | set(upstream)
    except Exception as e:
        logger.info(f"connections_up_down unavailable; falling back to connected_pairs: {e}")
        partner_ids = _partners_from_connected_pairs(neuron_db, rid_int)

    for pid in partner_ids:
        if pid in surprise_index:
            candidate_scores[pid] = candidate_scores.get(pid, 0) + surprise_index[pid].score + 1.0

    # 2) Cells sharing a neuropil-pair with this neuron.
    self_pairs = set()
    for i_np in nd_self.get("input_neuropils") or []:
        for o_np in nd_self.get("output_neuropils") or []:
            if i_np != o_np:
                self_pairs.add((i_np, o_np))

    if self_pairs:
        for rid_other, card in surprise_index.items():
            if rid_other == root_id:
                continue
            other_pairs = set()
            for i_np in card.input_neuropils:
                for o_np in card.output_neuropils:
                    if i_np != o_np:
                        other_pairs.add((i_np, o_np))
            shared = self_pairs & other_pairs
            if shared:
                candidate_scores[rid_other] = candidate_scores.get(rid_other, 0) + 0.5 * len(shared)

    # 3) NBLAST-similar shape cells (already ranked by similarity).
    sim = nd_self.get("similar_cell_scores") or {}
    if isinstance(sim, dict):
        for rid_other, sim_score in sim.items():
            if rid_other in surprise_index and rid_other != root_id:
                candidate_scores[rid_other] = candidate_scores.get(rid_other, 0) + min(sim_score, 5) * 0.4

    # ---- Build a "why is this here" reason for each candidate ----
    def reason_for(rid_other: int) -> str:
        bits = []
        if rid_other in partner_ids:
            bits.append("synapses with this cell")
        # Shared rare neuropil pair?
        nd_other = neuron_db.neuron_data.get(rid_other) or {}
        other_pairs = set()
        for i_np in nd_other.get("input_neuropils") or []:
            for o_np in nd_other.get("output_neuropils") or []:
                if i_np != o_np:
                    other_pairs.add((i_np, o_np))
        shared = self_pairs & other_pairs
        if shared:
            sample_pair = sorted(shared)[0]
            in_p = np_short(sample_pair[0]) or sample_pair[0]
            out_p = np_short(sample_pair[1]) or sample_pair[1]
            bits.append(f"shares the rare {in_p} → {out_p} bridge")
        # NBLAST-similar shape?
        if isinstance(sim, dict) and rid_other in sim:
            bits.append("similar 3D shape")
        if not bits:
            return "Surfaced from the global surprise feed."
        return "Here because it " + " · ".join(bits) + "."

    if not candidate_scores:
        cards = sorted(surprise_index.values(), key=lambda c: c.score, reverse=True)
        out = []
        for c in cards:
            if c.root_id == root_id:
                continue
            c2 = SurpriseCard(**{**asdict(c), "relation_reason": reason_for(c.root_id)})
            out.append(c2)
            if len(out) >= n:
                break
        return out

    ranked = sorted(candidate_scores.items(), key=lambda p: p[1], reverse=True)
    out = []
    for rid_other, _ in ranked:
        if rid_other in surprise_index:
            base = surprise_index[rid_other]
            c2 = SurpriseCard(**{**asdict(base), "relation_reason": reason_for(rid_other)})
            out.append(c2)
        if len(out) >= n:
            break
    return out
