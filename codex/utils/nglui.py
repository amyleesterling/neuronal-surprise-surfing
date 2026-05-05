import random
import urllib.parse
from nglui import statebuilder
import json

from codex.data.brain_regions import REGIONS, COLORS
from codex.data.versions import (
    DEFAULT_DATA_SNAPSHOT_VERSION,
    DATA_SNAPSHOT_VERSION_DESCRIPTIONS,
)

from codex import logger

NGL_FLAT_BASE_URL = "https://ngl.cave-explorer.org"


def url_for_root_ids(
    root_ids, version, point_to="ngl", position=None, show_side_panel=None
):
    if version not in DATA_SNAPSHOT_VERSION_DESCRIPTIONS:
        logger.error(
            f"Invalid version '{version}' passed to 'url_for_root_ids'. Falling back to default."
        )
        version = DEFAULT_DATA_SNAPSHOT_VERSION
    if point_to in ["flywire_prod", "flywire_public"]:
        img_layer = statebuilder.ImageLayerConfig(
            name="EM",
            source="precomputed://gs://microns-seunglab/drosophila_v0/alignment/vector_fixer30_faster_v01/v4/image_stitch_v02",
        )

        seg_layer_name = (
            "Production segmentation"
            if point_to == "flywire_prod"
            else "Public segmentation"
        )
        seg_layer_source = (
            "graphene://https://prodv1.flywire-daf.com/segmentation/table/fly_v31"
            if point_to == "flywire_prod"
            else "graphene://https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public"
        )

        seg_layer = statebuilder.SegmentationLayerConfig(
            name=seg_layer_name,
            source=seg_layer_source,
            fixed_ids=root_ids,
        )

        view_options = {
            "layout": "xy-3d",
            "show_slices": False,
            "zoom_3d": 2500,
            "zoom_image": 50,
        }

        if position is not None:
            view_options["position"] = position

        sb = statebuilder.StateBuilder(
            layers=[img_layer, seg_layer],
            resolution=[4, 4, 40],
            view_kws=view_options,
        )

        config = sb.render_state(return_as="dict")
        config["selectedLayer"] = {
            "layer": seg_layer_name,
            "visible": True,
        }
        config["jsonStateServer"] = "https://globalv1.flywire-daf.com/nglstate/post"

        return f"https://ngl.flywire.ai/#!{urllib.parse.quote(json.dumps(config))}"
    else:
        return url_for_cells(
            segment_ids=root_ids, data_version=version, show_side_panel=show_side_panel
        )


def url_for_random_sample(root_ids, version, sample_size=50):
    # make the random subset selections deterministic across executions
    random.seed(420)
    if len(root_ids) > sample_size:
        # make a sorted sample to preserve original order
        root_ids = [
            root_ids[i]
            for i in sorted(random.sample(range(len(root_ids)), sample_size))
        ]
    return url_for_root_ids(root_ids, version=version)


def url_for_cells(segment_ids, data_version, show_side_panel=None):
    if show_side_panel is None:
        show_side_panel = len(segment_ids) > 1
    else:
        show_side_panel = bool(show_side_panel)

    if data_version not in DATA_SNAPSHOT_VERSION_DESCRIPTIONS:
        logger.error(
            f"Invalid version '{data_version}' passed to 'url_for_cells'. Falling back to default."
        )
        data_version = DEFAULT_DATA_SNAPSHOT_VERSION

    config = {
        "dimensions": {"x": [1.6e-8, "m"], "y": [1.6e-8, "m"], "z": [4e-8, "m"]},
        "projectionScale": 30000,
        "layers": [
            {
                "type": "image",
                "source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
                "tab": "source",
                "name": "EM",
            },
            {
                "source": "precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3",
                "type": "segmentation",
                "objectAlpha": 0.05,
                "hideSegmentZero": False,
                "segments": ["1"],
                "segmentColors": {"1": "#b5b5b5"},
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "brain_mesh_v3",
            },
            {
                "type": "segmentation",
                "source": f"precomputed://gs://flywire_v141_m{data_version}",
                "tab": "segments",
                "segments": [
                    str(sid) for sid in segment_ids
                ],  # BEWARE: JSON can't handle big ints
                "name": f"flywire_v141_m{data_version}",
            },
        ],
        "showSlices": False,
        "perspectiveViewBackgroundColor": "#ffffff",
        "showDefaultAnnotations": False,
        "selectedLayer": {
            "visible": show_side_panel,
            "layer": f"flywire_v141_m{data_version}",
        },
        "layout": "3d",
    }

    return f"{NGL_FLAT_BASE_URL}/#!{urllib.parse.quote(json.dumps(config))}"


def _parse_position_nm(pos_field):
    """neuron_data['position'] is a list of strings like '[710944 262716 205680]'
    in nm. Returns [x, y, z] in nm, or None."""
    if not pos_field:
        return None
    raw = pos_field[0] if isinstance(pos_field, list) else pos_field
    if not isinstance(raw, str):
        return None
    s = raw.strip().lstrip("[").rstrip("]")
    try:
        parts = [int(p) for p in s.split() if p]
        if len(parts) == 3:
            return parts
    except (ValueError, IndexError):
        pass
    return None


def _nm_to_voxel(pos_nm, voxel_nm=(16, 16, 40)):
    return [pos_nm[i] / voxel_nm[i] for i in range(3)]


def _journey_link_annotations(neuron_db, journey_ids):
    """For each pair of cells in the journey that have a synaptic connection
    in either direction, build a neuroglancer line annotation between their
    soma positions. Returns a list of annotation dicts."""
    if neuron_db is None or len(journey_ids) < 2:
        return []
    cp = getattr(getattr(neuron_db, "connections_", None), "connected_pairs", None)

    positions = {}
    for rid in journey_ids:
        nd = neuron_db.neuron_data.get(rid)
        if not nd:
            continue
        pos_nm = _parse_position_nm(nd.get("position"))
        if pos_nm:
            positions[rid] = _nm_to_voxel(pos_nm)

    annotations = []
    seen = set()
    for i, a in enumerate(journey_ids):
        for b in journey_ids[i + 1:]:
            if a == b or a not in positions or b not in positions:
                continue
            forward = cp is not None and (a, b) in cp
            reverse = cp is not None and (b, a) in cp
            if not (forward or reverse):
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            if forward and reverse:
                desc = f"{a} ⇄ {b} (reciprocal)"
            elif forward:
                desc = f"{a} → {b}"
            else:
                desc = f"{b} → {a}"
            annotations.append({
                "type": "line",
                "id": f"jlink_{a}_{b}",
                "pointA": positions[a],
                "pointB": positions[b],
                "description": desc,
            })
    return annotations


def url_for_journey(current_id, trail_ids, data_version, neuron_db=None):
    """Render the current cell brightly, with the visited trail dimmer and
    desaturated. Lets the user literally see the path they've walked.

    current_id: int — the cell currently in focus
    trail_ids:  list[int] — previously visited cells (most recent last),
                excluding the current id
    neuron_db:  optional NeuronDB; if provided, every pair of journey cells
                that has a synaptic connection gets drawn as a line annotation.
    """
    if data_version not in DATA_SNAPSHOT_VERSION_DESCRIPTIONS:
        data_version = DEFAULT_DATA_SNAPSHOT_VERSION

    current_id = int(current_id)
    trail_ids = [int(x) for x in trail_ids if int(x) != current_id]
    all_ids = trail_ids + [current_id]

    # Bright cyan for the current cell, fading magenta-ish for the trail.
    current_color = "#7be4ff"
    trail_palette = ["#3a4f7a", "#5a4a7d", "#7a4a76", "#8a5a6e", "#9a6a66"]

    segment_colors = {str(current_id): current_color}
    n_trail = len(trail_ids)
    for i, rid in enumerate(trail_ids):
        # Walk backwards through the palette so the most recent past stop is brightest.
        color_idx = max(0, len(trail_palette) - 1 - (n_trail - 1 - i))
        segment_colors[str(rid)] = trail_palette[color_idx]

    config = {
        "dimensions": {"x": [1.6e-8, "m"], "y": [1.6e-8, "m"], "z": [4e-8, "m"]},
        "projectionScale": 30000,
        "layers": [
            {
                "source": "precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3",
                "type": "segmentation",
                "objectAlpha": 0.04,
                "hideSegmentZero": False,
                "segments": ["1"],
                "segmentColors": {"1": "#1c2742"},
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "brain_mesh_v3",
            },
            {
                "type": "segmentation",
                "source": f"precomputed://gs://flywire_v141_m{data_version}",
                "tab": "segments",
                "segments": [str(sid) for sid in all_ids],
                "segmentColors": segment_colors,
                "name": "journey",
            },
        ],
        "showSlices": False,
        "perspectiveViewBackgroundColor": "#04060c",
        "showDefaultAnnotations": False,
        "selectedLayer": {"visible": False, "layer": "journey"},
        "layout": "3d",
    }

    # Connection lines between journey cells: a soft visual showing which pairs
    # actually synapse on each other. Annotation source is inline JSON so no
    # external service is needed.
    link_annotations = _journey_link_annotations(neuron_db, all_ids)
    if link_annotations:
        config["layers"].append({
            "type": "annotation",
            "name": "synaptic_links",
            "annotations": link_annotations,
            "annotationColor": "#ffd84a",
            "shader": "void main() { setColor(prop_color()); setEndpointMarkerSize(6, 6); }",
        })

    return f"{NGL_FLAT_BASE_URL}/#!{urllib.parse.quote(json.dumps(config))}"


def url_for_neuropils(segment_ids=None):
    if segment_ids:
        # exclude "dummy" neuropils, e.g. unassigned, which by convention have negative ids
        segment_ids = [s for s in segment_ids if s >= 0]
    config = {
        "layers": [
            {
                "source": "precomputed://gs://flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3",
                "type": "segmentation",
                "objectAlpha": 0.1,
                "hideSegmentZero": False,
                "segments": ["1"],
                "segmentColors": {"1": "#b5b5b5"},
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "brain_mesh_v3",
            },
            {
                "type": "segmentation",
                "mesh": "precomputed://gs://flywire_neuropil_meshes/neuropils/neuropil_mesh_v141_v3",
                "objectAlpha": 1.0,  # workaround for broken transparency on iOS: https://github.com/google/neuroglancer/issues/471
                "tab": "source",
                "segments": segment_ids,
                "segmentColors": {
                    # exclude "dummy" neuropil colors, e.g. unassigned, which by convention have negative ids
                    seg_id: COLORS[key]
                    for key, (seg_id, _) in REGIONS.items()
                    if seg_id >= 0
                },
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "neuropil-regions-surface",
            },
        ],
        "navigation": {
            "pose": {
                "position": {
                    "voxelSize": [4, 4, 40],
                    "voxelCoordinates": [132000, 55390, 512],
                }
            },
            "zoomFactor": 40.875984234132744,
        },
        "showAxisLines": False,
        "perspectiveViewBackgroundColor": "#ffffff",
        "perspectiveZoom": 4000,
        "showSlices": False,
        "gpuMemoryLimit": 2000000000,
        "showDefaultAnnotations": False,
        "selectedLayer": {"layer": "neuropil-regions-surface", "visible": False},
        "layout": "3d",
    }

    return f"{NGL_FLAT_BASE_URL}/#!{urllib.parse.quote(json.dumps(config))}"
