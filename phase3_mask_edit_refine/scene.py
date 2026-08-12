"""Deterministic component and interface graph construction."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import (
    RefineContractError,
    SceneAnchorSegment,
    SceneComponent,
    SceneGraph,
    SceneInterface,
)

CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class SceneAnalysis:
    graph: SceneGraph
    component_masks: dict[str, np.ndarray]
    interface_masks: dict[str, np.ndarray]
    anchor_masks: dict[str, np.ndarray]
    prohibited_region_masks: dict[str, np.ndarray]

    def interfaces_for(
        self, *, source_labels: Iterable[str], target_label: str
    ) -> tuple[SceneInterface, ...]:
        sources = set(source_labels)
        return tuple(
            interface
            for interface in self.graph.interfaces
            if interface.source_label in sources and interface.target_label == target_label
        )


def build_scene_analysis(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    pixel_size_um: float | None = None,
) -> SceneAnalysis:
    """Build directed label-component interfaces from an immutable fine-ID mask."""

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise RefineContractError("source mask must be 2D")
    if not np.issubdtype(arr.dtype, np.integer):
        raise RefineContractError("source mask must contain integer IDs")
    known_ids = set(schema.skip_fine_ids)
    for fine_ids in schema.label_to_fine_ids.values():
        known_ids.update(int(value) for value in fine_ids)
    observed_ids = {int(value) for value in np.unique(arr)}
    unknown_ids = sorted(observed_ids - known_ids)
    if unknown_ids:
        raise RefineContractError(
            f"mask contains IDs unavailable in {schema.reference_profile}: {unknown_ids}"
        )

    components: list[SceneComponent] = []
    component_masks: dict[str, np.ndarray] = {}
    component_index_map = np.zeros(arr.shape, dtype=np.int32)
    label_counts: dict[str, int] = {}
    for label in sorted(schema.readable_labels):
        label_mask = np.isin(arr, schema.resolve_fine_ids(label))
        label_counts[label] = int(np.count_nonzero(label_mask))
        has_explicit_partitions = label in schema.component_partition_fine_ids
        partitions = schema.component_partition_fine_ids.get(
            label, (tuple(schema.resolve_fine_ids(label)),)
        )
        component_counter = 0
        for partition_index, partition_ids in enumerate(partitions, start=1):
            partition_mask = label_mask & np.isin(arr, partition_ids)
            labeled, count = ndimage.label(
                partition_mask, structure=CONNECTIVITY_8
            )
            for local_index in range(1, count + 1):
                component_mask = labeled == local_index
                area = int(np.count_nonzero(component_mask))
                if area == 0:
                    continue
                component_counter += 1
                component_id = (
                    f"cmp:{_slug(label)}:p{partition_index:02d}:"
                    f"{component_counter:04d}"
                    if has_explicit_partitions
                    else f"cmp:{_slug(label)}:{component_counter:04d}"
                )
                bbox = _bbox_xyxy(component_mask)
                touches_border = bool(
                    np.any(component_mask[0, :])
                    or np.any(component_mask[-1, :])
                    or np.any(component_mask[:, 0])
                    or np.any(component_mask[:, -1])
                )
                component_masks[component_id] = component_mask
                components.append(
                    SceneComponent(
                        component_id=component_id,
                        label=label,
                        fine_ids=tuple(partition_ids),
                        area_px=area,
                        bbox_xyxy=bbox,
                        touches_border=touches_border,
                    )
                )
                # One-based dense indices make component adjacency discoverable
                # in O(image pixels) below while retaining fine-label authority.
                component_index_map[component_mask] = len(components)

    interfaces: list[SceneInterface] = []
    interface_masks: dict[str, np.ndarray] = {}
    anchor_segments: list[SceneAnchorSegment] = []
    anchor_masks: dict[str, np.ndarray] = {}
    component_pairs = _adjacent_component_pairs(
        component_index_map,
        components=components,
    )
    for source_index, target_index in component_pairs:
        source = components[source_index - 1]
        target = components[target_index - 1]
        source_mask = component_masks[source.component_id]
        target_mask = component_masks[target.component_id]
        source_side = source_mask & ndimage.binary_dilation(
            target_mask, structure=CONNECTIVITY_8
        )
        contact = int(np.count_nonzero(source_side))
        if contact == 0:
            continue
        interface_base = (
            f"if:{source.component_id.removeprefix('cmp:')}"
            f"->{target.component_id.removeprefix('cmp:')}"
        )
        # A component pair can meet at several biologically distinct arcs
        # (for example an external stromal edge and a gland lumen connected
        # elsewhere in the coarse mask). They must be selectable separately;
        # a free-text anchor description is not an executable constraint.
        segment_labels, segment_count = ndimage.label(
            source_side, structure=CONNECTIVITY_8
        )
        for segment_index in range(1, segment_count + 1):
            segment = segment_labels == segment_index
            segment_contact = int(np.count_nonzero(segment))
            if segment_contact == 0:
                continue
            interface_id = f"{interface_base}:seg:{segment_index:04d}"
            interface_masks[interface_id] = segment
            interface_anchor_ids: list[str] = []
            for anchor_index, anchor in enumerate(
                _partition_interface_anchors(segment), start=1
            ):
                anchor_id = f"{interface_id}:anchor:{anchor_index:04d}"
                anchor_masks[anchor_id] = anchor
                interface_anchor_ids.append(anchor_id)
                anchor_rows, anchor_cols = np.where(anchor)
                anchor_segments.append(
                    SceneAnchorSegment(
                        anchor_segment_id=anchor_id,
                        interface_id=interface_id,
                        display_index=len(anchor_segments) + 1,
                        contact_pixels=int(anchor_rows.size),
                        bbox_xyxy=_bbox_xyxy(anchor),
                        centroid_xy=(
                            float(np.mean(anchor_cols)),
                            float(np.mean(anchor_rows)),
                        ),
                    )
                )
            interfaces.append(
                SceneInterface(
                    interface_id=interface_id,
                    source_component_id=source.component_id,
                    target_component_id=target.component_id,
                    source_label=source.label,
                    target_label=target.label,
                    contact_pixels=segment_contact,
                    bbox_xyxy=_bbox_xyxy(segment),
                    anchor_segment_ids=tuple(interface_anchor_ids),
                )
            )

    warnings: list[str] = []
    if pixel_size_um is None:
        warnings.append("pixel_size_um_missing")
    graph = SceneGraph(
        width=int(arr.shape[1]),
        height=int(arr.shape[0]),
        labels_present={key: value for key, value in label_counts.items() if value > 0},
        components=tuple(components),
        interfaces=tuple(interfaces),
        anchor_segments=tuple(anchor_segments),
        pixel_size_um=pixel_size_um,
        warnings=tuple(warnings),
    )
    return SceneAnalysis(
        graph=graph,
        component_masks=component_masks,
        interface_masks=interface_masks,
        anchor_masks=anchor_masks,
        prohibited_region_masks={},
    )


def _adjacent_component_pairs(
    component_index_map: np.ndarray,
    *,
    components: list[SceneComponent],
) -> tuple[tuple[int, int], ...]:
    """Return directed, cross-label 8-neighbour component adjacencies."""

    index_map = np.asarray(component_index_map, dtype=np.int32)
    height, width = index_map.shape
    pairs: set[tuple[int, int]] = set()
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            source_rows = slice(max(0, -delta_row), min(height, height - delta_row))
            source_cols = slice(max(0, -delta_col), min(width, width - delta_col))
            target_rows = slice(max(0, delta_row), min(height, height + delta_row))
            target_cols = slice(max(0, delta_col), min(width, width + delta_col))
            source = index_map[source_rows, source_cols]
            target = index_map[target_rows, target_cols]
            valid = (source > 0) & (target > 0) & (source != target)
            if not np.any(valid):
                continue
            encoded = np.unique(
                source[valid].astype(np.int64) * (len(components) + 1)
                + target[valid].astype(np.int64)
            )
            for value in encoded.tolist():
                source_index, target_index = divmod(value, len(components) + 1)
                if components[source_index - 1].label != components[target_index - 1].label:
                    pairs.add((int(source_index), int(target_index)))
    return tuple(sorted(pairs))


def _partition_interface_anchors(interface_mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Partition one connected directed interface into selectable spatial sub-arcs.

    The partition is deterministic, exhaustive and contains at most eight
    anchors.  Farthest-point seeds and graph-geodesic Voronoi assignment keep
    the regions local along a curved/branched interface; PCA bins can represent
    distant arcs with one ID or explode into dozens of IDs after splitting.
    """

    coordinates = np.argwhere(interface_mask)
    count = int(coordinates.shape[0])
    if count == 0:
        return ()
    bin_count = int(np.clip(np.ceil(count / 160.0), 1, 8))
    if bin_count == 1:
        return (np.asarray(interface_mask, dtype=bool),)
    seeds = [tuple(int(value) for value in coordinates[0])]
    distance_maps = [_interface_geodesic_distances(interface_mask, seeds[0])]
    minimum_distance = np.array(distance_maps[0], copy=True)
    minimum_distance[~interface_mask] = -1
    for _ in range(1, bin_count):
        farthest_flat = int(np.argmax(minimum_distance))
        farthest = tuple(int(value) for value in np.unravel_index(farthest_flat, interface_mask.shape))
        seeds.append(farthest)
        distance = _interface_geodesic_distances(interface_mask, farthest)
        distance_maps.append(distance)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[~interface_mask] = -1
    assignment = np.argmin(np.stack(distance_maps, axis=0), axis=0)
    anchors = tuple(
        interface_mask & (assignment == seed_index)
        for seed_index in range(len(seeds))
    )
    if any(not np.any(anchor) for anchor in anchors):
        raise RuntimeError("geodesic anchor partition produced an empty region")
    if not np.array_equal(np.logical_or.reduce(anchors), interface_mask):
        raise RuntimeError("geodesic anchor partition is not exhaustive")
    return anchors


def _interface_geodesic_distances(
    interface_mask: np.ndarray,
    seed: tuple[int, int],
) -> np.ndarray:
    """Unit-weight 8-neighbour shortest paths constrained to one interface."""

    interface = np.asarray(interface_mask, dtype=bool)
    unreachable = np.iinfo(np.int32).max
    distances = np.full(interface.shape, unreachable, dtype=np.int32)
    distances[seed] = 0
    queue: deque[tuple[int, int]] = deque([seed])
    height, width = interface.shape
    while queue:
        row, col = queue.popleft()
        next_distance = distances[row, col] + 1
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                next_row = row + delta_row
                next_col = col + delta_col
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and interface[next_row, next_col]
                    and next_distance < distances[next_row, next_col]
                ):
                    distances[next_row, next_col] = next_distance
                    queue.append((next_row, next_col))
    if np.any(distances[interface] == unreachable):
        raise RuntimeError("interface segment is disconnected before anchor partition")
    return distances


def profile_signature_metrics(
    mask: np.ndarray, *, schema: MaskProfileSchema
) -> dict[str, object]:
    arr = np.asarray(mask)
    observed = sorted(int(value) for value in np.unique(arr))
    allowed = sorted(
        set(schema.skip_fine_ids).union(
            *[set(values) for values in schema.label_to_fine_ids.values()]
        )
    )
    background = np.isin(arr, tuple(schema.skip_fine_ids))
    labeled, component_count = ndimage.label(background, structure=CONNECTIVITY_8)
    border_ids = {
        int(value)
        for value in np.unique(
            np.concatenate([labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]])
        )
    } - {0}
    border_pixels = int(np.count_nonzero(np.isin(labeled, tuple(border_ids))))
    background_pixels = int(np.count_nonzero(background))
    return {
        "reference_profile": schema.reference_profile,
        "observed_ids": observed,
        "allowed_ids": allowed,
        "unknown_ids": sorted(set(observed) - set(allowed)),
        "background_fraction": float(np.mean(background)),
        "background_components_per_mpx": float(
            component_count / max(arr.size / 1_000_000.0, 1e-9)
        ),
        "background_border_connected_fraction": float(
            border_pixels / max(background_pixels, 1)
        ),
        "internal_background_fraction": float(
            max(0, background_pixels - border_pixels) / max(arr.size, 1)
        ),
    }


def _bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return (0, 0, 0, 0)
    return (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "_")
