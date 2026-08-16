"""Build a deterministic tissue--cell scene graph without model inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.scene import SceneAnalysis, build_scene_analysis

from .models import (
    CellGraphEdge,
    CellSceneGraph,
    JointContractError,
    NucleusInstance,
    PopulationGraph,
    PopulationZone,
)
from .nuclei import (
    instance_bbox,
    instance_centroid,
    iter_instances,
    load_native_instances,
    normalize_nuclei_mask,
    touches_border,
)
from .reference_shapes import ReferenceShapeAuthority

STRUCTURAL_COMPARTMENT_LABELS = frozenset(
    {"Tumor", "Stroma", "Normal epithelium", "Necrosis", "Other tissue"}
)
POPULATION_OVERLAY_LABELS = frozenset({"Immune infiltrate"})
CELL_POPULATION_ROLES = {
    1: "neoplastic_population",
    2: "inflammatory_population",
    3: "connective_tissue_population",
    4: "dead_or_dying_population",
    5: "epithelial_population",
}


@dataclass(frozen=True)
class JointSceneAnalysis:
    tissue: SceneAnalysis
    cells: CellSceneGraph
    source_nuclei: np.ndarray
    instance_masks: dict[str, np.ndarray]
    auxiliary_structure_masks: dict[str, np.ndarray]
    population: PopulationGraph
    population_zone_masks: dict[str, np.ndarray]
    structural_hierarchy: dict[str, object]
    structural_unit_masks: dict[str, np.ndarray]
    reference_shape_authority: ReferenceShapeAuthority | None = None

    def to_metadata(self) -> dict:
        return {
            "tissue": self.tissue.graph.to_metadata(),
            "cells": self.cells.to_metadata(),
            "population": self.population.to_metadata(),
            "auxiliary_structures": {
                key: {"pixels": int(np.count_nonzero(value))}
                for key, value in sorted(self.auxiliary_structure_masks.items())
            },
            "structural_hierarchy": self.structural_hierarchy,
            "reference_shape_authority": (
                self.reference_shape_authority.to_metadata()
                if self.reference_shape_authority is not None
                else None
            ),
        }


def build_joint_scene_analysis(
    tissue_mask: np.ndarray,
    nuclei_mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    pixel_size_um: float | None,
    nuclei_instances_path: str | None = None,
    auxiliary_structure_paths: dict[str, str] | None = None,
    auxiliary_structure_provenance: dict[str, dict] | None = None,
    reference_shape_authority: ReferenceShapeAuthority | None = None,
) -> JointSceneAnalysis:
    tissue = np.asarray(tissue_mask)
    nuclei = normalize_nuclei_mask(nuclei_mask)
    if tissue.ndim != 2 or tissue.shape != nuclei.shape:
        raise JointContractError("tissue and nuclei masks must be aligned 2-D arrays")
    tissue_scene = build_scene_analysis(
        tissue,
        schema=schema,
        pixel_size_um=pixel_size_um,
    )
    auxiliary_masks = {}
    for structure_id, path in sorted((auxiliary_structure_paths or {}).items()):
        current = load_id_mask(path)
        if current.shape != tissue.shape:
            raise JointContractError(
                f"auxiliary structure {structure_id!r} is not aligned to the case"
            )
        values = np.asarray(current)
        if structure_id == "native_gland_instance_map":
            # Native GLaS gland IDs are execution authority.  Keeping only a
            # boolean foreground would merge distinct glands that share a
            # raster edge and would erase the deterministic instance boundary
            # supplied by the dataset/preprocessor.
            region = values.copy()
        else:
            region = values != 0
        # An empty, provenance-bound protection map is a valid negative
        # observation: the producer ran but found no enclosed native space.
        auxiliary_masks[structure_id] = region
    instances: list[NucleusInstance] = []
    masks: dict[str, np.ndarray] = {}
    centers: list[tuple[float, float]] = []
    counts = {class_id: 0 for class_id in range(1, 6)}
    height, width = tissue.shape
    component_id_map, component_ids = _component_id_map(tissue_scene)
    interface_tree, interface_point_ids = _interface_point_index(tissue_scene)
    native_instances = (
        load_native_instances(
            nuclei_instances_path,
            shape=tissue.shape,
            semantic_mask=nuclei,
        )
        if nuclei_instances_path
        else None
    )
    source_instances = native_instances if native_instances is not None else tuple(iter_instances(nuclei))
    for instance_id, class_id, component in source_instances:
        cx, cy = instance_centroid(component)
        row = int(np.clip(round(cy), 0, height - 1))
        col = int(np.clip(round(cx), 0, width - 1))
        component_index = int(component_id_map[row, col])
        tissue_component_id = (
            component_ids[component_index - 1] if component_index > 0 else None
        )
        nearest_interface_id = None
        interface_distance = None
        if interface_tree is not None:
            interface_distance, nearest_index = interface_tree.query((cy, cx), k=1)
            nearest_interface_id = interface_point_ids[int(nearest_index)]
        shape_metrics = _shape_metrics(component)
        instances.append(
            NucleusInstance(
                instance_id=instance_id,
                class_id=class_id,
                area_px=int(component.sum()),
                bbox_xyxy=instance_bbox(component),
                centroid_xy=(cx, cy),
                tissue_fine_id=int(tissue[row, col]),
                touches_border=touches_border(component),
                source=(
                    (
                        "instance_json_cellvit_seed"
                        if instance_id.startswith("native-raster-cellvit-")
                        else (
                            "instance_json_semantic_unseeded"
                            if instance_id.startswith(
                                "native-raster-semantic-unseeded-"
                            )
                            else (
                                "instance_json_semantic_seeded_residual"
                                if instance_id.startswith(
                                    "native-raster-semantic-residual-"
                                )
                                else (
                                    "instance_json_semantic_fallback"
                                    if instance_id.startswith(
                                        "native-raster-semantic-fallback-"
                                    )
                                    else "instance_json"
                                )
                            )
                        )
                    )
                    if native_instances is not None
                    else "semantic_distance_watershed"
                ),
                tissue_component_id=tissue_component_id,
                nearest_interface_id=nearest_interface_id,
                distance_to_interface_px=(
                    float(interface_distance)
                    if interface_distance is not None
                    else None
                ),
                perimeter_px=shape_metrics["perimeter_px"],
                solidity=shape_metrics["solidity"],
                eccentricity=shape_metrics["eccentricity"],
                completeness_status=(
                    "patch_boundary_censored"
                    if touches_border(component)
                    else "complete"
                ),
            )
        )
        masks[instance_id] = component
        centers.append((cy, cx))
        counts[class_id] += 1
    instances = _mark_instance_quality(instances)
    mean_nnd: float | None = None
    if len(centers) >= 2:
        distances, _ = cKDTree(np.asarray(centers, dtype=float)).query(
            np.asarray(centers, dtype=float), k=2
        )
        mean_nnd = float(np.mean(distances[:, 1]))
    warnings: list[str] = []
    if native_instances is None:
        warnings.append(
            "instance identity reconstructed by per-class distance watershed"
        )
    if any(item.touches_border for item in instances):
        warnings.append("border-touching nuclei are protected and cannot be resampled")
    edges = _cell_edges(instances)
    cell_graph = CellSceneGraph(
        width=width,
        height=height,
        instances=tuple(instances),
        class_counts=counts,
        mean_nearest_neighbor_px=mean_nnd,
        observation_quality=(
            "native_instance"
            if native_instances is not None
            else "semantic_distance_watershed"
        ),
        warnings=tuple(warnings),
        edges=edges,
        interface_relation_count=sum(
            item.nearest_interface_id is not None for item in instances
        ),
        merged_suspect_instance_ids=tuple(
            item.instance_id
            for item in instances
            if "merged_suspect" in item.quality_flags
        ),
        border_censored_instance_ids=tuple(
            item.instance_id for item in instances if item.touches_border
        ),
    )
    population, population_masks = _build_population_graph(
        tissue_scene=tissue_scene,
        instances=tuple(instances),
        observation_quality=cell_graph.observation_quality,
        shape=tissue.shape,
    )
    hierarchy, structural_unit_masks = _bind_structural_hierarchy(
        tissue_scene,
        tissue,
        auxiliary_structure_provenance or {},
        population=population,
        instances=tuple(instances),
    )
    return JointSceneAnalysis(
        tissue_scene,
        cell_graph,
        nuclei,
        masks,
        auxiliary_masks,
        population,
        population_masks,
        hierarchy,
        structural_unit_masks,
        reference_shape_authority,
    )


def _bind_structural_hierarchy(
    tissue_scene: SceneAnalysis,
    tissue_mask: np.ndarray,
    provenance_by_structure: dict[str, dict],
    *,
    population: PopulationGraph,
    instances: tuple[NucleusInstance, ...],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Bind producer units to tissue components without inventing histology."""

    units_by_id: dict[str, dict] = {}
    anonymous_units: list[dict] = []
    relations = []
    unit_masks: dict[str, np.ndarray] = {}
    for structure_id, provenance in sorted(provenance_by_structure.items()):
        if not isinstance(provenance, dict):
            continue
        for item in provenance.get("structure_units", ()):
            if not isinstance(item, dict):
                continue
            parent_id = None
            unit_id = str(item.get("unit_id") or "")
            fine_id = item.get("fine_id")
            unit_mask = _recover_structural_unit_mask(
                tissue_mask,
                unit_id=unit_id,
                fine_id=fine_id,
                expected_digest=item.get("component_sha256"),
            )
            if unit_id and unit_mask is not None:
                previous_mask = unit_masks.get(unit_id)
                if previous_mask is not None and not np.array_equal(
                    previous_mask, unit_mask
                ):
                    raise JointContractError(
                        "conflicting structural-unit masks for "
                        f"{unit_id}"
                    )
                unit_masks[unit_id] = unit_mask
                overlaps = [
                    (
                        int(np.count_nonzero(unit_mask & mask)),
                        component_id,
                    )
                    for component_id, mask in tissue_scene.component_masks.items()
                ]
                if overlaps:
                    count, candidate_parent = max(overlaps)
                    if count > 0:
                        parent_id = candidate_parent
            record = {
                **item,
                "auxiliary_structure_id": structure_id,
                "auxiliary_structure_ids": [structure_id],
                "parent_tissue_component_id": parent_id,
            }
            if unit_id:
                existing = units_by_id.get(unit_id)
                if existing is None:
                    units_by_id[unit_id] = record
                else:
                    for key in (
                        "fine_id",
                        "component_sha256",
                        "parent_tissue_component_id",
                    ):
                        if existing.get(key) != record.get(key):
                            raise JointContractError(
                                "conflicting structural-unit observations for "
                                f"{unit_id}: {key}"
                            )
                    existing["auxiliary_structure_ids"] = sorted(
                        {
                            *existing.get("auxiliary_structure_ids", ()),
                            structure_id,
                        }
                    )
                    existing["enclosed_space_ids"] = sorted(
                        {
                            *existing.get("enclosed_space_ids", ()),
                            *record.get("enclosed_space_ids", ()),
                        }
                    )
            else:
                anonymous_units.append(record)
            if parent_id is not None:
                relations.append(
                    {
                        "source_id": item.get("unit_id"),
                        "relation": "member_of",
                        "target_id": parent_id,
                    }
                )
        relations.extend(provenance.get("hierarchy_relations", ()))
    units = [
        *(units_by_id[unit_id] for unit_id in sorted(units_by_id)),
        *anonymous_units,
    ]
    compartments = [
        {
            "component_id": item.component_id,
            "label": item.label,
            "fine_ids": list(item.fine_ids),
            "area_px": item.area_px,
            "hierarchy_role": (
                "structural_compartment"
                if item.label in STRUCTURAL_COMPARTMENT_LABELS
                else (
                    "cellular_population_overlay"
                    if item.label in POPULATION_OVERLAY_LABELS
                    else "annotation_only_compartment"
                )
            ),
        }
        for item in tissue_scene.graph.components
    ]
    interfaces = [
        {
            "interface_id": item.interface_id,
            "source_component_id": item.source_component_id,
            "target_component_id": item.target_component_id,
            "source_label": item.source_label,
            "target_label": item.target_label,
            "contact_pixels": item.contact_pixels,
        }
        for item in tissue_scene.graph.interfaces
    ]
    population_zones = [
        {
            "zone_id": item.zone_id,
            "zone_kind": item.zone_kind,
            "tissue_component_id": item.tissue_component_id,
            "interface_id": item.interface_id,
            "side": item.side,
            "area_px": item.area_px,
            "nucleus_count": item.nucleus_count,
            "class_counts": item.class_counts,
        }
        for item in population.zones
    ]
    nucleus_instances = [
        {
            "instance_id": item.instance_id,
            "class_id": item.class_id,
            "population_role": CELL_POPULATION_ROLES.get(
                item.class_id, "unresolved_population"
            ),
            "tissue_component_id": item.tissue_component_id,
            "nearest_interface_id": item.nearest_interface_id,
            "completeness_status": item.completeness_status,
        }
        for item in instances
    ]
    cellular_populations = [
        {
            "class_id": class_id,
            "population_role": CELL_POPULATION_ROLES.get(
                class_id, "unresolved_population"
            ),
            "instance_count": sum(
                item.class_id == class_id for item in instances
            ),
            "observation_authority": "cell_observation_profile_class_id",
        }
        for class_id in sorted({item.class_id for item in instances})
    ]
    relations.extend(
        {
            "source_id": item.interface_id,
            "relation": "connects",
            "target_id": component_id,
        }
        for item in tissue_scene.graph.interfaces
        for component_id in (
            item.source_component_id,
            item.target_component_id,
        )
    )
    relations.extend(
        {
            "source_id": item.zone_id,
            "relation": (
                "samples_interface"
                if item.interface_id is not None
                else "population_of"
            ),
            "target_id": item.interface_id or item.tissue_component_id,
        }
        for item in population.zones
        if item.interface_id is not None or item.tissue_component_id is not None
    )
    relations.extend(
        {
            "source_id": item.instance_id,
            "relation": "member_of_tissue_component",
            "target_id": item.tissue_component_id,
        }
        for item in instances
        if item.tissue_component_id is not None
    )
    relations.extend(
        {
            "source_id": item.instance_id,
            "relation": "member_of_cellular_population",
            "target_id": CELL_POPULATION_ROLES.get(
                item.class_id, "unresolved_population"
            ),
        }
        for item in instances
    )
    return {
        "schema_version": "joint-structural-hierarchy-v2",
        "levels": [
            "structural_compartment",
            "cellular_population",
            "morphology",
        ],
        "execution_semantics": {
            "structural_compartment": (
                "owns tissue label transitions and component topology"
            ),
            "cellular_population": (
                "owns complete-instance retain, resample, remove and add actions; "
                "a population is not automatically a structural barrier"
            ),
            "morphology": (
                "owns producer-bound gland, lumen, pattern and architecture units "
                "that constrain both tissue and cell execution"
            ),
            "single_label_annotation_limit": (
                "a dataset tissue label that encodes a population overlay remains "
                "pixel-protected unless the primitive explicitly authorizes that "
                "label transition"
            ),
        },
        "tissue_components": compartments,
        "cellular_populations": cellular_populations,
        "structure_units": units,
        "interfaces": interfaces,
        "population_zones": population_zones,
        "nucleus_instances": nucleus_instances,
        "relations": relations,
        "observation_policy": "semantic_producer_only_no_H&E_invention",
    }, unit_masks


def _recover_structural_unit_mask(
    tissue: np.ndarray,
    *,
    unit_id: str,
    fine_id,
    expected_digest,
) -> np.ndarray | None:
    if not unit_id or not isinstance(fine_id, int):
        return None
    try:
        component_index = int(unit_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None
    labeled, count = ndimage.label(
        np.asarray(tissue) == int(fine_id),
        structure=np.ones((3, 3), dtype=bool),
    )
    if not 1 <= component_index <= count:
        return None
    component = labeled == component_index
    digest = hashlib.sha256(
        np.packbits(component.astype(np.uint8), axis=None).tobytes()
    ).hexdigest()
    if expected_digest and digest != expected_digest:
        raise JointContractError(
            f"structural unit {unit_id!r} digest does not match producer provenance"
        )
    return component


def _component_id_map(scene: SceneAnalysis) -> tuple[np.ndarray, tuple[str, ...]]:
    result = np.zeros((scene.graph.height, scene.graph.width), dtype=np.int32)
    ids = tuple(item.component_id for item in scene.graph.components)
    for index, component_id in enumerate(ids, start=1):
        result[scene.component_masks[component_id]] = index
    return result, ids


def _interface_point_index(
    scene: SceneAnalysis,
) -> tuple[cKDTree | None, tuple[str, ...]]:
    points: list[tuple[int, int]] = []
    ids: list[str] = []
    for interface in scene.graph.interfaces:
        rows, cols = np.nonzero(scene.interface_masks[interface.interface_id])
        points.extend((int(row), int(col)) for row, col in zip(rows, cols))
        ids.extend([interface.interface_id] * len(rows))
    if not points:
        return None, ()
    return cKDTree(np.asarray(points, dtype=float)), tuple(ids)


def _shape_metrics(component: np.ndarray) -> dict[str, float | None]:
    labeled = np.asarray(component, dtype=np.uint8)
    props = regionprops(labeled)
    if not props:
        return {"perimeter_px": None, "solidity": None, "eccentricity": None}
    item = props[0]
    return {
        "perimeter_px": float(item.perimeter),
        "solidity": float(item.solidity),
        "eccentricity": float(item.eccentricity),
    }


def _mark_instance_quality(
    instances: list[NucleusInstance],
) -> list[NucleusInstance]:
    all_areas_by_class: dict[int, list[float]] = {}
    seed_areas_by_class: dict[int, list[float]] = {}
    for item in instances:
        if not item.touches_border:
            all_areas_by_class.setdefault(item.class_id, []).append(
                float(item.area_px)
            )
            if item.source == "instance_json_cellvit_seed":
                seed_areas_by_class.setdefault(item.class_id, []).append(
                    float(item.area_px)
                )
    limits: dict[int, float] = {}
    for class_id, all_values in all_areas_by_class.items():
        # Exact semantic coverage can leave many tiny pieces around a clipped
        # CellViT seed. Those pieces are provenance records, not an independent
        # morphology population, and must not make every true seed look merged.
        values = seed_areas_by_class.get(class_id) or all_values
        array = np.asarray(values, dtype=float)
        if array.size < 4:
            limits[class_id] = float(np.median(array) * 3.0) if array.size else np.inf
            continue
        q1, q3 = np.quantile(array, (0.25, 0.75))
        limits[class_id] = float(max(q3 + 3.0 * (q3 - q1), np.median(array) * 3.0))
    result = []
    for item in instances:
        flags = list(item.quality_flags)
        if item.touches_border:
            flags.append("patch_boundary_censored")
        if item.area_px > limits.get(item.class_id, np.inf):
            flags.append("merged_suspect")
        if item.solidity is not None and item.solidity < 0.45:
            flags.append("irregular_or_fragmented_shape")
        result.append(replace(item, quality_flags=tuple(sorted(set(flags)))))
    return result


def _cell_edges(instances: list[NucleusInstance], k: int = 6) -> tuple[CellGraphEdge, ...]:
    if len(instances) < 2:
        return ()
    points = np.asarray(
        [(item.centroid_xy[1], item.centroid_xy[0]) for item in instances],
        dtype=float,
    )
    query_k = min(k + 1, len(instances))
    distances, neighbors = cKDTree(points).query(points, k=query_k)
    edges: dict[tuple[int, int], CellGraphEdge] = {}
    for source_index in range(len(instances)):
        for distance, target_index in zip(
            np.atleast_1d(distances[source_index])[1:],
            np.atleast_1d(neighbors[source_index])[1:],
        ):
            left, right = sorted((source_index, int(target_index)))
            key = (left, right)
            if key in edges:
                continue
            source, target = instances[left], instances[right]
            edges[key] = CellGraphEdge(
                source_instance_id=source.instance_id,
                target_instance_id=target.instance_id,
                relation="knn",
                distance_px=float(distance),
                same_class=source.class_id == target.class_id,
                same_tissue_component=(
                    source.tissue_component_id is not None
                    and source.tissue_component_id == target.tissue_component_id
                ),
            )
    return tuple(edges[key] for key in sorted(edges))


def _build_population_graph(
    *,
    tissue_scene: SceneAnalysis,
    instances: tuple[NucleusInstance, ...],
    observation_quality: str,
    shape: tuple[int, int],
) -> tuple[PopulationGraph, dict[str, np.ndarray]]:
    native_seed_areas = [
        item.area_px
        for item in instances
        if item.source == "instance_json_cellvit_seed"
        and not item.touches_border
        and "merged_suspect" not in item.quality_flags
    ]
    complete_areas = native_seed_areas or [
        item.area_px
        for item in instances
        if not item.touches_border and "merged_suspect" not in item.quality_flags
    ]
    median_area = float(np.median(complete_areas)) if complete_areas else None
    nominal_diameter = (
        float(2.0 * np.sqrt(median_area / np.pi))
        if median_area is not None and median_area > 0
        else 8.0
    )
    band_width = max(2, round(nominal_diameter))
    masks: dict[str, np.ndarray] = {}
    adjacency: set[tuple[str, str]] = set()
    component_zone_ids: dict[str, str] = {}
    for component in tissue_scene.graph.components:
        zone_id = f"pop:component:{component.component_id}"
        masks[zone_id] = np.asarray(
            tissue_scene.component_masks[component.component_id], dtype=bool
        )
        component_zone_ids[component.component_id] = zone_id
    for interface in tissue_scene.graph.interfaces:
        interface_mask = tissue_scene.interface_masks[interface.interface_id]
        for side, component_id in (
            ("source", interface.source_component_id),
            ("target", interface.target_component_id),
        ):
            component = tissue_scene.component_masks[component_id]
            distance = ndimage.distance_transform_edt(~interface_mask)
            previous = None
            for band_index, (low, high) in enumerate(
                ((0, band_width), (band_width, 2 * band_width)), start=1
            ):
                zone_id = (
                    f"pop:interface:{interface.interface_id}:{side}:band:{band_index}"
                )
                masks[zone_id] = component & (distance > low) & (distance <= high)
                adjacency.add((component_zone_ids[component_id], zone_id))
                if previous is not None:
                    adjacency.add((previous, zone_id))
                previous = zone_id
    del shape
    zones = tuple(
        _summarize_population_zone(
            zone_id,
            mask,
            instances=instances,
            observation_quality=observation_quality,
            band_width_px=band_width,
        )
        for zone_id, mask in sorted(masks.items())
    )
    warnings = []
    if observation_quality != "native_instance":
        warnings.append("population_statistics_use_semantic_instance_fallback")
    return (
        PopulationGraph(
            zones=zones,
            adjacency=tuple(sorted(adjacency)),
            median_nucleus_area_px=median_area,
            nominal_nucleus_diameter_px=nominal_diameter,
            warnings=tuple(warnings),
        ),
        masks,
    )


def _summarize_population_zone(
    zone_id: str,
    mask: np.ndarray,
    *,
    instances: tuple[NucleusInstance, ...],
    observation_quality: str,
    band_width_px: int,
) -> PopulationZone:
    area = int(np.count_nonzero(mask))
    selected = []
    for item in instances:
        x, y = item.centroid_xy
        row, col = round(y), round(x)
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col]:
            selected.append(item)
    class_counts = {class_id: 0 for class_id in range(1, 6)}
    for item in selected:
        class_counts[item.class_id] += 1
    centers = np.asarray(
        [(item.centroid_xy[1], item.centroid_xy[0]) for item in selected],
        dtype=float,
    )
    nnd = []
    if len(centers) >= 2:
        distances, _ = cKDTree(centers).query(centers, k=2)
        nnd = list(np.asarray(distances)[:, 1])
    area_values = [float(item.area_px) for item in selected]
    zone_kind = "interface_band" if zone_id.startswith("pop:interface:") else "component"
    component_id = None
    interface_id = None
    side = None
    distance_band = None
    if zone_kind == "component":
        component_id = zone_id.removeprefix("pop:component:")
    else:
        marker = ":source:band:" if ":source:band:" in zone_id else ":target:band:"
        interface_id, band_index = zone_id.removeprefix("pop:interface:").split(marker)
        side = "source" if marker.startswith(":source") else "target"
        distance_band = (
            (0.0, float(band_width_px))
            if band_index == "1"
            else (float(band_width_px), float(2 * band_width_px))
        )
    return PopulationZone(
        zone_id=zone_id,
        zone_kind=zone_kind,
        tissue_component_id=component_id,
        interface_id=interface_id,
        side=side,
        distance_band_px=distance_band,
        area_px=area,
        nucleus_count=len(selected),
        density_per_10k_px=(10000.0 * len(selected) / area if area else 0.0),
        class_counts=class_counts,
        class_density_per_10k_px={
            key: (10000.0 * value / area if area else 0.0)
            for key, value in class_counts.items()
        },
        nucleus_area_quantiles=_quantiles(area_values),
        nearest_neighbor_quantiles=_quantiles(nnd),
        observation_quality=observation_quality,
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }
