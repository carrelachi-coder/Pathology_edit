"""Conservative, confidence-gated semantic post-processing.

P1 is deliberately a label-map correction layer, not model adaptation. It can
only relabel tiny uncertain islands when both the surrounding labels and the
underlying posterior support one replacement class. Every changed component is
logged and the raw prediction remains the authoritative artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .metrics import build_edit_regions, normalized_entropy


@dataclass(frozen=True)
class ConservativeP1Policy:
    policy_id: str = "conservative-island-p1-v1"
    status: str = "canary_candidate"
    max_component_area_pixels: int = 16
    max_total_changed_fraction: float = 0.0005
    ring_radius_pixels: int = 2
    protected_boundary_radius_pixels: int = 4
    component_top1_probability_max: float = 0.70
    component_normalized_entropy_min: float = 0.45
    ring_label_dominance_min: float = 0.90
    ring_replacement_probability_min: float = 0.70
    component_replacement_probability_min: float = 0.15
    component_top1_replacement_margin_max: float = 0.45
    stable_source_overlap_max: float = 0.50

    def validate(self) -> None:
        if self.max_component_area_pixels < 1:
            raise ValueError("max_component_area_pixels must be positive")
        if self.ring_radius_pixels < 1:
            raise ValueError("ring_radius_pixels must be positive")
        if self.protected_boundary_radius_pixels < 1:
            raise ValueError("protected_boundary_radius_pixels must be positive")
        for name in (
            "max_total_changed_fraction",
            "component_top1_probability_max",
            "component_normalized_entropy_min",
            "ring_label_dominance_min",
            "ring_replacement_probability_min",
            "component_replacement_probability_min",
            "component_top1_replacement_margin_max",
            "stable_source_overlap_max",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class P1Operation:
    operation: str
    source_class_id: int
    replacement_class_id: int
    area_pixels: int
    component_mean_top1_probability: float
    component_mean_normalized_entropy: float
    component_mean_replacement_probability: float
    component_mean_top1_replacement_margin: float
    ring_label_dominance: float
    ring_mean_replacement_probability: float
    stable_source_overlap: float | None
    region: str
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class P1Result:
    raw_mask: np.ndarray
    audited_mask: np.ndarray
    changed_mask: np.ndarray
    operations: tuple[P1Operation, ...]
    policy: ConservativeP1Policy
    stopped_at_change_budget: bool

    def to_metadata(self) -> dict[str, Any]:
        changed_pixels = int(np.count_nonzero(self.changed_mask))
        return {
            "schema_version": 1,
            "policy": asdict(self.policy),
            "raw_preserved": True,
            "changed_pixels": changed_pixels,
            "changed_fraction": changed_pixels / int(self.changed_mask.size),
            "operation_count": len(self.operations),
            "operations": [asdict(item) for item in self.operations],
            "stopped_at_change_budget": self.stopped_at_change_budget,
        }


def apply_conservative_p1(
    *,
    predicted_mask: np.ndarray,
    probabilities: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_prediction: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    policy: ConservativeP1Policy | None = None,
    ignore_index: int = 255,
    semantic_change_region: np.ndarray | None = None,
) -> P1Result:
    """Return a shadow audited mask without mutating the raw prediction.

    Pixels in the semantic boundary band are immutable. Outside the requested
    edit, an island that agrees with the source prediction is also immutable.
    This prevents P1 from erasing stable structures merely because they are
    small.
    """

    policy = policy or ConservativeP1Policy()
    policy.validate()
    raw = np.asarray(predicted_mask)
    source = np.asarray(source_mask)
    target = np.asarray(target_mask)
    if raw.ndim != 2 or not (raw.shape == source.shape == target.shape):
        raise ValueError("predicted, source, and target masks must share a rank-2 shape")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 3 or probs.shape[1:] != raw.shape:
        raise ValueError("probabilities must have CHW layout matching the mask")
    if np.any(raw < 0) or np.any(raw >= probs.shape[0]):
        raise ValueError("predicted mask contains a class outside probability channels")
    entropy_map = (
        normalized_entropy(probs)
        if entropy is None
        else np.asarray(entropy, dtype=np.float64)
    )
    if entropy_map.shape != raw.shape:
        raise ValueError("entropy must match the mask shape")
    source_pred = (
        None if source_prediction is None else np.asarray(source_prediction)
    )
    if source_pred is not None and source_pred.shape != raw.shape:
        raise ValueError("source_prediction must match the predicted mask shape")

    regions = build_edit_regions(
        source,
        target,
        ignore_index=ignore_index,
        boundary_radius=policy.protected_boundary_radius_pixels,
        semantic_change_region=semantic_change_region,
    )
    immutable = regions["B"] | ~regions["valid"]
    audited = np.array(raw, copy=True)
    top1 = np.take_along_axis(probs, raw[None, ...], axis=0)[0]
    operation_log: list[P1Operation] = []
    changed_budget = max(
        1, int(np.floor(raw.size * policy.max_total_changed_fraction))
    )
    changed_count = 0
    stopped = False

    for class_id in sorted(int(value) for value in np.unique(raw)):
        class_mask = raw == class_id
        for component in _connected_components(class_mask):
            area = int(np.count_nonzero(component))
            if area > policy.max_component_area_pixels:
                continue
            if np.any(component & immutable):
                continue
            component_top1 = float(np.mean(top1[component]))
            component_entropy = float(np.mean(entropy_map[component]))
            if component_top1 > policy.component_top1_probability_max:
                continue
            if component_entropy < policy.component_normalized_entropy_min:
                continue

            in_edit = bool(np.any(component & regions["R"]))
            stable_overlap = None
            if source_pred is not None:
                stable_overlap = float(np.mean(source_pred[component] == class_id))
                if not in_edit and stable_overlap > policy.stable_source_overlap_max:
                    continue

            ring = _binary_dilation(component, policy.ring_radius_pixels) & ~component
            ring &= regions["valid"] & ~immutable
            if not np.any(ring):
                continue
            ring_labels = audited[ring]
            values, counts = np.unique(ring_labels, return_counts=True)
            order = np.argsort(counts)[::-1]
            replacement = int(values[order[0]])
            dominance = float(counts[order[0]] / counts.sum())
            if replacement == class_id or dominance < policy.ring_label_dominance_min:
                continue
            ring_replacement_probability = float(np.mean(probs[replacement][ring]))
            if (
                ring_replacement_probability
                < policy.ring_replacement_probability_min
            ):
                continue
            component_replacement_probability = float(
                np.mean(probs[replacement][component])
            )
            component_margin = float(
                np.mean(top1[component] - probs[replacement][component])
            )
            if (
                component_replacement_probability
                < policy.component_replacement_probability_min
                or component_margin
                > policy.component_top1_replacement_margin_max
            ):
                continue
            if changed_count + area > changed_budget:
                stopped = True
                break

            ys, xs = np.where(component)
            operation = (
                "enclosed_hole_fill"
                if dominance >= 0.999
                else "low_confidence_island_relabel"
            )
            audited[component] = replacement
            changed_count += area
            operation_log.append(
                P1Operation(
                    operation=operation,
                    source_class_id=class_id,
                    replacement_class_id=replacement,
                    area_pixels=area,
                    component_mean_top1_probability=component_top1,
                    component_mean_normalized_entropy=component_entropy,
                    component_mean_replacement_probability=(
                        component_replacement_probability
                    ),
                    component_mean_top1_replacement_margin=component_margin,
                    ring_label_dominance=dominance,
                    ring_mean_replacement_probability=(
                        ring_replacement_probability
                    ),
                    stable_source_overlap=stable_overlap,
                    region="R" if in_edit else "U_far",
                    bbox_xyxy=(
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ),
                )
            )
        if stopped:
            break

    changed = audited != raw
    return P1Result(
        raw_mask=np.array(raw, copy=True),
        audited_mask=audited,
        changed_mask=changed,
        operations=tuple(operation_log),
        policy=policy,
        stopped_at_change_budget=stopped,
    )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return deterministic four-connected components without SciPy."""

    values = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(values, dtype=bool)
    components: list[np.ndarray] = []
    height, width = values.shape
    for y, x in np.argwhere(values):
        y = int(y)
        x = int(x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack = [(y, x)]
        pixels: list[tuple[int, int]] = []
        while stack:
            py, px = stack.pop()
            pixels.append((py, px))
            for ny, nx in (
                (py - 1, px),
                (py + 1, px),
                (py, px - 1),
                (py, px + 1),
            ):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and values[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        component = np.zeros_like(values, dtype=bool)
        ys, xs = zip(*pixels)
        component[np.asarray(ys), np.asarray(xs)] = True
        components.append(component)
    return components


def _binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        expanded = result.copy()
        expanded[1:, :] |= result[:-1, :]
        expanded[:-1, :] |= result[1:, :]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        result = expanded
    return result
