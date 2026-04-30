"""Current-mask facts for Phase 3 edit applicability and execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from phase3_mask_edit.core.labels import MaskProfileSchema


class MaskEditContextError(ValueError):
    """Raised when a mask cannot be summarized into an edit context."""


@dataclass(frozen=True)
class MaskEditContext:
    """Structured facts derived from the current old mask."""

    reference_profile: str
    mask_shape: tuple[int, int]
    present_labels: frozenset[str]
    label_area_fractions: dict[str, float]
    fine_id_area_fractions: dict[int, float]
    adjacency: dict[str, frozenset[str]]
    component_counts: dict[str, int]
    normalized_mask: np.ndarray
    risk_flags: tuple[str, ...] = ()
    semantic_warnings: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mask(
        cls, mask: np.ndarray, schema: MaskProfileSchema
    ) -> "MaskEditContext":
        """Summarize a 2D unified fine-id mask under a reference profile."""

        mask_array = np.asarray(mask)
        if mask_array.ndim != 2:
            raise MaskEditContextError("MaskEditContext requires a 2D id mask.")

        total_pixels = int(mask_array.size)
        if total_pixels == 0:
            raise MaskEditContextError("MaskEditContext requires a non-empty mask.")

        fine_id_to_label = _fine_id_to_label(schema)
        normalized_mask, remapped_ids = _normalize_unknown_ids_to_other_tissue(
            mask_array, fine_id_to_label, schema.skip_fine_ids
        )
        normalized_fine_id_to_label = dict(fine_id_to_label)
        normalized_fine_id_to_label.setdefault(7, "Other tissue")
        normalized_label_to_fine_ids = dict(schema.label_to_fine_ids)
        normalized_label_to_fine_ids.setdefault("Other tissue", (7,))

        unique_ids, counts = np.unique(normalized_mask, return_counts=True)

        fine_id_area_fractions = {
            int(fine_id): int(count) / total_pixels
            for fine_id, count in zip(unique_ids, counts)
        }

        label_pixel_counts: dict[str, int] = {}
        unknown_ids: list[int] = []
        for fine_id, count in zip(unique_ids, counts):
            fine_id_int = int(fine_id)
            if fine_id_int in schema.skip_fine_ids:
                continue
            label = normalized_fine_id_to_label.get(fine_id_int)
            label_pixel_counts[label] = label_pixel_counts.get(label, 0) + int(count)

        present_labels = frozenset(label_pixel_counts)
        label_area_fractions = {
            label: count / total_pixels for label, count in label_pixel_counts.items()
        }

        risk_flags = tuple(
            f"remapped_unknown_fine_ids_to_other_tissue:{fine_id}"
            for fine_id in sorted(remapped_ids)
        )

        return cls(
            reference_profile=schema.reference_profile,
            mask_shape=tuple(int(dim) for dim in normalized_mask.shape),
            present_labels=present_labels,
            label_area_fractions=label_area_fractions,
            fine_id_area_fractions=fine_id_area_fractions,
            adjacency=_compute_adjacency(
                normalized_mask, normalized_fine_id_to_label, schema.skip_fine_ids
            ),
            component_counts=_compute_component_counts(
                normalized_mask, normalized_label_to_fine_ids, present_labels
            ),
            normalized_mask=normalized_mask,
            risk_flags=risk_flags,
            semantic_warnings=dict(schema.semantic_warnings),
        )


def _fine_id_to_label(schema: MaskProfileSchema) -> dict[int, str]:
    fine_id_to_label: dict[int, str] = {}
    for label, fine_ids in schema.label_to_fine_ids.items():
        for fine_id in fine_ids:
            fine_id_to_label[int(fine_id)] = label
    return fine_id_to_label


def _normalize_unknown_ids_to_other_tissue(
    mask: np.ndarray, fine_id_to_label: dict[int, str], skip_fine_ids: frozenset[int]
) -> tuple[np.ndarray, tuple[int, ...]]:
    known_ids = set(fine_id_to_label) | set(skip_fine_ids)
    normalized = np.array(mask, copy=True)
    remapped_ids: set[int] = set()

    for fine_id in np.unique(mask):
        fine_id_int = int(fine_id)
        if fine_id_int in known_ids:
            continue
        normalized[mask == fine_id_int] = 7
        remapped_ids.add(fine_id_int)

    return normalized, tuple(sorted(remapped_ids))


def _compute_adjacency(
    mask: np.ndarray, fine_id_to_label: dict[int, str], skip_fine_ids: frozenset[int]
) -> dict[str, frozenset[str]]:
    adjacency: dict[str, set[str]] = {
        label: set() for label in set(fine_id_to_label.values())
    }

    for shifted_a, shifted_b in (
        (mask[:, :-1], mask[:, 1:]),
        (mask[:-1, :], mask[1:, :]),
    ):
        different = shifted_a != shifted_b
        if not np.any(different):
            continue
        left_ids = shifted_a[different]
        right_ids = shifted_b[different]
        for left_id, right_id in zip(left_ids, right_ids):
            left_label = _label_for_id(int(left_id), fine_id_to_label, skip_fine_ids)
            right_label = _label_for_id(int(right_id), fine_id_to_label, skip_fine_ids)
            if left_label is None or right_label is None or left_label == right_label:
                continue
            adjacency[left_label].add(right_label)
            adjacency[right_label].add(left_label)

    return {label: frozenset(neighbors) for label, neighbors in adjacency.items()}


def _label_for_id(
    fine_id: int, fine_id_to_label: dict[int, str], skip_fine_ids: frozenset[int]
) -> str | None:
    if fine_id in skip_fine_ids:
        return None
    return fine_id_to_label.get(fine_id)


def _compute_component_counts(
    mask: np.ndarray,
    label_to_fine_ids: dict[str, tuple[int, ...]],
    present_labels: frozenset[str],
) -> dict[str, int]:
    return {
        label: _count_components(np.isin(mask, fine_ids))
        for label, fine_ids in label_to_fine_ids.items()
        if label in present_labels
    }


def _count_components(binary_mask: np.ndarray) -> int:
    visited = np.zeros(binary_mask.shape, dtype=bool)
    count = 0
    rows, cols = binary_mask.shape

    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not binary_mask[row, col]:
                continue
            count += 1
            _mark_component(binary_mask, visited, row, col)
    return count


def _mark_component(
    binary_mask: np.ndarray, visited: np.ndarray, start_row: int, start_col: int
) -> None:
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    visited[start_row, start_col] = True
    rows, cols = binary_mask.shape

    while queue:
        row, col = queue.popleft()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if visited[next_row, next_col] or not binary_mask[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            queue.append((next_row, next_col))
