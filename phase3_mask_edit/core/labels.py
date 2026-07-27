"""Reference mask profile schema helpers for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_config import get_config
from dataset_config.unified_labels import COARSE_LABELS


BIOLOGICAL_LABELS = tuple(label for idx, label in COARSE_LABELS.items() if idx != 0)
DEFAULT_BACKFILL_PRIORITY = (
    "Stroma",
    "Other tissue",
    "Normal epithelium",
    "Immune infiltrate",
)


class MaskProfileSchemaError(ValueError):
    """Raised when a reference mask profile cannot provide a requested label."""


@dataclass(frozen=True)
class MaskProfileSchema:
    """Readable/writable unified label schema for a reference training profile."""

    reference_profile: str
    readable_labels: frozenset[str]
    writable_labels: frozenset[str]
    label_to_fine_ids: dict[str, tuple[int, ...]]
    tumor_fine_ids: tuple[int, ...]
    skip_fine_ids: frozenset[int]
    backfill_priority: tuple[str, ...]
    semantic_warnings: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_reference_profile(cls, reference_profile: str) -> "MaskProfileSchema":
        """Build schema from an existing dataset config used as reference profile."""

        try:
            cfg = get_config(reference_profile)
        except KeyError as exc:
            raise MaskProfileSchemaError(
                f"Unknown reference_profile: {reference_profile}"
            ) from exc

        label_to_fine_ids = _label_to_fine_ids(cfg)
        readable = frozenset(label_to_fine_ids)
        backfill_priority = tuple(
            label for label in DEFAULT_BACKFILL_PRIORITY if label in readable
        )

        return cls(
            reference_profile=cfg.name,
            readable_labels=readable,
            writable_labels=readable,
            label_to_fine_ids=label_to_fine_ids,
            tumor_fine_ids=tuple(cfg.tumor_ids),
            skip_fine_ids=frozenset(cfg.skip_tissues),
            backfill_priority=backfill_priority,
            semantic_warnings=_semantic_warnings(cfg.name, readable),
        )

    def resolve_fine_ids(self, label: str) -> tuple[int, ...]:
        """Return unified fine ids for a readable/writable unified tissue label."""

        fine_ids = self.label_to_fine_ids.get(label)
        if not fine_ids:
            raise MaskProfileSchemaError(
                f"Label {label!r} is not available in reference_profile "
                f"{self.reference_profile}."
            )
        return fine_ids

    def choose_default_backfill_label(
        self, exclude_labels: tuple[str, ...] = ()
    ) -> str:
        """Choose the first available non-tumor backfill label."""

        excluded = set(exclude_labels)
        for label in self.backfill_priority:
            if label not in excluded:
                return label
        raise MaskProfileSchemaError(
            f"No valid backfill label remains for {self.reference_profile}."
        )


def _label_to_fine_ids(cfg) -> dict[str, tuple[int, ...]]:
    candidates = {
        "Tumor": tuple(cfg.tumor_ids),
        "Stroma": tuple(cfg.stroma_ids),
        "Necrosis": tuple(cfg.necrosis_ids),
        "Immune infiltrate": tuple(cfg.immune_ids),
        "Normal epithelium": tuple(cfg.normal_epi_ids),
        "Blood vessel": tuple(cfg.vessel_ids),
        "Other tissue": _other_tissue_ids(cfg),
    }
    return {
        label: fine_ids
        for label, fine_ids in candidates.items()
        if label in BIOLOGICAL_LABELS and fine_ids
    }


def _other_tissue_ids(cfg) -> tuple[int, ...]:
    original_ids = cfg.coarse_to_original.get(7, [])
    if not original_ids:
        return ()

    fine_ids = sorted({cfg.to_fine_map[original_id] for original_id in original_ids})
    return tuple(fine_ids)


def _semantic_warnings(profile_name: str, readable_labels: frozenset[str]) -> dict[str, str]:
    warnings: dict[str, str] = {}
    if profile_name.upper() == "UNIFIED_COARSE":
        warnings["__profile__"] = (
            "UNIFIED_COARSE describes mask encoding only; record image/dataset "
            "provenance and downstream organ conditioning separately."
        )
    if profile_name.upper() == "ORCA" and "Other tissue" in readable_labels:
        warnings["Other tissue"] = (
            "ORCA non-carcinoma tissue is a coarse mixed class; use it as "
            "non-tumor context/backfill, not as specific stroma, immune, or "
            "normal epithelium."
        )
    return warnings
