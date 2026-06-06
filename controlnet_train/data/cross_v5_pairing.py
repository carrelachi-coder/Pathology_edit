"""Cross V5 pairing sampler for reference-bank training."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

from controlnet_train.training.cross_v5_glue import CrossV5PairingPolicy


_PAIR_MODES = ("same_wsi", "cross_wsi", "high_appearance_gap")
_COVERAGE_BUCKETS = ("full", "partial", "low")


@dataclass(frozen=True)
class CrossV5PairingSamplerConfig:
    """Controls metadata-driven V5 pair sampling."""

    high_gap_quantile: float = 0.75
    keep_at_least_one_bank_class: bool = True


class CrossV5PairingSampler:
    """Sample pair records by WSI relation, appearance gap, and coverage.

    Records are expected to contain the existing cross metadata fields plus,
    when available, ``reference_case_id``/``reference_wsi_id`` and an
    ``appearance_gap``-like scalar. Missing reference case IDs are treated as
    same-WSI for backward compatibility with existing same-case metadata.
    """

    def __init__(
        self,
        records: Sequence[Mapping],
        *,
        policy: CrossV5PairingPolicy | None = None,
        config: CrossV5PairingSamplerConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("CrossV5PairingSampler requires at least one pair record.")
        self.records = [dict(record) for record in records]
        self.policy = policy or CrossV5PairingPolicy()
        self.config = config or CrossV5PairingSamplerConfig()
        self.rng = random.Random(seed)
        self._high_gap_threshold = _compute_high_gap_threshold(
            self.records,
            quantile=float(self.config.high_gap_quantile),
        )

    def sample(self) -> dict:
        pair_mode = self._draw_pair_mode()
        mode_candidates = self._filter_by_pair_mode(pair_mode)
        if not mode_candidates:
            mode_candidates = list(self.records)
            pair_mode = "fallback_all"

        coverage_mode = self._draw_coverage_mode(mode_candidates)
        coverage_candidates = [
            record
            for record in mode_candidates
            if _coverage_bucket(record) == coverage_mode
        ]
        if not coverage_candidates:
            coverage_candidates = mode_candidates
            coverage_mode = "fallback_any"

        record = dict(self.rng.choice(coverage_candidates))
        keep_ids, drop_ids = self._sample_bank_dropout(record)
        record.update(
            {
                "v5_pair_mode": pair_mode,
                "v5_coverage_mode": coverage_mode,
                "v5_reference_bank_keep_tissue_ids": keep_ids,
                "v5_reference_bank_drop_tissue_ids": drop_ids,
            }
        )
        return record

    def _draw_pair_mode(self) -> str:
        return _weighted_choice(self.policy.normalized_pair_mode_weights(), self.rng)

    def _draw_coverage_mode(self, records: Sequence[Mapping]) -> str:
        weights = self.policy.normalized_coverage_weights()
        available = {_coverage_bucket(record) for record in records}
        weights = {key: value for key, value in weights.items() if key in available and value > 0.0}
        if not weights:
            return _coverage_bucket(self.rng.choice(list(records)))
        return _weighted_choice(weights, self.rng)

    def _filter_by_pair_mode(self, pair_mode: str) -> list[dict]:
        if pair_mode == "same_wsi":
            return [record for record in self.records if _is_same_wsi(record)]
        if pair_mode == "cross_wsi":
            return [record for record in self.records if not _is_same_wsi(record)]
        if pair_mode == "high_appearance_gap":
            return [
                record
                for record in self.records
                if _appearance_gap(record) >= self._high_gap_threshold
            ]
        raise ValueError(f"Unsupported V5 pair mode {pair_mode!r}.")

    def _sample_bank_dropout(self, record: Mapping) -> tuple[list[int], list[int]]:
        available_ids = _available_bank_tissue_ids(record)
        if not available_ids:
            return [], []
        keep_ids: list[int] = []
        drop_ids: list[int] = []
        for tissue_id in available_ids:
            if self.rng.random() < float(self.policy.class_bank_dropout_prob):
                drop_ids.append(tissue_id)
            else:
                keep_ids.append(tissue_id)
        if (
            self.config.keep_at_least_one_bank_class
            and not keep_ids
            and drop_ids
        ):
            keep_index = self.rng.randrange(len(drop_ids))
            keep_ids.append(drop_ids.pop(keep_index))
        return sorted(keep_ids), sorted(drop_ids)


class CrossV5PairingDataset:
    """Dataset wrapper that samples V5 pair records per epoch item.

    When ``base_dataset`` is provided, the wrapper samples a record index from
    ``base_dataset.records`` and returns the fully loaded image/mask item with
    V5 pairing metadata attached. Without ``base_dataset`` it preserves the
    earlier metadata-only behavior for sampler smoke tests.
    """

    def __init__(
        self,
        records: Sequence[Mapping] | None = None,
        *,
        base_dataset=None,
        pairs_per_epoch: int | None = None,
        policy: CrossV5PairingPolicy | None = None,
        config: CrossV5PairingSamplerConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        if records is None:
            if base_dataset is None or not hasattr(base_dataset, "records"):
                raise ValueError("CrossV5PairingDataset requires records or a base_dataset with records.")
            records = getattr(base_dataset, "records")
        indexed_records = [
            {**dict(record), "__v5_record_index": index}
            for index, record in enumerate(records)
        ]
        self.records = getattr(base_dataset, "records", indexed_records)
        self.sampler = CrossV5PairingSampler(
            indexed_records,
            policy=policy,
            config=config,
            seed=seed,
        )
        self.pairs_per_epoch = int(pairs_per_epoch or len(indexed_records))
        if self.pairs_per_epoch <= 0:
            raise ValueError("pairs_per_epoch must be positive.")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int) -> dict:
        del index
        sampled = self.sampler.sample()
        if self.base_dataset is None:
            sampled.pop("__v5_record_index", None)
            return sampled
        source_index = int(sampled["__v5_record_index"])
        item = dict(self.base_dataset[source_index])
        item.update(
            {
                "v5_pair_mode": sampled["v5_pair_mode"],
                "v5_coverage_mode": sampled["v5_coverage_mode"],
                "v5_reference_bank_keep_tissue_ids": sampled["v5_reference_bank_keep_tissue_ids"],
                "v5_reference_bank_drop_tissue_ids": sampled["v5_reference_bank_drop_tissue_ids"],
            }
        )
        return item


def _weighted_choice(weights: Mapping[str, float], rng: random.Random) -> str:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        raise ValueError(f"At least one sampling weight must be positive, got {weights}.")
    cursor = rng.random() * total
    for key, value in weights.items():
        cursor -= max(0.0, float(value))
        if cursor <= 0.0:
            return key
    return next(reversed(weights))


def _coverage_bucket(record: Mapping) -> str:
    value = str(record.get("pair_difficulty", record.get("coverage", "full")) or "full").lower()
    return value if value in _COVERAGE_BUCKETS else "full"


def _is_same_wsi(record: Mapping) -> bool:
    target = _record_case_key(record, prefix="")
    reference = _record_case_key(record, prefix="reference_")
    return reference == target


def _record_case_key(record: Mapping, *, prefix: str) -> str:
    for key in (
        f"{prefix}wsi_id",
        f"{prefix}case_id",
        f"{prefix}slide_id",
    ):
        value = record.get(key)
        if value is not None:
            return str(value)
    return str(record.get("case_id", ""))


def _appearance_gap(record: Mapping) -> float:
    for key in ("appearance_gap", "appearance_gap_score", "stain_distance", "stain_gap"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _compute_high_gap_threshold(records: Sequence[Mapping], *, quantile: float) -> float:
    gaps = sorted(_appearance_gap(record) for record in records)
    if not gaps:
        return 0.0
    q = min(max(float(quantile), 0.0), 1.0)
    index = int(round(q * (len(gaps) - 1)))
    return gaps[index]


def _available_bank_tissue_ids(record: Mapping) -> list[int]:
    raw = (
        record.get("covered_target_tissue_ids")
        or record.get("reference_tissue_ids")
        or record.get("target_tissue_ids")
        or []
    )
    return sorted({int(value) for value in raw})


__all__ = [
    "CrossV5PairingDataset",
    "CrossV5PairingSampler",
    "CrossV5PairingSamplerConfig",
]
