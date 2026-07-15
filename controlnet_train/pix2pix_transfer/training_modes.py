"""Pure helpers for hard-pair sampling and cross-WSI reference swaps."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic replacement sampling followed by rank-wise sharding."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.weights = weights.detach().double().cpu()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_cross_wsi_permutation(case_ids: Sequence[str]) -> list[int] | None:
    """Find a source permutation where every target receives another WSI."""

    cases = [str(case_id) for case_id in case_ids]
    count = len(cases)
    if count < 2 or len(set(cases)) < 2:
        return None
    order = sorted(range(count), key=lambda index: sum(case == cases[index] for case in cases), reverse=True)
    assignment = [-1] * count
    used = [False] * count

    def assign(position: int) -> bool:
        if position == count:
            return True
        target_index = order[position]
        candidates = [
            source_index
            for source_index in range(count)
            if not used[source_index] and cases[source_index] != cases[target_index]
        ]
        candidates.sort(key=lambda source_index: cases[source_index])
        for source_index in candidates:
            assignment[target_index] = source_index
            used[source_index] = True
            if assign(position + 1):
                return True
            used[source_index] = False
            assignment[target_index] = -1
        return False

    return assignment if assign(0) else None


def build_difficulty_sampling_weights(
    records: Sequence[dict[str, Any]],
    *,
    full_mass: float = 0.40,
    hard_mass: float = 0.30,
) -> torch.Tensor:
    """Give full and partial/low pools the requested aggregate probability mass."""

    full = [str(record.get("pair_difficulty") or "full") == "full" for record in records]
    full_count = sum(full)
    hard_count = len(full) - full_count
    if full_count == 0 or hard_count == 0:
        return torch.ones(len(records), dtype=torch.double)
    full_value = float(full_mass) / float(full_count)
    hard_value = float(hard_mass) / float(hard_count)
    return torch.tensor(
        [full_value if is_full else hard_value for is_full in full],
        dtype=torch.double,
    )
