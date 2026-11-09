"""Unintegrated budget candidate. Production Codec and Adapter stay unchanged.

Avoid reparsing concrete Path objects and recomputing one claim's reservation.
All rounding, headroom, limits and observed-allocation accounting are inherited.
No filesystem result is cached. Intended for independent differential review.
"""
from pathlib import Path, PosixPath, WindowsPath
from .guard_blocks_v2 import Budget, BudgetError, CONTROL_RESERVE


class CandidateBudget(Budget):
    def reservation(self, path, size):
        name = path.name if type(path) in (PosixPath, WindowsPath) else Path(path).name
        if name.endswith('.index.pending'):
            size = 2 * size + 65536
            return max(((size + 4095) // 4096 + 1) * 4096, self.charges.get(str(path), 0))
        return ((size + 4095) // 4096 + 1) * 4096

    def claim(self, path, size, control=False):
        if type(path) not in (str, PosixPath, WindowsPath):
            return super().claim(path, size, control)
        charge = self.reservation(path, size)
        key = str(path)
        allowed = self.limit if control else self.limit - CONTROL_RESERVE
        if self.total - self.charges.get(key, 0) + charge > allowed:
            raise BudgetError('unified storage budget')
        self.total += charge - self.charges.get(key, 0)
        self.charges[key] = charge
        self.sizes[key] = size


def attach_for_independent_trial(writer):
    """Explicit test-only substitution. Does not patch module globals or files."""
    if type(writer.budget) is not Budget or writer.events or writer.arrays:
        raise ValueError('requires fresh writer with original budget')
    old = writer.budget
    candidate = CandidateBudget(old.limit)
    candidate.charges = dict(old.charges)
    candidate.sizes = dict(old.sizes)
    candidate.total = old.total
    writer.budget = candidate
