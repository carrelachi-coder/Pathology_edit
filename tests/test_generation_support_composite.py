from __future__ import annotations

import numpy as np
from PIL import Image

from controlnet_train.inference.composite import (
    source_exact_generation_composite,
)


def test_generation_composite_is_source_exact_outside_support() -> None:
    source = np.full((16, 16, 3), 20, dtype=np.uint8)
    generated = Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8))
    support = np.zeros((16, 16), dtype=bool)
    support[3:13, 3:13] = True

    result, metadata = source_exact_generation_composite(
        source,
        generated,
        support,
        feather_px=3,
    )
    observed = np.asarray(result)

    assert np.array_equal(observed[~support], source[~support])
    assert np.all(observed[7:9, 7:9] == 220)
    assert np.all((observed[3, 3] > 20) & (observed[3, 3] < 220))
    assert metadata["outside_support_source_exact"] is True
    assert metadata["outside_support_changed_pixels"] == 0


def test_generation_composite_with_empty_support_returns_source() -> None:
    source = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
    generated = Image.fromarray(np.zeros_like(source))

    result, metadata = source_exact_generation_composite(
        source,
        generated,
        np.zeros((12, 10), dtype=bool),
    )

    assert np.array_equal(np.asarray(result), source)
    assert metadata["support_pixels"] == 0
    assert metadata["outside_support_source_exact"] is True
