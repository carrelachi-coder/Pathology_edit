from __future__ import annotations

import numpy as np

from phase3_joint_edit_refine.lumen_observer import observe_luminal_spaces


def _paint_nucleus(mask: np.ndarray, y: int, x: int, class_id: int = 3) -> None:
    rows, cols = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(rows - y) ** 2 + (cols - x) ** 2 <= 3**2] = class_id


def test_observes_closed_and_colored_lumen_with_sparse_cells() -> None:
    shape = (128, 128)
    tissue = np.full(shape, 2, dtype=np.uint8)
    image = np.full((*shape, 3), (205, 150, 175), dtype=np.uint8)
    nuclei = np.zeros(shape, dtype=np.uint8)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    radius = np.sqrt((rows - 64) ** 2 + (cols - 64) ** 2)
    tissue[(radius >= 25) & (radius <= 38)] = 9
    image[radius < 25] = (235, 205, 215)
    image[(radius >= 25) & (radius <= 38)] = (185, 95, 140)
    for angle in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        _paint_nucleus(
            nuclei,
            int(round(64 + 31 * np.sin(angle))),
            int(round(64 + 31 * np.cos(angle))),
            1,
        )
    _paint_nucleus(nuclei, 61, 66, 2)
    for y in range(8, 121, 18):
        _paint_nucleus(nuclei, y, 10, 3)

    observed = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        architecture_fine_ids=(8, 9, 10),
    )

    assert observed.confirmed_lumen[64, 64]
    assert not observed.external_stroma[64, 64]


def test_observes_patch_edge_truncated_lumen() -> None:
    shape = (128, 128)
    tissue = np.full(shape, 2, dtype=np.uint8)
    image = np.full((*shape, 3), (210, 155, 180), dtype=np.uint8)
    nuclei = np.zeros(shape, dtype=np.uint8)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    radius = np.sqrt((rows - 64) ** 2 + (cols - 3) ** 2)
    shell = (radius >= 24) & (radius <= 37)
    tissue[shell] = 9
    image[radius < 24] = (248, 235, 238)
    image[shell] = (180, 90, 135)
    for angle in np.linspace(-1.35, 1.35, 12):
        y = int(round(64 + 30 * np.sin(angle)))
        x = int(round(3 + 30 * np.cos(angle)))
        if 3 < x < 125 and 3 < y < 125:
            _paint_nucleus(nuclei, y, x, 1)
    for y in range(10, 120, 14):
        _paint_nucleus(nuclei, y, 70, 3)
        _paint_nucleus(nuclei, y, 100, 3)

    observed = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        architecture_fine_ids=(8, 9, 10),
    )

    assert observed.confirmed_lumen[64, 0]
    assert any(
        item.classification == "open_edge_lumen" for item in observed.regions
    )


def test_cellular_external_stroma_remains_external() -> None:
    shape = (128, 128)
    tissue = np.full(shape, 2, dtype=np.uint8)
    tissue[:, 0:28] = 9
    image = np.full((*shape, 3), (220, 160, 185), dtype=np.uint8)
    image[:, 0:28] = (180, 90, 135)
    nuclei = np.zeros(shape, dtype=np.uint8)
    for y in range(7, 126, 11):
        for x in range(36, 126, 13):
            _paint_nucleus(nuclei, y, x, 3)
    for y in range(7, 126, 11):
        _paint_nucleus(nuclei, y, 18, 1)

    observed = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        architecture_fine_ids=(8, 9, 10),
    )

    assert observed.external_stroma[64, 70]
    assert not observed.protected_space[64, 70]


def test_large_sparse_edge_stroma_around_separate_glands_is_not_lumen() -> None:
    shape = (160, 160)
    tissue = np.full(shape, 2, dtype=np.uint8)
    image = np.full((*shape, 3), (238, 218, 224), dtype=np.uint8)
    nuclei = np.zeros(shape, dtype=np.uint8)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    for cy, cx in ((45, 45), (112, 106), (46, 122)):
        radius = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
        tissue[(radius >= 12) & (radius <= 19)] = 9
        image[(radius >= 12) & (radius <= 19)] = (178, 86, 132)
        for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            _paint_nucleus(
                nuclei,
                int(round(cy + 15 * np.sin(angle))),
                int(round(cx + 15 * np.cos(angle))),
                1,
            )
    # The interiors are plausible lumina; the large edge-connected field is not.
    for y, x in ((12, 12), (18, 78), (76, 18), (145, 76), (80, 145)):
        _paint_nucleus(nuclei, y, x, 3)

    observed = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        architecture_fine_ids=(8, 9, 10),
    )

    assert not observed.confirmed_lumen[5, 80]
    assert not observed.uncertain_low_cell_space[5, 80]
    assert observed.external_stroma[5, 80]
    assert observed.confirmed_lumen[45, 45]


def test_dark_sparse_debris_space_is_not_confirmed_lumen() -> None:
    shape = (128, 128)
    tissue = np.full(shape, 2, dtype=np.uint8)
    image = np.full((*shape, 3), (210, 160, 180), dtype=np.uint8)
    nuclei = np.zeros(shape, dtype=np.uint8)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    radius = np.sqrt((rows - 64) ** 2 + (cols - 64) ** 2)
    tissue[(radius >= 25) & (radius <= 38)] = 9
    image[radius < 25] = (35, 22, 30)
    image[(radius >= 25) & (radius <= 38)] = (180, 90, 135)
    for angle in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        _paint_nucleus(
            nuclei,
            int(round(64 + 31 * np.sin(angle))),
            int(round(64 + 31 * np.cos(angle))),
            1,
        )

    observed = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        architecture_fine_ids=(8, 9, 10),
    )

    assert not observed.confirmed_lumen[64, 64]


def test_glas_lumen_is_found_inside_shared_gland_tumor_label() -> None:
    shape = (144, 144)
    tissue = np.full(shape, 2, dtype=np.uint8)
    image = np.full((*shape, 3), (214, 162, 184), dtype=np.uint8)
    nuclei = np.zeros(shape, dtype=np.uint8)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    radius = np.sqrt((rows - 72) ** 2 + (cols - 72) ** 2)
    # GLaS uses one gland/tumor label for both epithelial wall and lumen.
    tissue[radius <= 42] = 11
    image[radius < 25] = (242, 221, 226)
    image[(radius >= 25) & (radius <= 42)] = (177, 88, 132)
    for angle in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        _paint_nucleus(
            nuclei,
            int(round(72 + 33 * np.sin(angle))),
            int(round(72 + 33 * np.cos(angle))),
            1,
        )
    for y in range(8, 137, 13):
        _paint_nucleus(nuclei, y, 12, 3)

    observed = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        architecture_fine_ids=(5, 11, 12, 13),
        lumen_encoding="within_architecture",
    )

    assert observed.confirmed_lumen[72, 72]
    assert not observed.protected_space[72, 12]


def test_glas_light_tumor_cytoplasm_is_not_promoted_to_lumen() -> None:
    shape = (144, 144)
    tissue = np.full(shape, 11, dtype=np.uint8)
    image = np.full((*shape, 3), (190, 132, 161), dtype=np.uint8)
    nuclei = np.zeros(shape, dtype=np.uint8)
    for y in range(8, 140, 13):
        for x in range(8, 140, 13):
            _paint_nucleus(nuclei, y, x, 1)

    observed = observe_luminal_spaces(
        image,
        tissue,
        nuclei,
        architecture_fine_ids=(5, 11, 12, 13),
        lumen_encoding="within_architecture",
    )

    assert not np.any(observed.protected_space)
