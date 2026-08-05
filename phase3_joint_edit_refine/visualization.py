"""Review boards for source and paired joint conditions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.visualization import id_mask_to_rgb

from .models import JointCaseContext
from .nuclei import load_nuclei_mask

NUCLEI_RGB = {
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 80, 255],
    4: [255, 255, 0],
    5: [255, 0, 255],
}


def build_source_review_boards(
    cases: list[JointCaseContext],
    *,
    output_dir: str | Path,
    tile_size: int = 256,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for case in cases:
        grouped[case.pathology_domain_id].append(case)
    result = {}
    for domain, values in sorted(grouped.items()):
        row_height = tile_size + 36
        canvas = Image.new("RGB", (tile_size * 3, row_height * len(values)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, case in enumerate(values):
            tissue = load_id_mask(case.source_tissue_mask_uri)
            nuclei = load_nuclei_mask(case.source_nuclei_mask_uri)
            image = np.asarray(Image.open(case.source_image_uri).convert("RGB"), dtype=np.uint8)
            tissue_view = _blend(image, id_mask_to_rgb(tissue), 0.42)
            nuclei_view = np.array(image, copy=True)
            for class_id, color in NUCLEI_RGB.items():
                nuclei_view[nuclei == int(class_id)] = np.asarray(color, dtype=np.uint8)
            y = index * row_height
            draw.text((5, y + 4), f"{case.case_id} | {case.primitive_id}", fill="black")
            for column, panel in enumerate((image, tissue_view, nuclei_view)):
                resized = Image.fromarray(panel).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                canvas.paste(resized, (column * tile_size, y + 36))
        path = output / f"source_review_{domain}.png"
        canvas.save(path)
        result[domain] = str(path)
    return result


def _blend(left, right, alpha):
    return np.clip((1 - alpha) * left.astype(float) + alpha * right.astype(float), 0, 255).astype(np.uint8)
