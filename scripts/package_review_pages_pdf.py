#!/usr/bin/env python3
"""Package rendered PNG review pages into a landscape PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pages = sorted(args.pages_dir.glob("page_*.png"))
    if not pages:
        raise FileNotFoundError(f"no page PNGs under {args.pages_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = 16 * 72, 10 * 72
    document = canvas.Canvas(str(args.output), pagesize=(page_width, page_height))
    document.setTitle(args.output.stem)
    for path in pages:
        with Image.open(path) as image:
            image_width, image_height = image.size
        scale = min(page_width / image_width, page_height / image_height)
        draw_width = image_width * scale
        draw_height = image_height * scale
        document.drawImage(
            ImageReader(str(path)),
            (page_width - draw_width) / 2,
            (page_height - draw_height) / 2,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        document.showPage()
    document.save()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
