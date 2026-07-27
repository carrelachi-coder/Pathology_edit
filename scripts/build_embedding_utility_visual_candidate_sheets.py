#!/usr/bin/env python3
"""Build visual case-selection sheets for embedding-utility trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from matplotlib import font_manager
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


MODERATE_COLOR = (22, 138, 173)
SIGNIFICANT_COLOR = (231, 111, 81)
TEXT_COLOR = (24, 34, 52)
MUTED_COLOR = (86, 97, 116)
BORDER_COLOR = (215, 222, 232)
BACKGROUND_COLOR = (251, 252, 254)

COLUMN_LABELS = (
    "Original",
    "Moderate mask change",
    "Moderate generated",
    "Significant mask change",
    "Significant generated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases-per-page", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=420)
    parser.add_argument("--primitive-label", default="")
    parser.add_argument("--filename-prefix", default="")
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    properties = font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"
    )
    return ImageFont.truetype(font_manager.findfont(properties), size=size)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def mask_boundary(mask: np.ndarray, width: int = 7) -> np.ndarray:
    source = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    dilated = np.asarray(source.filter(ImageFilter.MaxFilter(width))) > 0
    eroded = np.asarray(source.filter(ImageFilter.MinFilter(width))) > 0
    return dilated ^ eroded


def overlay_mask(
    original: Image.Image,
    moderate_mask: np.ndarray,
    significant_mask: np.ndarray | None = None,
) -> Image.Image:
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    alpha = 0.34
    overlay[moderate_mask] = (
        (1.0 - alpha) * base[moderate_mask]
        + alpha * np.asarray(MODERATE_COLOR, dtype=np.float32)
    )
    moderate_edge = mask_boundary(moderate_mask)
    overlay[moderate_edge] = np.asarray(MODERATE_COLOR, dtype=np.float32)

    if significant_mask is not None:
        incremental = significant_mask & ~moderate_mask
        overlay[incremental] = (
            (1.0 - alpha) * base[incremental]
            + alpha * np.asarray(SIGNIFICANT_COLOR, dtype=np.float32)
        )
        incremental_edge = mask_boundary(incremental)
        overlay[incremental_edge] = np.asarray(
            SIGNIFICANT_COLOR, dtype=np.float32
        )
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def fit_tile(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def case_images(case_dir: Path) -> list[Image.Image]:
    original = load_rgb(case_dir / "original.png")
    moderate_mask = load_mask(case_dir / "moderate_change_region.png")
    significant_mask = load_mask(case_dir / "significant_change_region.png")
    return [
        original,
        overlay_mask(original, moderate_mask),
        load_rgb(case_dir / "moderate_generated.png"),
        overlay_mask(original, moderate_mask, significant_mask),
        load_rgb(case_dir / "significant_generated.png"),
    ]


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    draw.text(xy, text, font=text_font, fill=fill, anchor="mm")


def draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: int,
    y: int,
    body_font: ImageFont.FreeTypeFont,
) -> None:
    entries = (
        (MODERATE_COLOR, "Moderate changed area"),
        (SIGNIFICANT_COLOR, "Additional significant-only area"),
    )
    widths = []
    for _, label in entries:
        box = draw.textbbox((0, 0), label, font=body_font)
        widths.append(18 + 10 + box[2] - box[0])
    total = sum(widths) + 34
    x = center_x - total // 2
    for (color, label), width in zip(entries, widths):
        draw.rounded_rectangle((x, y - 8, x + 18, y + 10), radius=4, fill=color)
        draw.text((x + 28, y + 1), label, font=body_font, fill=MUTED_COLOR, anchor="lm")
        x += width + 34


def build_case_strip(
    case_dir: Path,
    record: dict,
    *,
    method_name: str,
    output_path: Path,
    tile_size: int,
) -> None:
    gap = 16
    margin = 34
    title_height = 64
    header_height = 46
    info_height = 48
    width = 2 * margin + 5 * tile_size + 4 * gap
    height = title_height + header_height + info_height + tile_size + 34
    sheet = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(sheet)
    title_font = font(27, bold=True)
    header_font = font(20, bold=True)
    info_font = font(20, bold=True)
    small_font = font(16)

    centered_text(
        draw,
        (width / 2, 29),
        method_name,
        text_font=title_font,
        fill=TEXT_COLOR,
    )
    for column, label in enumerate(COLUMN_LABELS):
        x = margin + column * (tile_size + gap) + tile_size / 2
        centered_text(
            draw,
            (x, title_height + header_height / 2),
            label,
            text_font=header_font,
            fill=TEXT_COLOR,
        )
    info = (
        f"#{record['rank']:02d}  {record['display_id']}  |  "
        f"changed area: {record['moderate_fraction'] * 100:.1f}% -> "
        f"{record['significant_fraction'] * 100:.1f}%"
    )
    draw.text(
        (margin, title_height + header_height + info_height / 2),
        info,
        font=info_font,
        fill=TEXT_COLOR,
        anchor="lm",
    )
    draw.text(
        (width - margin, title_height + header_height + info_height / 2),
        record["wsi_id"],
        font=small_font,
        fill=MUTED_COLOR,
        anchor="rm",
    )
    image_y = title_height + header_height + info_height
    for column, image in enumerate(case_images(case_dir)):
        x = margin + column * (tile_size + gap)
        sheet.paste(fit_tile(image, tile_size), (x, image_y))
        draw.rectangle(
            (x, image_y, x + tile_size - 1, image_y + tile_size - 1),
            outline=BORDER_COLOR,
            width=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)


def build_page(
    records: list[dict],
    assets_root: Path,
    *,
    method_name: str,
    page_number: int,
    page_count: int,
    tile_size: int,
) -> Image.Image:
    gap = 14
    margin = 34
    title_height = 72
    header_height = 48
    info_height = 38
    row_gap = 22
    footer_height = 48
    width = 2 * margin + 5 * tile_size + 4 * gap
    height = (
        title_height
        + header_height
        + len(records) * (info_height + tile_size)
        + max(len(records) - 1, 0) * row_gap
        + footer_height
    )
    sheet = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(sheet)
    title_font = font(27, bold=True)
    header_font = font(18, bold=True)
    info_font = font(17, bold=True)
    small_font = font(14)

    draw.text(
        (margin, 30), method_name, font=title_font, fill=TEXT_COLOR, anchor="lm"
    )
    draw.text(
        (width - margin, 30),
        f"Candidates {records[0]['rank']:02d}-{records[-1]['rank']:02d}  |  page {page_number}/{page_count}",
        font=small_font,
        fill=MUTED_COLOR,
        anchor="rm",
    )
    for column, label in enumerate(COLUMN_LABELS):
        x = margin + column * (tile_size + gap) + tile_size / 2
        centered_text(
            draw,
            (x, title_height + header_height / 2),
            label,
            text_font=header_font,
            fill=TEXT_COLOR,
        )

    y = title_height + header_height
    for record in records:
        case_dir = assets_root / record["case_dir"]
        info = (
            f"#{record['rank']:02d}  {record['display_id']}  |  "
            f"changed area {record['moderate_fraction'] * 100:.1f}% -> "
            f"{record['significant_fraction'] * 100:.1f}%"
        )
        draw.text(
            (margin, y + info_height / 2),
            info,
            font=info_font,
            fill=TEXT_COLOR,
            anchor="lm",
        )
        draw.text(
            (width - margin, y + info_height / 2),
            record["wsi_id"],
            font=small_font,
            fill=MUTED_COLOR,
            anchor="rm",
        )
        y += info_height
        for column, image in enumerate(case_images(case_dir)):
            x = margin + column * (tile_size + gap)
            sheet.paste(fit_tile(image, tile_size), (x, y))
            draw.rectangle(
                (x, y, x + tile_size - 1, y + tile_size - 1),
                outline=BORDER_COLOR,
                width=2,
            )
        y += tile_size + row_gap
    draw_legend(
        draw,
        center_x=width // 2,
        y=height - footer_height // 2,
        body_font=small_font,
    )
    return sheet


def write_pdf(page_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = 12 * 72, 10 * 72
    document = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
    document.setTitle(output_path.stem)
    for page_path in page_paths:
        with Image.open(page_path) as image:
            image_width, image_height = image.size
        scale = min(page_width / image_width, page_height / image_height)
        draw_width = image_width * scale
        draw_height = image_height * scale
        document.drawImage(
            ImageReader(str(page_path)),
            (page_width - draw_width) / 2,
            (page_height - draw_height) / 2,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        document.showPage()
    document.save()


def write_index(records: list[dict], output_path: Path) -> None:
    fields = (
        "rank",
        "display_id",
        "sample_id",
        "pair_id",
        "wsi_id",
        "moderate_fraction",
        "significant_fraction",
        "dose_increase",
        "incremental_visible_change",
        "sharpness_ratio",
        "selection_score",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def main() -> None:
    args = parse_args()
    if args.cases_per_page <= 0:
        raise ValueError("cases-per-page must be positive")
    base_method_names = {
        "inpaint": "Local synthesis",
        "cross": "Reference-guided global synthesis",
    }
    for backend, base_method_name in base_method_names.items():
        method_name = (
            f"{args.primitive_label} - {base_method_name}"
            if args.primitive_label
            else base_method_name
        )
        backend_assets = args.assets_root / backend
        records = json.loads(
            (backend_assets / "selection_manifest.json").read_text()
        )
        if args.filename_prefix:
            method_slug = {
                "inpaint": "local_synthesis",
                "cross": "reference_guided_global_synthesis",
            }[backend]
            pdf_name = (
                f"{args.filename_prefix}_{method_slug}_visual_candidates_"
                f"{len(records)}.pdf"
            )
        else:
            pdf_name = {
                "inpaint": "local_synthesis_visual_candidates_20.pdf",
                "cross": (
                    "reference_guided_global_synthesis_visual_candidates_20.pdf"
                ),
            }[backend]
        output = args.output_root / backend
        output.mkdir(parents=True, exist_ok=True)
        write_index(records, output / "candidate_index.csv")

        strips_root = output / "case_strips"
        for record in records:
            build_case_strip(
                backend_assets / record["case_dir"],
                record,
                method_name=method_name,
                output_path=strips_root
                / f"{record['rank']:02d}_{record['display_id']}.png",
                tile_size=args.tile_size,
            )

        chunks = [
            records[index : index + args.cases_per_page]
            for index in range(0, len(records), args.cases_per_page)
        ]
        page_paths = []
        pages_root = output / "pages"
        pages_root.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(chunks, start=1):
            page = build_page(
                chunk,
                backend_assets,
                method_name=method_name,
                page_number=index,
                page_count=len(chunks),
                tile_size=args.tile_size,
            )
            page_path = pages_root / f"page_{index:02d}.png"
            page.save(page_path, optimize=True)
            page_paths.append(page_path)
        write_pdf(page_paths, output / pdf_name)


if __name__ == "__main__":
    main()
