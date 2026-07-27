#!/usr/bin/env python3
"""Plot the mask-edit taxonomy as a three-column alluvial ribbon diagram."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


STRENGTHS = ("Mild", "Moderate", "Significant", "Extra-large")
PRIMITIVES = (
    ("Tumour", "Increase", STRENGTHS),
    ("Tumour", "Decrease", STRENGTHS),
    ("Necrosis", "Increase", STRENGTHS),
    ("Necrosis", "Decrease", STRENGTHS),
    ("Immune infiltrate", "Increase", STRENGTHS[:3]),  # stromal immune
    ("Immune infiltrate", "Increase", STRENGTHS[:3]),  # intratumoral immune
    ("Immune infiltrate", "Decrease", STRENGTHS),
    ("Stroma", "Increase", STRENGTHS),  # desmoplastic stroma
    ("Stroma", "Decrease", STRENGTHS),
)

TISSUE_ORDER = ("Tumour", "Necrosis", "Immune infiltrate", "Stroma")
DIRECTION_ORDER = ("Increase", "Decrease")
STRENGTH_ORDER = STRENGTHS

TISSUE_COLORS = {
    "Tumour": "#F79256",
    "Necrosis": "#FBD1A2",
    "Immune infiltrate": "#7DCFB6",
    "Stroma": "#9CB8C5",
}
DIRECTION_COLORS = {
    "Increase": "#F79256",
    "Decrease": "#7DCFB6",
}
STRENGTH_COLORS = {
    "Mild": "#EEF3F1",
    "Moderate": "#DDEBE5",
    "Significant": "#FBD1A2",
    "Extra-large": "#F7A06C",
}


@dataclass(frozen=True)
class Node:
    x: float
    top: float
    bottom: float
    value: int
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/mask_edit_benchmark_v0_3/figures/"
            "mask_edit_taxonomy_ribbon"
        ),
        help="Output path without a file extension.",
    )
    return parser.parse_args()


def _aggregate_links() -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
]:
    tissue_to_direction: dict[tuple[str, str], int] = defaultdict(int)
    direction_to_strength: dict[tuple[str, str], int] = defaultdict(int)
    for tissue, direction, strengths in PRIMITIVES:
        tissue_to_direction[(tissue, direction)] += len(strengths)
        for strength in strengths:
            direction_to_strength[(direction, strength)] += 1
    return dict(tissue_to_direction), dict(direction_to_strength)


def _node_values(
    links: dict[tuple[str, str], int],
    *,
    side: int,
) -> dict[str, int]:
    values: dict[str, int] = defaultdict(int)
    for endpoints, value in links.items():
        values[endpoints[side]] += value
    return dict(values)


def _layout_column(
    order: tuple[str, ...],
    values: dict[str, int],
    colors: dict[str, str],
    *,
    x: float,
    scale: float,
    gap: float,
    center_y: float = 0.515,
) -> dict[str, Node]:
    total_height = sum(values[name] * scale for name in order)
    total_height += gap * (len(order) - 1)
    cursor = center_y + total_height / 2
    nodes: dict[str, Node] = {}
    for name in order:
        height = values[name] * scale
        nodes[name] = Node(
            x=x,
            top=cursor,
            bottom=cursor - height,
            value=values[name],
            color=colors[name],
        )
        cursor -= height + gap
    return nodes


def _draw_ribbon(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    source_top: float,
    source_bottom: float,
    target_top: float,
    target_bottom: float,
    color: str,
    alpha: float,
) -> None:
    bend = 0.46 * (x1 - x0)
    vertices = [
        (x0, source_top),
        (x0 + bend, source_top),
        (x1 - bend, target_top),
        (x1, target_top),
        (x1, target_bottom),
        (x1 - bend, target_bottom),
        (x0 + bend, source_bottom),
        (x0, source_bottom),
        (x0, source_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
    )


def _draw_link_set(
    ax: plt.Axes,
    links: dict[tuple[str, str], int],
    source_nodes: dict[str, Node],
    target_nodes: dict[str, Node],
    *,
    source_order: tuple[str, ...],
    target_order: tuple[str, ...],
    source_colors: dict[str, str],
    node_width: float,
    scale: float,
    alpha: float,
) -> None:
    source_cursor = {name: source_nodes[name].top for name in source_order}
    target_cursor = {name: target_nodes[name].top for name in target_order}
    for source in source_order:
        for target in target_order:
            value = links.get((source, target), 0)
            if not value:
                continue
            height = value * scale
            source_top = source_cursor[source]
            target_top = target_cursor[target]
            _draw_ribbon(
                ax,
                x0=source_nodes[source].x + node_width / 2,
                x1=target_nodes[target].x - node_width / 2,
                source_top=source_top,
                source_bottom=source_top - height,
                target_top=target_top,
                target_bottom=target_top - height,
                color=source_colors[source],
                alpha=alpha,
            )
            source_cursor[source] -= height
            target_cursor[target] -= height


def _draw_nodes(
    ax: plt.Axes,
    nodes: dict[str, Node],
    order: tuple[str, ...],
    *,
    width: float,
) -> None:
    for name in order:
        node = nodes[name]
        height = node.top - node.bottom
        ax.add_patch(
            Rectangle(
                (node.x - width / 2, node.bottom),
                width,
                height,
                facecolor=node.color,
                edgecolor="#FFFFFF",
                linewidth=1.3,
                zorder=3,
            )
        )
        ax.text(
            node.x,
            (node.top + node.bottom) / 2 + 0.012,
            name,
            ha="center",
            va="center",
            fontsize=11.2,
            fontweight="bold",
            color="#202020",
            zorder=4,
        )
        ax.text(
            node.x,
            (node.top + node.bottom) / 2 - 0.022,
            f"{node.value} combinations",
            ha="center",
            va="center",
            fontsize=8.8,
            color="#404646",
            zorder=4,
        )


def create_figure(output: Path) -> None:
    tissue_links, strength_links = _aggregate_links()
    tissue_values = _node_values(tissue_links, side=0)
    direction_values = _node_values(tissue_links, side=1)
    strength_values = _node_values(strength_links, side=1)

    scale = 0.0195
    node_width = 0.145
    tissue_nodes = _layout_column(
        TISSUE_ORDER,
        tissue_values,
        TISSUE_COLORS,
        x=0.11,
        scale=scale,
        gap=0.034,
    )
    direction_nodes = _layout_column(
        DIRECTION_ORDER,
        direction_values,
        DIRECTION_COLORS,
        x=0.50,
        scale=scale,
        gap=0.065,
    )
    strength_nodes = _layout_column(
        STRENGTH_ORDER,
        strength_values,
        STRENGTH_COLORS,
        x=0.89,
        scale=scale,
        gap=0.034,
    )

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.94, bottom=0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_link_set(
        ax,
        tissue_links,
        tissue_nodes,
        direction_nodes,
        source_order=TISSUE_ORDER,
        target_order=DIRECTION_ORDER,
        source_colors=TISSUE_COLORS,
        node_width=node_width,
        scale=scale,
        alpha=0.48,
    )
    _draw_link_set(
        ax,
        strength_links,
        direction_nodes,
        strength_nodes,
        source_order=DIRECTION_ORDER,
        target_order=STRENGTH_ORDER,
        source_colors=DIRECTION_COLORS,
        node_width=node_width,
        scale=scale,
        alpha=0.42,
    )

    _draw_nodes(ax, tissue_nodes, TISSUE_ORDER, width=node_width)
    _draw_nodes(ax, direction_nodes, DIRECTION_ORDER, width=node_width)
    _draw_nodes(ax, strength_nodes, STRENGTH_ORDER, width=node_width)

    for x, heading in (
        (0.11, "Edit tissue type"),
        (0.50, "Direction"),
        (0.89, "Strength"),
    ):
        ax.text(
            x,
            0.975,
            heading,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#202020",
        )

    ax.text(
        0.5,
        0.025,
        "Ribbon width represents the number of supported edit primitive-strength combinations (34 total).",
        ha="center",
        va="center",
        fontsize=10,
        color="#505656",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 320} if extension == "png" else {}
        fig.savefig(
            output.with_suffix(f".{extension}"),
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    create_figure(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
