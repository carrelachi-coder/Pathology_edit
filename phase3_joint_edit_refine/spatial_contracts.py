"""Shared deterministic spatial contracts for cell-layout morphology."""

from math import ceil, sqrt

SMALL_CLUSTER_TARGET_FOCUS_COUNT = 2
SMALL_CLUSTER_MINIMUM_FOCUS_SIZE = 2
SMALL_CLUSTER_MAXIMUM_FOCUS_SIZE = 4

# Members of one budding-like focus must remain visibly adjacent.  Separate
# foci use the wider independent-focus graph distance below.
SMALL_CLUSTER_MEMBER_RADIUS_DIAMETERS = 1.05
COMPACT_PAIR_SMALL_CLUSTER_MEMBER_SPACING_DIAMETERS = 0.90
SMALL_CLUSTER_WITHIN_FOCUS_LINK_DIAMETERS = 1.35
SMALL_CLUSTER_MAXIMUM_FOCUS_DIAMETER_DIAMETERS = 2.25
SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS = 1.6

BREAST_SMALL_CLUSTER_TARGET_FOCUS_COUNT = 3
BREAST_SMALL_CLUSTER_MINIMUM_FOCUS_SIZE = 3
BREAST_SMALL_CLUSTER_MEMBER_SPACING_DIAMETERS = 0.75
BREAST_SMALL_CLUSTER_WITHIN_FOCUS_LINK_DIAMETERS = 1.10
BREAST_SMALL_CLUSTER_MAXIMUM_FOCUS_DIAMETER_DIAMETERS = 1.35
BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS = (
    SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS
    + sqrt(2.0) * BREAST_SMALL_CLUSTER_MEMBER_SPACING_DIAMETERS
    + 0.10
)
GLAS_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS = 2.4
SCATTER_MINIMUM_CENTER_SEPARATION_DIAMETERS = 2.25
CELL_EFFECT_MINIMUM_INTER_FOCUS_SEPARATION_DIAMETERS = 3.0
CELL_EFFECT_WITHIN_FOCUS_LINK_DIAMETERS = 1.5
CELL_EFFECT_MAXIMUM_FOCUS_DIAMETER_DIAMETERS = 2.5

# The edit is a localized invasive-front hotspot, not a second scatter field.
# Use both a scale-relative and patch-scale ceiling: semantic nuclei can have
# a large nominal diameter, so a diameter-only ceiling can still admit an
# almost full-patch diagonal layout on a 256 px crop.
SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_DIAMETERS = 4.0
SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_PX = 128.0
BREAST_SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_DIAMETERS = 6.5
BREAST_SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_PX = 192.0
PERITUMORAL_CAPACITY_FALLBACK_MAXIMUM_DIAMETERS = 2.5


def peritumoral_outer_maximum_px(
    *,
    configured_maximum_px: int,
    nominal_nucleus_diameter_px: float,
    capacity_fallback_enabled: bool,
) -> int:
    """Keep the reviewed annulus first; widen failed sparse cases by scale."""

    configured = max(1, int(configured_maximum_px))
    if not capacity_fallback_enabled:
        return configured
    return max(
        configured,
        int(
            ceil(
                PERITUMORAL_CAPACITY_FALLBACK_MAXIMUM_DIAMETERS
                * max(1.0, float(nominal_nucleus_diameter_px))
            )
        ),
    )


def small_cluster_maximum_hotspot_span_px(
    nominal_nucleus_diameter_px: float,
    minimum_effect_span_px: float = 0.0,
    *,
    compact_breast: bool = False,
) -> float:
    """Return the strict relative/absolute localized-hotspot ceiling."""

    diameter_ceiling = (
        BREAST_SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_DIAMETERS
        if compact_breast
        else SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_DIAMETERS
    )
    pixel_ceiling = (
        BREAST_SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_PX
        if compact_breast
        else SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_PX
    )

    return min(
        max(
            diameter_ceiling * max(1.0, float(nominal_nucleus_diameter_px)),
            1.25 * max(0.0, float(minimum_effect_span_px)),
        ),
        pixel_ceiling,
    )


def small_cluster_maximum_focus_diameter_px(
    nominal_nucleus_diameter_px: float,
    *,
    member_spacing_diameters: float = SMALL_CLUSTER_MEMBER_RADIUS_DIAMETERS,
    maximum_focus_diameter_diameters: float = (
        SMALL_CLUSTER_MAXIMUM_FOCUS_DIAMETER_DIAMETERS
    ),
    compact_template: bool = False,
) -> float:
    """Mirror the executor's compact square-template raster diameter."""

    nominal = max(1.0, float(nominal_nucleus_diameter_px))
    raster_member_spacing = max(
        4,
        round(nominal * member_spacing_diameters),
    )
    raster_diameter = (
        sqrt(2.0) * (raster_member_spacing + sqrt(0.5))
        if compact_template
        else 2.0 * (raster_member_spacing + sqrt(0.5))
    )
    return max(
        maximum_focus_diameter_diameters * nominal,
        raster_diameter,
    )


def breast_small_cluster_within_focus_link_px(
    nominal_nucleus_diameter_px: float,
) -> float:
    """Return the compact-template link with its four-pixel raster floor."""

    nominal = max(1.0, float(nominal_nucleus_diameter_px))
    raster_member_spacing = max(
        4,
        round(nominal * BREAST_SMALL_CLUSTER_MEMBER_SPACING_DIAMETERS),
    )
    return max(
        BREAST_SMALL_CLUSTER_WITHIN_FOCUS_LINK_DIAMETERS * nominal,
        raster_member_spacing + sqrt(0.5),
    )


def small_cluster_member_spacing_px(
    nominal_nucleus_diameter_px: float,
) -> float:
    """Mirror the generic small-cluster template's raster spacing floor."""

    nominal = max(1.0, float(nominal_nucleus_diameter_px))
    return float(
        max(4, round(nominal * SMALL_CLUSTER_MEMBER_RADIUS_DIAMETERS))
    )


def small_cluster_within_focus_link_px(
    nominal_nucleus_diameter_px: float,
) -> float:
    """Keep the focus graph reachable when the four-pixel floor is active."""

    nominal = max(1.0, float(nominal_nucleus_diameter_px))
    return max(
        SMALL_CLUSTER_WITHIN_FOCUS_LINK_DIAMETERS * nominal,
        small_cluster_member_spacing_px(nominal) + sqrt(0.5),
    )


def small_cluster_minimum_anchor_separation_px(
    nominal_nucleus_diameter_px: float,
) -> float:
    """Separate two complete raster-ring foci without graph bridging."""

    nominal = max(1.0, float(nominal_nucleus_diameter_px))
    return (
        2.0 * small_cluster_member_spacing_px(nominal)
        + max(
            small_cluster_within_focus_link_px(nominal),
            SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS * nominal,
        )
    )
