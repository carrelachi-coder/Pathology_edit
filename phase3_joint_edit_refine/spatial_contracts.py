"""Shared deterministic spatial contracts for cell-layout morphology."""

from math import sqrt

SMALL_CLUSTER_TARGET_FOCUS_COUNT = 2
SMALL_CLUSTER_MINIMUM_FOCUS_SIZE = 2
SMALL_CLUSTER_MAXIMUM_FOCUS_SIZE = 4

# Members of one budding-like focus must remain visibly adjacent.  Separate
# foci use the wider independent-focus graph distance below.
SMALL_CLUSTER_MEMBER_RADIUS_DIAMETERS = 1.05
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
SCATTER_MINIMUM_CENTER_SEPARATION_DIAMETERS = 2.25

# The edit is a localized invasive-front hotspot, not a second scatter field.
# Use both a scale-relative and patch-scale ceiling: semantic nuclei can have
# a large nominal diameter, so a diameter-only ceiling can still admit an
# almost full-patch diagonal layout on a 256 px crop.
SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_DIAMETERS = 4.0
SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_PX = 128.0


def small_cluster_maximum_hotspot_span_px(
    nominal_nucleus_diameter_px: float,
    minimum_effect_span_px: float = 0.0,
) -> float:
    """Return the strict relative/absolute localized-hotspot ceiling."""

    return min(
        max(
            SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_DIAMETERS
            * max(1.0, float(nominal_nucleus_diameter_px)),
            1.25 * max(0.0, float(minimum_effect_span_px)),
        ),
        SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_PX,
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
