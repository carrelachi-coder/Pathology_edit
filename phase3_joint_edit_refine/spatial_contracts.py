"""Shared deterministic spatial contracts for cell-layout morphology."""

SMALL_CLUSTER_TARGET_FOCUS_COUNT = 3
SMALL_CLUSTER_MINIMUM_FOCUS_SIZE = 2
SMALL_CLUSTER_MAXIMUM_FOCUS_SIZE = 4

# Members of one budding-like focus must remain visibly adjacent.  Separate
# foci use the wider independent-focus graph distance below.
SMALL_CLUSTER_MEMBER_RADIUS_DIAMETERS = 1.05
SMALL_CLUSTER_WITHIN_FOCUS_LINK_DIAMETERS = 1.35
SMALL_CLUSTER_MAXIMUM_FOCUS_DIAMETER_DIAMETERS = 2.25
SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS = 2.25

# The edit is a localized invasive-front hotspot, not a second scatter field.
# The existing primitive span floor remains active; this upper bound prevents
# the foci from being distributed around the full peritumoral annulus.
SMALL_CLUSTER_MAXIMUM_HOTSPOT_SPAN_DIAMETERS = 7.5
