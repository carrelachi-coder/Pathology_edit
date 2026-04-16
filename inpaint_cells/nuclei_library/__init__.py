from .library import (
    NucleiLibrary,
    poisson_disk_sampling,
    # Legacy (raw IDs 101-105, combined tissue+nuclei map)
    place_nucleus,
    fill_nuclei_in_region,
    # Layered storage (AD-1, internal index 0-5, separate tissue/nuclei maps)
    place_nucleus_layered,
    fill_nuclei_in_region_layered,
)
