from .mask_utils import (
    # Constants
    COLOR_MAP, NUCLEI_CLASSES, TISSUE_NAMES, NUCLEI_NAMES,
    NUM_TISSUE, NUM_NUCLEI, NUM_CANCER_TYPES,
    NUCLEI_RAW_TO_INDEX, NUCLEI_INDEX_TO_RAW,
    # Embedding dimensions (AD-4)
    TISSUE_EMB_DIM, CELL_EMB_DIM, CANCER_EMB_DIM, PROBNET_IN_CH,
    # Layered storage I/O (AD-1)
    load_tissue_mask, load_nuclei_mask, save_nuclei_mask,
    # RGB <-> class conversion (legacy / visualization)
    rgb_to_class_map, class_map_to_rgb,
    split_tissue_nuclei,
    index_to_rgb, overlay,
)
