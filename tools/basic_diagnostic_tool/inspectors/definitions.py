from enum import Enum


class InspectorsEnum(Enum):
    NORMALIZATION = "normalization"
    COMPRESSION = "compression"
    OUTPUT_CONCAT = "output_concat"
    HIGH_PRECISIOn = "high_precision"
    CLIPPING = "clipping"
    BATCH_NORM = "batch_norm"
