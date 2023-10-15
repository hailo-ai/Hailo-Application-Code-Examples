from inspectors.batch_norm_inspector import BatchNormInspector
from inspectors.clipping_inspector import ClippingInspector
from inspectors.compression_inspector import CompressionInspector
from inspectors.concatenated_outputs_inspector import ConcatenatedOutputsInspector
from inspectors.definitions import InspectorsEnum
from inspectors.high_precision_inspector import HighPrecisionInspector
from inspectors.norm_inspector import NormInspector


INSPECTORS = {
    InspectorsEnum.NORMALIZATION: NormInspector,
    InspectorsEnum.COMPRESSION: CompressionInspector,
    InspectorsEnum.OUTPUT_CONCAT: ConcatenatedOutputsInspector,
    InspectorsEnum.HIGH_PRECISIOn: HighPrecisionInspector,
    InspectorsEnum.CLIPPING: ClippingInspector,
    InspectorsEnum.BATCH_NORM: BatchNormInspector
}


# Assuming constant item order of dicts in python 3.7
DEFAULT_ORDER = [inspector for inspector in INSPECTORS]
DEFAULT_ORDER.remove(InspectorsEnum.CLIPPING)  # clipping is not ready yet


def run_inspectors(runner, dataset, logger=None, custom_order=None, **kwargs):
    if custom_order is None:
        custom_order = DEFAULT_ORDER
    for inspector in custom_order:
        INSPECTORS[inspector](runner, dataset, logger=logger, **kwargs).run()
