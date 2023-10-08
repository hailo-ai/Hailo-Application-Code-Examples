#!/usr/bin/env python
import argparse
from inspectors.clipping_inspector import ClippingInspector

from inspectors.high_precision_inspector import HighPrecisionInspector
from inspectors.norm_inspector import NormInspector
from inspectors.compression_inspector import CompressionInspector
from inspectors.concatenated_outputs_inspector import ConcatenatedOutputsInspector

from hailo_sdk_common.logger.logger import create_custom_logger
from hailo_sdk_client.runner.client_runner import ClientRunner
from hailo_sdk_client.exposed_definitions import SUPPORTED_HW_ARCHS
from hailo_model_optimization.acceleras.utils.dataset_util import data_to_dataset
from hailo_model_optimization.acceleras.utils.acceleras_definitions import CalibrationDataType


def get_parser():
    """
    Get a parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("har", help="Path to HAR file", type=str)
    parser.add_argument("--dataset", help="Data sample to check the validty of the model", type=str, required=False)
    parser.add_argument("--hw_arch", help="Target HW arch", choices=SUPPORTED_HW_ARCHS, required=False)
    return parser


def parse_arguments(args=None):
    """
    Parse the arguments
    """
    parser = get_parser()
    parsed_args = parser.parse_args(args)
    return parsed_args


def main():
    args = parse_arguments()
    # TODO: wait for logger as a kwarg?
    runner = ClientRunner(hw_arch=args.hw_arch)
    # Override the logger to suppress and warning, and control the log of this logic
    runner._logger = create_custom_logger('diagnostic_client_runner.log')
    runner.load_har(har=args.har)
    if args.dataset:
        dataset, _ = data_to_dataset(args.dataset, CalibrationDataType.auto)
    else:
        dataset = None
    # ClippingInspector(runner, dataset).run()
    NormInspector(runner, dataset).run()
    CompressionInspector(runner, dataset).run()
    ConcatenatedOutputsInspector(runner, dataset).run()
    HighPrecisionInspector(runner, dataset).run()


if __name__ == "__main__":
    main()
