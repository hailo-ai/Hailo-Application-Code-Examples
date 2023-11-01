#!/usr/bin/env python
import argparse

from inspectors_manager import INSPECTORS_BY_NAME  # this import takes a sec~

from hailo_sdk_client.exposed_definitions import SUPPORTED_HW_ARCHS


def get_parser():
    """
    Get a parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "har",
        help="Path to HAR file",
        type=str)
    parser.add_argument(
        "-d", "--dataset",
        help="Calibration Dataset, npy / npz file formats",
        type=str,
        required=False)
    parser.add_argument(
        "-a", "--hw_arch",
        help="Target HW arch {%(choices)s}",
        choices=SUPPORTED_HW_ARCHS,
        required=False,
        metavar='HW_ARCH')
    parser.add_argument(
        "--log-path",
        help="Default path: %(default)s",
        type=str,
        default="diagnostic_tool.log"
    )

    advanced_parser = parser.add_argument_group("Advanced", description="Advanced diagnostic tool features")
    inspectors = [name for name in INSPECTORS_BY_NAME]
    advanced_parser.add_argument(
        "--order",
        help="Choose which inspectors to run and set a custom order {%(choices)s}",
        choices=inspectors,
        required=False,
        metavar='INSPECTOR',
        nargs="+")
    return parser


def parse_arguments(args=None):
    """
    Parse the arguments
    """
    parser = get_parser()
    parsed_args = parser.parse_args(args)
    return parsed_args


def _data_initialization(args):
    from hailo_sdk_client.runner.client_runner import ClientRunner
    from hailo_sdk_common.logger.logger import create_custom_logger
    from hailo_model_optimization.acceleras.utils.dataset_util import data_to_dataset

    runner = ClientRunner(hw_arch=args.hw_arch)
    # Override the logger to suppress and warning, and control the log of this logic
    # TODO: remove this assignment once ClientRunner supports logger as a kwarg
    runner._logger = create_custom_logger('diagnostic_client_runner.log')
    runner.load_har(har=args.har)
    if args.dataset:
        dataset, _ = data_to_dataset(args.dataset, 'auto')
    else:
        dataset = None
    return runner, dataset


def main(args):
    from inspectors_manager import run_inspectors
    from hailo_sdk_common.logger.logger import create_custom_logger

    runner, dataset = _data_initialization(args)
    logger = create_custom_logger(log_path=args.log_path, console=True)
    run_inspectors(runner, dataset, logger=logger)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
