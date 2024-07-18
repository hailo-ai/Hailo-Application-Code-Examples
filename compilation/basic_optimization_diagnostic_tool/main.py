#!/usr/bin/env python
try:
    import hailo_sdk_client
except ModuleNotFoundError:
    print("ERROR: Hailo SDK is not installed")
    exit(2)
else:
    sdk_version = hailo_sdk_client.__version__.split('.')
    if sdk_version[0] != '3' or (sdk_version[1] != '25' and (sdk_version[1] != '26'):
        print(f"ERROR: Incompatible Hailo sdk version. Expected: 3.25, Version: {hailo_sdk_client.__version__}")
        exit(2)
    elif len(sdk_version) > 3:
        print(f"WARNING: Hailo sdk version {hailo_sdk_client.__version__} are you using a released version?")


import argparse

from inspectors_manager import INSPECTORS_BY_NAME  # this import takes a sec~

from hailo_sdk_client.exposed_definitions import SUPPORTED_HW_ARCHS, States
from logger import init_logger



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
        required=True)
    parser.add_argument(
        "-c", "--data-count",
        help="Number of data samples that will be used in the inspectors",
        type=int,
        required=False,
        default=64
    )
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
    parser.add_argument('--no-interactive', dest='interactive', action='store_false')

    parser.add_argument(
        "--output-model-script",
        help="Create output model script with new recommended commands in the provided path.",
        type=str,
        default=""
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
    advanced_parser.add_argument(
        "--no-debug",
        help="By default, an additional hidden log file with debug info is created. "
             "If this flag is present, the debug log won't be created.",
        action="store_false",
        dest="debug_log"
    )
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
    suppress_logger = create_custom_logger('diagnostic_client_runner.log')
    runner._logger = suppress_logger
    runner.load_har(har=args.har)
    if args.dataset:
        dataset, _ = data_to_dataset(args.dataset, 'auto')
    else:
        dataset = None
    return runner, dataset.take(args.data_count)


def main(args):
    from inspectors_manager import run_inspectors
    logger = init_logger("optimization_diagnostic", args.log_path, debug_log=args.debug_log)

    logger.info(f"Running optimization diagnostic tool {args.har}")
    logger.debug(f"Arguments: {vars(args)}")
    runner, dataset = _data_initialization(args)
    if runner.state not in {States.COMPILED_MODEL, States.QUANTIZED_MODEL}:
        logger.error("Model is not Quantized.")
        return 1

    run_inspectors(runner, dataset, interactive=args.interactive,
                   output_model_script=args.output_model_script,
                   custom_order=args.order, data_count=args.data_count,
                   logger=logger)
    return 0

if __name__ == "__main__":
    args = parse_arguments()
    retval = main(args)
    exit(retval)
