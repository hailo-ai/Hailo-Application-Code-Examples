from abc import ABC, abstractmethod

from hailo_sdk_common.logger.logger import create_custom_logger
from hailo_sdk_client.runner.client_runner import ClientRunner


class BaseInspector(ABC):
    """
    Base inspector class for the basic diagnotic tool
    """
    def __init__(self, runner: ClientRunner, dataset, logger=None) -> None:
        self._runner = runner
        self._dataset = dataset
        if logger is None:
            self._logger = create_custom_logger(log_path="normalization_checker.log", console=True)
        else:
            self._logger = logger

    @abstractmethod
    def run(self):
        pass
