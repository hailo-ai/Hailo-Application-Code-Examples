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

    def run(self):
        self._logger.info(f"Running inspector module: {self.name}")
        self._run()
        # TODO: consider adding error / warning counters?
        self._logger.debug(f"Inspector module has finished ({self.name})")

    @abstractmethod
    def _run(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
