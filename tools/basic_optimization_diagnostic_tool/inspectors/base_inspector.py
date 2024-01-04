import shutil
from abc import ABC, abstractmethod
from enum import Enum

from inspectors.cli import yes_no_prompt

from hailo_sdk_common.logger.logger import create_custom_logger
from hailo_sdk_client.runner.client_runner import ClientRunner


class InspectorPriority(int, Enum):
    UNSET = 0
    LOW = 10
    MEDIUM = 20
    HIGH = 30


class BaseInspector(ABC):
    """
    Base inspector class for the basic diagnotic tool
    """
    PRIORITY = InspectorPriority.UNSET

    def __init__(self, runner: ClientRunner, dataset, interactive=True, logger=None, **kwargs) -> None:
        self._runner = runner
        self._dataset = dataset
        self._interactive = interactive
        if logger is None:
            self._logger = create_custom_logger(log_path="diagnostic_tool.log", console=True)
        else:
            self._logger = logger
        self._new_commands = []
        self._additional_info = {}

    def should_skip(self) -> str:
        return ""
    
    def add_info(self, **kwargs):
        self._additional_info.update(kwargs)

    def get_new_commands(self):
        return self._new_commands

    def run(self):
        self._logger.info(f"Module: {self.name}")
        skip_reason = self.should_skip()
        if skip_reason:
            self._logger.warning(f"Skipping {self.name}, {skip_reason}")
            self.print_line_seperator()
            return
        self._run()
        # TODO: consider adding error / warning counters?
        self._logger.debug(f"Inspector module has finished ({self.name})")
        self.print_line_seperator()

    @abstractmethod
    def _run(self):
        pass
    
    def print_line_seperator(self, c="=", length=None):
        if length is None:
            term_size = shutil.get_terminal_size(fallback=(120, 50))
            length = term_size.columns
        print(c * length)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def yes_no_prompt(self, msg, default=True):
        if self._interactive:
            return yes_no_prompt(msg, default=default)
        else:
            return default