
from inspectors.base_inspector import BaseInspector, InspectorPriority

from hailo_model_optimization.acceleras.utils.acceleras_definitions import (
    ModelOptimizationCommand)
from hailo_model_optimization.tools.simple_alls_parser import (
    CommandInfo, parse_model_script)


class PreQuantAlgoInspector(BaseInspector):
    PRIORITY = InspectorPriority.LOW

    def _run(self):
        commands = parse_model_script(self._runner.model_script)
        calibset_size = None
        for cmd_info in commands:
            if isinstance(cmd_info, CommandInfo):
                if cmd_info.command == ModelOptimizationCommand.model_optimization_config.value:
                    calibset_size = cmd_info.kwargs.get('calibset_size', calibset_size)
        if calibset_size is None:
            calibset_size = 64
        else:
            calibset_size = int(calibset_size)
        if calibset_size > 64:
            self._logger.warning(f"Calibset size is {calibset_size}, which is greater than the default value of 64. "
                                 f"Increasing the calibration size affects only the stats collection, and might affect accuracy. "
                                 f"If the intention was to increase the dataset size of a specific algorithm, "
                                 f"please remove the command and read the desired algorithm's documentation section.")
        elif calibset_size < 64:
            self._logger.info(f"Calibset size is {calibset_size}, which is lower than the default value of 64.")
