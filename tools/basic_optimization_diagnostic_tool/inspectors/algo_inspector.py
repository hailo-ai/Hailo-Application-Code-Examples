
from inspectors.base_inspector import BaseInspector, InspectorPriority

from hailo_model_optimization.acceleras.utils.acceleras_definitions import (
    ModelOptimizationCommand, PostQuantizationFeature)
from hailo_model_optimization.tools.simple_alls_parser import (
    CommandInfo, parse_model_script)


class PostQuantAlgoInspector(BaseInspector):
    PRIORITY = InspectorPriority.LOW

    def _run(self):
        commands = parse_model_script(self._runner.model_script)
        opt_level = self._runner._sdk_backend.mo_flavor.optimization_level
        features_summary = {}
        explicit_optimization = None
        for cmd_info in commands:
            if isinstance(cmd_info, CommandInfo):
                if cmd_info.command == ModelOptimizationCommand.model_optimization_flavor.value:
                    explicit_optimization = cmd_info.kwargs.get('optimization_level')
                elif cmd_info.command == ModelOptimizationCommand.post_quantization_optimization.value:
                    policy = cmd_info.kwargs.get('policy', 'disabled')
                    features_summary.setdefault(cmd_info.args[0], {})
                    features_summary[cmd_info.args[0]]['enabled'] = policy == 'enabled'
                    if features_summary[cmd_info.args[0]]['enabled']:
                        features_summary[cmd_info.args[0]]['kwargs'] = cmd_info.kwargs
                            
        implicit = explicit_optimization is None
        self._logger.info(f"{'Implicit' if implicit else 'Explicit'} optimization level: {opt_level}")
        for feature in PostQuantizationFeature:
            feature_name = feature.value
            info = features_summary.get(feature_name, {'enabled': False})
            self._logger.info(f"{feature_name} is {'enabled' if info['enabled'] else 'disabled'}")
            if info['enabled']:
                self._logger.debug(f"{feature_name}: {info['kwargs']}")
        if implicit:
            feautres_count = sum(v['enabled'] for v in features_summary.values())
            if opt_level == 0 and feautres_count == 0:
                self._logger.warning("Please apply optimization on a machine with GPU")
            elif opt_level == 1 and feautres_count == 1 and features_summary['bias_correction']['enabled']:
                self._logger.warning("Please apply optimization with 1024 or more samples in the calibration set")
            elif opt_level == 2 and feautres_count == 1 and features_summary['finetune']['enabled']:
                if self.yes_no_prompt("Would you like to set optimization level to 3?"):
                    self._new_commands.append("model_optimization_flavor(optimization_level=3)")
            else:
                self._logger.info("Non-default config with implicit optimization level")
        else:
            if opt_level != 4 and self.yes_no_prompt("Would you like to increase the optimization level? (Optimization will take a longer time)"):
                self._new_commands.append(f"model_optimization_flavor(optimization_level={opt_level+1})")
            else:
                self._logger.info("Please read the user manual and use advanced quantization commands")
