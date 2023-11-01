from inspectors.base_inspector import BaseInspector, InspectorPriority

from hailo_model_optimization.acceleras.utils.acceleras_definitions import ModelOptimizationCommand, MOConfigCommand
from hailo_model_optimization.tools.simple_alls_parser import CommandInfo, parse_model_script


class CompressionInspector(BaseInspector):
    PRIORITY = InspectorPriority.MEDIUM

    def _run(self):
        compressed_ratio = self.get_compressed_ratio()
        if compressed_ratio == 0:
            return
        self._logger.info(f"The model has {compressed_ratio*100:.2f}% 4bit compression")
        self.check_compression(compressed_ratio)

    def get_compressed_ratio(self):
        nn_model = self._runner.get_hn_model()
        total = 0
        compressed = 0
        for layer in nn_model.stable_toposort():
            total += layer.weights
            is_4bit = layer.precision_config.precision_mode.value.split('_')[1] == 'w4'
            if is_4bit:
                compressed += layer.weights
        compressed_ratio = compressed / total
        return compressed_ratio

    def check_compression(self, compressed_ratio):
        commands = parse_model_script(self._runner.model_script)
        compression_level = self._runner._sdk_backend.mo_flavor.compression_level

        explicit_compression = None
        desired_ratio = None
        for cmd_info in commands:
            if isinstance(cmd_info, CommandInfo):
                if cmd_info.command == ModelOptimizationCommand.model_optimization_flavor.value:
                    explicit_compression = cmd_info.kwargs.get('compression_level')
                elif cmd_info.command == ModelOptimizationCommand.model_optimization_config.value:
                    if cmd_info.args[0] == MOConfigCommand.compression_params.value:
                        desired_ratio = cmd_info.kwargs.get('auto_4bit_weights_ratio')
        if explicit_compression is not None:
            explicit_compression = int(explicit_compression)
        if desired_ratio is not None:
            desired_ratio = float(desired_ratio)

        if compressed_ratio == 0:
            self._logger.info("Model has no compression")
        elif explicit_compression is None:
            self._logger.warning("Model might have implicit compression")
            # TODO (Optional): Replace it with multiple log message
            # TODO (Optional): Do we need to inform the user about both alternatives?
            self._logger.info("Consider disabling compression")
            should_add_command = self.yes_no_prompt("Would you like to disable compression?")
            if should_add_command:
                self._new_commands.append("model_optimization_flavor(compression_level=0)")
                self._new_commands.append("model_optimization_config(compression_params, auto_4bit_weights_ratio=0)")
        elif explicit_compression == compression_level:
            self._logger.info(f"Model has explicit compression_level: {explicit_compression}")
        else:
            # Unexpected flow
            self._logger.info(
                f"Model has explicit compression_level: {explicit_compression}, "
                f"but compression level is {compression_level}")
