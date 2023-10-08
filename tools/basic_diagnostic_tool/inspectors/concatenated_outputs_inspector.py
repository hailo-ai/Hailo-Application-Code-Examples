

from inspectors.base_inspector import BaseInspector
from hailo_model_optimization.acceleras.utils.acceleras_definitions import LayerType


class ConcatenatedOutputsInspector(BaseInspector):
    def _run(self):
        self.search_concatenated_outputs()

    def search_concatenated_outputs(self):
        nn_model = self._runner.get_hn_model()
        output_layers = nn_model.get_output_layers()
        for output_layer in output_layers:
            real_output_layers = nn_model.get_real_output_layer(output_layer)
            concat_layers = list(filter(self._is_concat, real_output_layers))
            for cc_layer in concat_layers:
                preds = list(nn_model.predecessors(cc_layer))
                pred_names = [pred.name for pred in preds]
                orig_names = [pred.original_names for pred in preds]
                self._logger.warning(f"Output layer {output_layer} derives from concat layer {cc_layer}. "
                                     f"Having concatenated output might reduce the accuracy of the model. "
                                     f"Consider using {pred_names} as model outputs instead "
                                     f"(Original names: {orig_names})")

    @staticmethod
    def _is_concat(layer):
        return LayerType(layer.op.value) == LayerType.CONCAT