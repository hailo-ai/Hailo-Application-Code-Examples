

from inspectors.base_inspector import BaseInspector, InspectorPriority
from hailo_sdk_common.hailo_nn.hn_definitions import LayerType


class ConcatenatedOutputsInspector(BaseInspector):
    PRIORITY = InspectorPriority.MEDIUM

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
        return layer.op == LayerType.concat
