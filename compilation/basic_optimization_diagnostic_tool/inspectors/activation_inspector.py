import inspectors.messages as msg 
from inspectors.base_inspector import BaseInspector, InspectorPriority

from hailo_sdk_common.hailo_nn.hn_definitions import ActivationType, LayerType

class ActivationInspector(BaseInspector):
    PRIORITY = InspectorPriority.LOW

    def _run(self):
        silu_layers = self.get_silu_layers()
        if len(silu_layers) != 0:
            self._logger.warning(
                "Some layers in the model has SiLU activation, these layers might reduce the accuracy. "
                "If possible, consider retraining the model with ReLU activation.")
            self._logger.debug(f"SiLU layers: {silu_layers}")  # TODO: use original names?
        
        swish_blocks = self.get_swish_blocks()
        if len(swish_blocks) != 0:
            self._logger.warning(
                "Original model might've had swish activation with beta != 1. "
                "That activation cannot be represented natively in Hailo, and might cause lower accuracy or lower FPS. "
                "If possible, consider retraining the model with ReLU activation. "
                "(or with SiLU activation, but accuracy might still be affected)"
            )
            self._logger.debug(f"Swish blocks: {swish_blocks}")  # TODO: use original names?
        
    
    def get_silu_layers(self):
        def silu_filter(hn_layer):
            activation = getattr(hn_layer, 'activation', None)
            return activation == ActivationType.silu
                
        hn_model = self._runner.get_hn_model()        
        layers_with_silu = list(filter(silu_filter, hn_model.stable_toposort()))
        return layers_with_silu

    def get_swish_blocks(self):
        hn_model = self._runner.get_hn_model()
        swish_blocks = []
        for layer in hn_model.stable_toposort():
            # Step 1: search for norm with sigmoid
            if layer.op != LayerType.normalization:
                continue
            if getattr(layer, 'activation', None) != ActivationType.sigmoid:
                continue
            norm_layer = layer

            # Step 2: search for ew_mult
            ew_mult_cand = list(hn_model.successors(norm_layer))
            if len(ew_mult_cand) != 1 or ew_mult_cand[0].op != LayerType.ew_mult:
                continue
            ew_mult = ew_mult_cand[0]
            
            # Step 3: make sure the predecessor of the ew_mult and norm are the same.
            ew_mult_preds = hn_model.predecessors(ew_mult)
            origin_cand = list(hn_model.predecessors(norm_layer))
            if len(ew_mult_preds) != 2 or len(origin_cand) != 1:
                continue
            origin = origin_cand[0]
            origin_act = getattr(origin, 'activation', ActivationType.linear)
            if (origin in ew_mult_preds) and (norm_layer in ew_mult_preds) and (origin_act == ActivationType.linear):
                swish_blocks.append((origin, norm_layer, ew_mult))
        return swish_blocks