
import numpy as np
import tensorflow as tf

import inspectors.messages as msg
from inspectors.base_inspector import BaseInspector, InspectorPriority

from hailo_sdk_common.hailo_nn.hn_definitions import LayerType
from hailo_sdk_client.exposed_definitions import InferenceContext
from hailo_model_optimization.acceleras.hailo_layers.base_hailo_layer import BaseHailoLayer
from hailo_model_optimization.acceleras.hailo_layers.base_hailo_conv import BaseHailoConv


class BatchNormInspector(BaseInspector):
    """
    Search for batch norm layers in the model.
    Should work with conv-nets
    If a model is parsed from onnx, the batch norm might've been folded automatically,
    and there would be no indication in the nn_model
    TODO: What about transformers?
    TODO: collect pre-act distribution to verify the data is normaliezd?
    """
    PRIORITY = InspectorPriority.HIGH
    MEAN_TH = 5
    STD_TH = 3

    def _run(self):
        simple_run = self._dataset is None
        nn_model = self._runner.get_hn_model()
        fused_bn = self._get_bn_fused_layers(nn_model)
        if len(fused_bn) == 0:
            if simple_run:
                self._logger.warning("The model doesn't have batch norm layers. "
                                     "This means that either the layers of the model aren't normalized correctly, "
                                     "or that the model was batch norm layers were fused during export (from torch)")
                self._logger.warning(f"Skipping distribution check, {msg.SKIP_NO_DATASET}")
                # TODO: collect pre-act distribution and check data
            else:
                outliers = self._get_outliers()
                if outliers:
                    self._logger.warning(f"The model might not have been trained with batch norm layers. "
                                         f"Some layers had abnormal distribution. "
                                         f"Outliers: {list(outliers.keys())}")
                    self._logger.error("This might indicate the model wasn't trained batch norm, or that the input isn't distributed properly. "
                                       "If the model wasn't trained with batch norm, please consider retraining the model with batch norm.")

    def _get_distributions(self):
        with self._runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            model = self._runner.get_keras_model(ctx)._model
            for layer in model.layers.values():
                layer._debug_mode = True  # experimental feature, collect all inter-op tensors
            axes = np.arange(3)  # Assumes 4 dimensions with batch, and channels is last channel

            @tf.function
            def infer_pre_act_distribution(data):
                model(data)
                result = {}
                for lname, layer in model.layers.items():
                    if not isinstance(layer, BaseHailoLayer) or not isinstance(layer, BaseHailoConv):
                        continue
                    pre_act = layer._layer_flow.predecessors_sorted(layer.activation_atomic_op.name)[0]
                    tensor = layer.intermidiate_results[pre_act]
                    ch_mean_sample = tf.reduce_mean(tensor, axis=axes)
                    ch_std_sample = tf.math.reduce_std(tensor, axis=axes)
                    ch_max_sample = tf.reduce_max(tensor, axis=axes)
                    ch_min_sample = tf.reduce_min(tensor, axis=axes)
                    result[lname] = (ch_mean_sample, ch_std_sample, ch_min_sample, ch_max_sample)
                return result

            ch_mean_by_layer = {}
            ch_std_by_layer = {}
            ch_min_by_layer = {}
            ch_max_by_layer = {}
            for data_sample, _ in self._dataset.batch(1):
                result = infer_pre_act_distribution(data_sample)
                for layer, (mean, std, min_, max_) in result.items():
                    ch_mean_by_layer.setdefault(layer, [])
                    ch_std_by_layer.setdefault(layer, [])
                    ch_min_by_layer.setdefault(layer, [])
                    ch_max_by_layer.setdefault(layer, [])
                    ch_mean_by_layer[layer].append(mean.numpy())
                    ch_std_by_layer[layer].append(std.numpy())
                    ch_min_by_layer[layer].append(min_.numpy())
                    ch_max_by_layer[layer].append(max_.numpy())

        ch_mean = {}
        ch_std = {}
        ch_min = {}
        ch_max = {}
        for layer in ch_std_by_layer:
            ch_min[layer] = np.min(ch_min_by_layer[layer], axis=0)
            ch_max[layer] = np.max(ch_max_by_layer[layer], axis=0)
            ch_mean[layer] = np.mean(ch_mean_by_layer[layer], axis=0)
            ch_std[layer] = np.sqrt(np.mean(np.power(ch_std_by_layer[layer], 2), axis=0))
        return ch_mean, ch_std, ch_min, ch_max

    def _get_outliers(self, mean_ratio_th=0.05, std_ratio_th=0.05):
        result = self._get_distributions()
        per_layer_ch_mean = result[0]
        per_layer_ch_std = result[1]
        
        outliers = {}
        for layer in per_layer_ch_mean:
            ch_means = per_layer_ch_mean[layer]
            ch_stds = per_layer_ch_std[layer]
            self._logger.debug(f"{layer}, mean: {ch_means}, std: {ch_stds}")
            mean_outliers_ratio = np.sum(np.abs(ch_means) > self.MEAN_TH) / len(ch_means)
            std_outliers_ratio = np.sum(ch_stds > self.STD_TH) / len(ch_stds)
            if mean_outliers_ratio > mean_ratio_th or std_outliers_ratio > std_ratio_th:
                outliers[layer] = (mean_outliers_ratio, std_outliers_ratio)
        return outliers

    def _get_bn_fused_layers(self, nn_model):
        fused_bn = []
        for layer in nn_model.stable_toposort():
            if layer.bn_enabled:
                fused_bn.append(layer.name)
        return fused_bn

    def _get_bn_layers(self, nn_model):
        pure_bn = []
        for layer in nn_model.stable_toposort():
            if layer.op == LayerType.batch_norm:
                pure_bn.append(layer.name)
        return pure_bn
