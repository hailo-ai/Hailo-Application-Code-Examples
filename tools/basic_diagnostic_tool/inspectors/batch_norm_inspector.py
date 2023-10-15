
import numpy as np
import tensorflow as tf

from inspectors.base_inspector import BaseInspector

from hailo_sdk_common.hailo_nn.hn_definitions import LayerType
from hailo_sdk_client.exposed_definitions import InferenceContext


class BatchNormInspector(BaseInspector):
    """
    Search for batch norm layers in the model.
    Should work with conv-nets
    If a model is parsed from onnx, the batch norm might've been folded automatically,
    and there would be no indication in the nn_model
    TODO: What about transformers?
    TODO: collect pre-act distribution to verify the data is normaliezd?
    """
    MEAN_TH = 5
    STD_TH = 2

    def _run(self):
        simple_run = self._dataset is None
        nn_model = self._runner.get_hn_model()
        fused_bn = self._get_bn_fused_layers(nn_model)
        if len(fused_bn) == 0:
            if simple_run:
                self._logger.warning("The model doesn't have batch norm layers. "
                                     "This means that either the layers of the model aren't normalized correctly, "
                                     "or that the model was batch norm layers were fused during export (from torch)")
                self._logger.info("Dataset was not provided, so data distirbution won't be checked")
                # TODO: collect pre-act distribution and check data
            else:
                outliers = self._get_outliers()
                if outliers:
                    self._logger.warning(f"The model doesn't have batch norm layers. "
                                         f"The following layers had abnormal ratio of distribution outliers: "
                                         f"{list(outliers.keys())}")
                    self._logger.error("This might indicate the model wasn't trained properly, or that the input isn't distributed properly")

    def _get_distributions(self):
        with self._runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            model = self._runner.get_keras_model(ctx)._model
            for layer in model.layers.values():
                layer._debug_mode = True  # experimental feature, collect all inter-op tensors
            axes = np.arange(len(self._dataset.element_spec[0].shape))

            @tf.function
            def infer_pre_act_distribution(data):
                model(data)
                result = {}
                for lname, layer in model.layers.items():
                    if layer.activation_atomic_op is None:
                        continue
                    pre_act = layer._layer_flow.predecessors_sorted(layer.activation_atomic_op.name)[0]
                    tensor = layer.intermidiate_results[pre_act]
                    ch_mean_sample = tf.reduce_mean(tensor, axis=axes)
                    ch_std_sample = tf.math.reduce_std(tensor, axis=axes)
                    result[lname] = (ch_mean_sample, ch_std_sample)
                return result

            ch_mean_by_layer = {}
            ch_std_by_layer = {}
            for data_sample, _ in self._dataset.batch(1):
                result = infer_pre_act_distribution(data_sample)
                for layer, (mean, std) in result.items():
                    ch_mean_by_layer.setdefault(layer, [])
                    ch_std_by_layer.setdefault(layer, [])
                    ch_mean_by_layer[layer].append(mean.numpy())
                    ch_std_by_layer[layer].append(std.numpy())

        ch_mean = {}
        ch_std = {}
        for layer in ch_std_by_layer:
            ch_mean[layer] = np.mean(ch_mean_by_layer[layer], axis=0)
            ch_std[layer] = np.sqrt(np.mean(np.power(ch_std_by_layer[layer], 2), axis=0))
        return ch_mean, ch_std

    def _get_outliers(self, mean_ratio_th=0.05, std_ratio_th=0.05):
        per_layer_ch_mean, per_layer_ch_std = self._get_distributions()
        outliers = {}
        for layer in per_layer_ch_mean:
            ch_means = per_layer_ch_mean[layer]
            ch_stds = per_layer_ch_std[layer]
            mean_outliers_ratio = np.sum(ch_means > self.MEAN_TH) / len(ch_means)
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
