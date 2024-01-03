from dataclasses import dataclass
from enum import Enum
from typing import List

import networkx as nx
import numpy as np
import tensorflow as tf

import inspectors.messages as msg
from inspectors.base_inspector import BaseInspector, InspectorPriority

from hailo_sdk_client.runner.client_runner import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
from hailo_sdk_common.hailo_nn.hn_definitions import LayerType
from hailo_model_optimization.tools.simple_alls_parser import CommandInfo, parse_model_script



class MeasuredLayerType(Enum):
    NORM_LAYER = "norm_layer"
    NORM_INPUT_LAYER = "normalized_input_layer"
    NON_NORM_INPUT_LAYER = "not_normalized_input_layer"


@dataclass
class MeasuredLayer:
    name: str
    type_: MeasuredLayerType
    
    def __str__(self) -> str:
        return f"{self.name}, {self.type_.value}"


class NormInspector(BaseInspector):
    """
    Check the normalization layers in the model.
    """
    PRIORITY = InspectorPriority.HIGH

    NORMALIZED_MEAN_TH_FLEX = (-2, 5)  # should either be around 0 or around 0.5
    NORMALIZED_MEAN_TH_STRICT = (-0.5, 1)  # should either be around 0 or around 0.5
    NORMALIZED_STD_TH_FLEX = (0.1, 5)  # should either be around 1 or ?
    NORMALIZED_STD_TH_STRICT = (0.1, 2)  # should either be around 1 or ?

    def __init__(self, runner: ClientRunner, dataset, interactive=True, logger=None, **kwargs) -> None:
        super().__init__(runner, dataset, interactive, logger, **kwargs)
        self._nn_model = runner.get_hn_model()

    def _run(self):
        measured_layers = self.check_normailzation_layer()
        self.check_is_input_data_normalized(measured_layers)

    def check_normailzation_layer(self) -> List[MeasuredLayer]:
        """
        Basic sanity, check if the model has a normalization layer.
        """
        commands = parse_model_script(self._runner.model_script)
        norm_layers_from_alls = {command.return_val[0].split('/')[-1]: command.args  # Get layer name without prefix
                                 for command in commands
                                 if isinstance(command, CommandInfo) and command.command == 'normalization'}
        measured_layers = self._get_layers_to_sample(norm_layers_from_alls)
        params = self._runner.get_params_fp_optimized()

        for layer in measured_layers:
            if layer.type_ == MeasuredLayerType.NON_NORM_INPUT_LAYER:
                self._logger.warning(f"Input layer {layer.name} doesn't have normalization. "
                                     f"Was the data normalized manually?")
            elif layer.type_ == MeasuredLayerType.NORM_LAYER:
                hn_layer = self._nn_model.get_layer_by_name(layer.name)
                if hn_layer.op == LayerType.normalization:
                    std = 1 / params[layer.name]['kernel'][0, 0, :, 0]
                    mean = -params[layer.name]['bias'] * std
                else:
                    # Fetch values from model script command
                    fused_names = hn_layer._get_fused_model_script_layer_names()
                    norm_layers = set(fused_names) & set(norm_layers_from_alls)
                    if len(norm_layers) != 1:
                        self._logger.info("Normalization layer has been fused. and cannot be identified and verified.")
                        continue
                    args = norm_layers_from_alls[norm_layers.pop()]
                    mean = np.array(args[0], dtype=float)
                    std = np.array(args[1], dtype=float)

                bad_std = np.all(std < 1)
                bad_mean = np.all(np.logical_and(mean > 0, mean < 1))
                new_std = std * 255 if bad_std else std
                new_mean = mean * 255 if bad_mean else mean

                if bad_std or bad_mean:
                    self._logger.warning("Mean or std values were less than 1. Expected values in range of [0, 255]")
                    new_mean = map(lambda x: f"{x:.3f}", new_mean)
                    new_std = map(lambda x: f"{x:.3f}", new_std)
                    self._logger.info(f"{layer.name} = normalization([{', '.join(new_mean)}], [{', '.join(new_std)}])")

        return measured_layers

    def check_is_input_data_normalized(self, measured_layers: List[MeasuredLayer]):
        """
        Advanced check:
        - If the model doesn't have normalization layer, check if the input is normalized.
        - If the model has a normalization layer, check if the layer output is normalized.
        """
        if self._dataset is None:
            self._logger.warning(f"Normalization layer has not been validated, {msg.SKIP_NO_DATASET}")
            return

        ch_mean_by_layer, ch_std_by_layer = self._get_mean_and_std_per_sample(measured_layers)
        # TODO: We could create a different behavior if the data is normalized around [0, 1] and [-1, 1]
        for layer in measured_layers:
            self._logger.debug(f"{layer}, mean: {ch_mean_by_layer[layer.name]}, std: {ch_std_by_layer[layer.name]}")
            if layer.type_ in {MeasuredLayerType.NORM_LAYER, MeasuredLayerType.NON_NORM_INPUT_LAYER}:
                mean_th_low, mean_th_high = self.NORMALIZED_MEAN_TH_FLEX
                mean_over_th = np.any(ch_mean_by_layer[layer.name] > mean_th_high)
                mean_below_th = np.any(ch_mean_by_layer[layer.name] < mean_th_low)

                std_th_low, std_th_high = self.NORMALIZED_STD_TH_FLEX
                std_over_th = np.any(ch_std_by_layer[layer.name] > std_th_high)
                std_below_th = np.any(ch_std_by_layer[layer.name] < std_th_low)

                has_issue = mean_over_th or std_over_th or mean_below_th or std_below_th

                if has_issue:
                    self._logger.error(
                        f"Unexpected data distribution at {layer.name}. (expected normalized data) "
                        f"mean: {ch_mean_by_layer[layer.name]}, std: {ch_std_by_layer[layer.name]}")
                if (mean_over_th or std_over_th) and not (mean_below_th or std_below_th):
                    self._logger.error("Mean or std are above threshold. Data might not be noramlized")
                elif mean_below_th or std_below_th:
                    self._logger.error("Mean or std are below threshold. Data might be normalized twice")
                elif has_issue:
                    self._logger.error("Unexected normalized distribution. Data might not be normalized correctly")
            else:
                mean_th_low, mean_th_high = self.NORMALIZED_MEAN_TH_STRICT
                mean_values = ch_mean_by_layer[layer.name]
                mean_in_range = np.all((mean_values <= mean_th_high) & (mean_values >= mean_th_low))

                std_th_low, std_th_high = self.NORMALIZED_STD_TH_STRICT
                std_values = ch_std_by_layer[layer.name]
                std_in_range = np.all((std_values <= std_th_high) & (std_values >= std_th_low))

                if mean_in_range and std_in_range:
                    self._logger.error(
                        f"The input data of {layer.name} appears to be normalized, is it normalized twice? "
                        f"mean: {ch_mean_by_layer[layer.name]}, std: {ch_std_by_layer[layer.name]}")

    def _get_layers_to_sample(self, norm_layers_from_alls) -> List[MeasuredLayer]:
        """
        Find which layers should be sampled (mean and std)
        If an input has a norm layer as a decendent - pick the norm layer, otherwise pick the input layer.
        """
        norm_layers = list(filter(lambda x: x.op == LayerType.normalization,
                                  self._nn_model.stable_toposort()))

        fused_norm_layers = [layer for layer in self._nn_model.stable_toposort()
                             if set(layer._get_fused_model_script_layer_names()) & set(norm_layers_from_alls)]
        norm_layers.extend(fused_norm_layers)
        input_layers = self._nn_model.get_input_layers()
        measured_layers = list()
        for input_l in input_layers:
            has_norm = False
            for norm_l in norm_layers:
                if nx.has_path(self._nn_model, input_l, norm_l):
                    has_norm = True
            if not has_norm:
                measured_layers.append(MeasuredLayer(input_l.name, MeasuredLayerType.NON_NORM_INPUT_LAYER))
            else:
                measured_layers.append(MeasuredLayer(input_l.name, MeasuredLayerType.NORM_INPUT_LAYER))
        for layer in norm_layers:
            measured_layers.append(MeasuredLayer(layer.name, MeasuredLayerType.NORM_LAYER))
        return measured_layers

    def _get_mean_and_std_per_sample(self, measured_layers):
        measured_layers_names = [layer.name for layer in measured_layers]
        with self._runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            model = self._runner.get_keras_model(ctx)._model
            model.compile(save_interlayer=measured_layers_names)
            axes = np.arange(3)  # Assumes 4 dimensions with batch, channels is last

            ch_mean_by_layer = {layer: [] for layer in measured_layers_names}
            ch_std_by_layer = {layer: [] for layer in measured_layers_names}

            @tf.function
            def infer_mean_std(data):
                model(data)
                result = {}
                for layer, tensor in model.interlayer_tensors.items():
                    ch_mean_sample = tf.reduce_mean(tensor, axis=axes)
                    ch_std_sample = tf.math.reduce_std(tensor, axis=axes)
                    result[layer] = (ch_mean_sample, ch_std_sample)
                return result

            for data_sample, _ in self._dataset.batch(1):
                result = infer_mean_std(data_sample)
                for layer, (mean, std) in result.items():
                    ch_mean_by_layer[layer].append(mean.numpy())
                    ch_std_by_layer[layer].append(std.numpy())

        ch_mean = {}
        ch_std = {}
        for layer in measured_layers_names:
            ch_mean[layer] = np.mean(ch_mean_by_layer[layer], axis=0)
            ch_std[layer] = np.sqrt(np.mean(np.power(ch_std_by_layer[layer], 2), axis=0))

        return ch_mean, ch_std
