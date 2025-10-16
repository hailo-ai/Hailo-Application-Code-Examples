from dataclasses import dataclass
from typing import List

import numpy as np

from inspectors.base_inspector import BaseInspector, InspectorPriority
from hailo_model_optimization.acceleras.utils.acceleras_definitions import PrecisionMode
from hailo_sdk_common.hailo_nn.hn_definitions import LayerType


@dataclass
class PrecisionLayerInfo:
    layer_name: str
    sqnr: float
    prec_mode: PrecisionMode
    
    def __str__(self) -> str:
        return f"layer_name={self.layer_name}, sqnr={float(self.sqnr):0.3f}, precision_mode={self.prec_mode.value}"


class HighPrecisionInspector(BaseInspector):
    # TODO: add measure SNR function in case data is missing?
    # TODO: if advanced LAT was applied, use the output SNR for indication
    PRIORITY = InspectorPriority.MEDIUM

    SQNR_THRESHOLD = 10

    def _run(self):
        layers_prec_info = self._get_layers_precision_info()
        self.list_low_snr(layers_prec_info)

    def _get_layers_precision_info(self):
        nn_model = self._runner.get_hn_model()
        params_statistics = self._runner.get_params_statistics()
        layer_info_list = []
        input_layers = nn_model.get_input_layers()
        output_layers = nn_model.get_output_layers(False)
        missing_simple_lat = []
        missing_advanced_lat = []
        for layer in nn_model.stable_toposort():
            if self._skip_snr(layer, input_layers, output_layers):
                continue
            partial_sqnr = params_statistics[layer.name].get('layer_noise_analysis/partial_snr')
            sqnr = params_statistics[layer.name].get('layer_noise_analysis/snr')
            if sqnr is None:
                if partial_sqnr is not None:
                    sqnr = partial_sqnr
                    missing_advanced_lat.append(layer.name)
                else:
                    missing_simple_lat.append(layer.name)
                    continue
            layer_info = PrecisionLayerInfo(
                layer_name=layer.name,
                sqnr=sqnr,
                prec_mode=layer.precision_config.precision_mode
            )
            self._logger.debug(layer_info)
            layer_info_list.append(layer_info)
        if missing_advanced_lat:
            self._logger.warning("Missing data from layer_noise_analysis. Data from simple analysis was used.")
        if missing_simple_lat or missing_advanced_lat:
            self._logger.warning("Please run hailo analyze to fully utilize this inspector:\n"
                                 "hailo analyze-noise <model_har> --analyze-mode advanced --data-path <dataset>")

        return layer_info_list

    def _skip_snr(self, layer, input_layers, output_layers):
        """
        Skip input and output layers.
        Input will always be relatively high and it might affect the percentiles.
        Output SNR will exist also in the "real output"
        skip NMS and argmax because SNR is bad metric for them
        """
        is_io_layer = (layer in input_layers or layer in output_layers)
        is_bad_metric = layer.op in {LayerType.nms, LayerType.argmax}
        return is_io_layer or is_bad_metric

    def list_low_snr(self, layers_prec_info: List[PrecisionLayerInfo]):
        # TODO: add heurstic that includes centrality / flow / residual block(?)
        if len(layers_prec_info) == 0:
            return
        sqnr = [info.sqnr for info in layers_prec_info]
        q25 = np.percentile(np.concatenate(sqnr), 25)
        q75 = np.percentile(np.concatenate(sqnr), 75)
        iqr = q75 - q25
        low_rel_th = (q25 - 1.5 * iqr)
        relative_ourliers = [info for info in layers_prec_info if np.any(info.sqnr <= low_rel_th)]
        absolute_outliers = [info for info in layers_prec_info if np.any(info.sqnr <= self.SQNR_THRESHOLD)]

        if len(relative_ourliers) > 0:
            self._logger.warning("The following layers are relative outliers")
            for linfo in relative_ourliers:
                prec_mode = linfo.prec_mode.value
                self._logger.warning(f"\t{linfo.layer_name}, SQNR: {float(linfo.sqnr):.03f}, Precision Mode: {prec_mode}")
        if len(absolute_outliers) > 0:
            self._logger.warning(f"The following layers have SQNR lower than threshold {self.SQNR_THRESHOLD}")
            for linfo in absolute_outliers:
                prec_mode = linfo.prec_mode.value
                self._logger.warning(f"\t{linfo.layer_name}, SQNR: {float(linfo.sqnr):.03f}, Precision Mode: {prec_mode}")
        if len(relative_ourliers) > 0 or len(absolute_outliers) > 0:
            self._logger.warning("If you encounter accuracy issues, consider increasing the preicsion mode of "
                                 "some of these layer. Bottleneck usually have larger effect on accuracy.")
