import numpy as np
import tensorflow as tf
import logging

import inspectors.messages as msg 
from inspectors.base_inspector import BaseInspector, InspectorPriority

from hailo_sdk_client.exposed_definitions import InferenceContext
from hailo_model_optimization.acceleras.hailo_layers.base_hailo_layer import BaseHailoLayer


class ClippingInspector(BaseInspector):
    PRIORITY = InspectorPriority.LOW
    ITEM_TH = 3

    def _run(self):
        # TODO: Is there an indication whether the range was matched?
        self._logger.info(f"Warning is printed if more than 20% of the range has only {self.ITEM_TH} items. "
                          f"Info is printed if more than 5% of the range has only {self.ITEM_TH} items. "
                          f"Consider analyzing the data in depth before applying clipping")
        self._logger.info("In some cases the range might not be fixable and affected by other factors.")
        self._logger.info("In some cases QFT might utilize the unused range and improve performance.")
        self._logger.info("In general, activation clipping suggestion if very sensitive to the calibration set. "
                          "Applying activation clipping in some cases might reduce accuracy.")

        hist_data, hist_ranges = self._collect_hist_per_layer()
        recommendations = self.check_histograms(hist_data, hist_ranges)
        self.add_to_model_script(recommendations)

    def should_skip(self) -> str:
        if self._dataset is None:
            return msg.SKIP_NO_DATASET
        return ""
    
    @staticmethod
    def has_activation(layer):
        return isinstance(layer, BaseHailoLayer) and layer.activation_atomic_op is not None

    def _collect_hist_per_layer(self):
        qparams = self._runner.get_params_translated()
        with self._runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            model = self._runner.get_keras_model(ctx)._model
            hist_layers = [lname for lname, layer in model.layers.items() if self.has_activation(layer)]
            model.compile(save_interlayer=hist_layers)

            # TODO: Find correct ranges or collect them based on the dataset...
            hist_ranges = {}
            for lname in hist_layers:
                l_min, l_max = 0, 0
                for i in range(model.layers[lname].num_outputs):
                    i_min, i_max = qparams[lname][f'stats/output_{i}/stats_limvals:0']
                    l_min, l_max = min(l_min, i_min), max(l_max, i_max)
                hist_ranges[lname] = np.array([l_min, l_max])
                self._logger.debug(f"{lname}: activation range is ({l_min}, {l_max})")
                if l_max - l_min > 40:
                    self._logger.warning(f"Activation range of layer {lname} is high ({l_min:.03f}, {l_max:.03f})")
            full_result = {lname: np.zeros(100, dtype=np.uint32) for lname in hist_layers}

            @tf.function
            def infer_hist(data):
                model(data)
                batch_result = {lname: np.zeros(100, dtype=np.uint32) for lname in hist_layers}
                for layer, tensor in model.interlayer_tensors.items():
                    if model.layers[layer].num_outputs == 1:
                        hist1 = tf.histogram_fixed_width(tensor, hist_ranges[layer])
                        batch_result[layer] += hist1
                    else:
                        for ten in tensor:
                            hist1 = tf.histogram_fixed_width(ten, hist_ranges[layer])
                            batch_result[layer] += hist1
                return batch_result

            for data_sample, _ in self._dataset.batch(1):
                batch_result = infer_hist(data_sample)
                for lname, lresult in batch_result.items():
                    full_result[lname] += lresult.numpy().astype('uint32')
        return full_result, hist_ranges

    def check_histograms(self, hist_data, hist_ranges):
        recommendations = {}
        for layer, hist in hist_data.items():
            bin_size = (hist_ranges[layer][1] - hist_ranges[layer][0]) / len(hist)
            right_msg = left_msg = ""
            min_bins = np.where(np.cumsum(hist) <= self.ITEM_TH)[0]
            bin1 = 0 if len(min_bins) == 0 else np.min(min_bins)
            bin2 = np.max(np.where(np.cumsum(hist[::-1])[::-1] > self.ITEM_TH)[0]) + 1
            count_left = np.sum(hist[:bin1])
            count_right = np.sum(hist[bin2:])
            log_level = 0
            if bin1 != 0 and (hist_ranges[layer][0] + bin_size * (bin1)) < 0:
                left_msg = f"{bin1}% of the range (of the low range) has {count_left} items"
                if bin1 <= 20:
                    new_log_level = logging.DEBUG
                else:
                    new_log_level = logging.WARNING
                log_level = max(log_level, new_log_level)
            if bin2 != len(hist) and (hist_ranges[layer][0] + bin_size * (bin2)) > 0:
                right_msg = f"{len(hist) - bin2}% of the range (of the high range) has {count_right} items"
                if bin2 >= 80:
                    new_log_level = logging.DEBUG
                else:
                    new_log_level = logging.WARNING
                log_level = max(log_level, new_log_level)
            should_right = len(right_msg) > 0
            should_left = len(left_msg) > 0
            if should_right or should_left:
                spacer = ', ' if should_left and should_right else ''
                max_range = hist_ranges[layer][0] + bin_size * (bin2) if should_right else hist_ranges[layer][1]
                min_range = hist_ranges[layer][0] + bin_size * (bin1) if should_left else hist_ranges[layer][0]
                message = f"Layer {layer}, {left_msg}{spacer}{right_msg}. Suggested manual range [{min_range:.03f}, {max_range:.03f}]"
                self._logger.log(log_level, message)
                recommendations[layer] = (log_level, (min_range, max_range))
        return recommendations

    def add_to_model_script(self, recommendations):
        if self._interactive:
            selection = input("Would you like to add recommendation to model script? (none/warning/all) ").lower()
            while selection not in ('none', 'warning', 'all'):
                selection = input("Invalid selection, choose one of: none/warning/all").lower()
        else:
            selection = 'warning'
        commands = []
        if selection == 'none':
            return commands
        cmd_tmp = "pre_quantization_optimization(activation_clipping, layers=[{layer}], mode=manual, clipping_values=[{lim_low:.03f},{lim_high:.03f}])"
        for layer, (log_level, limvals) in recommendations.items():
            if selection == 'all':
                commands.append(cmd_tmp.format(layer=layer, lim_low=limvals[0], lim_high=limvals[1]))
            elif selection == 'warning' and log_level == logging.WARNING:
                commands.append(cmd_tmp.format(layer=layer, lim_low=limvals[0], lim_high=limvals[1]))
        self._new_commands = commands
