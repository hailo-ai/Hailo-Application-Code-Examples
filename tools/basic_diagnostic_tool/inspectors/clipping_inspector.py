import numpy as np
import tensorflow as tf

from inspectors.base_inspector import BaseInspector

from hailo_sdk_client.exposed_definitions import InferenceContext


class ClippingInspector(BaseInspector):
    def _run(self):
        hist_data, hist_ranges = self._collect_hist_per_layer()
        self.check_histograms(hist_data, hist_ranges)

    def _collect_hist_per_layer(self):
        qparams = self._runner.get_params_translated()
        with self._runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            model = self._runner.get_keras_model(ctx)._model
            hist_layers = [lname for lname, layer in model.layers.items() if layer.activation_atomic_op]
            model.compile(save_interlayer=hist_layers)

            # TODO: Find correct ranges or collect them based on the dataset...
            hist_ranges = {}
            for lname in hist_layers:
                l_min, l_max = None, None
                for out_index in range(model.layers[lname].num_outputs):
                    i_min, i_max = qparams[lname][f'stats/output_{out_index}/stats_limvals']
                    if l_min is None or i_min < l_min:
                        l_min = i_min
                    if l_max is None or i_max > l_max:
                        l_max = i_max
                hist_ranges[lname] = np.array([l_min, l_max])
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
        for layer, hist in hist_data.items():
            cum_sum = np.cumsum(hist)
            percentiles = cum_sum / cum_sum[-1] * 100
            max_bin_99_9 = np.max(np.where(percentiles < 99.9))
            min_bin_00_1 = np.min(np.where(percentiles > 0.1))
            bin_size = (hist_ranges[layer][1] - hist_ranges[layer][0]) / 100
            should = False
            if max_bin_99_9 < 95 and (hist_ranges[layer][0] + bin_size * (max_bin_99_9 + 1)) > 0:
                should = True
            if min_bin_00_1 > 5 and (hist_ranges[layer][0] + bin_size * (min_bin_00_1 + 1)) < 0:
                should = True
            if should:
                min_range = hist_ranges[layer][0] + bin_size * (min_bin_00_1 + 1)
                max_range = hist_ranges[layer][0] + bin_size * (max_bin_99_9 + 1)
                print(layer, [min_bin_00_1, max_bin_99_9], [min_range, max_range])

    # TODO: filter by snr?
    # TODO: collect histograms?
    # TODO: 