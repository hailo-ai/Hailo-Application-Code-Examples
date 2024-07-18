import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import inspectors.messages as msg 
from inspectors.base_inspector import BaseInspector, InspectorPriority


class ImageInspector(BaseInspector):
    PRIORITY = InspectorPriority.LOW

    def _run(self):
        ds = self._dataset.map(lambda x, y: x)
        ds = self._ds_dtype_correction(ds)
        ds_inv = self._ds_inverse(ds)
        self._preview_images(ds, 'standard.jpg')
        self._preview_images(ds_inv, 'inverse.jpg')
        if not self._interactive:
            self._logger.info("The images are assumed to be in RGB / BGR format")
        self._logger.warning("Please check `standard.jpg` and `inverse.jpg`, "
                            "if the images in `inverse.jpg` look more natural - "
                            "the calibration data might be stored as BGR.")
    
    def should_skip(self) -> str:
        if self._dataset is None:
            return msg.SKIP_NO_DATASET
        if self._dataset.element_spec[0].shape[-1] != 3:
            return msg.SKIP_BAD_DIMS_3
        sample = next(iter(self._dataset))[0].numpy()
        if np.any(sample < 0):
            return msg.SKIP_NEG_VALUES
        is_rgb = self.yes_no_prompt("Are images in RGB / BGR format")
        if not is_rgb:
            return msg.SKIP_NON_RGB
        return ""

    def _preview_images(self, dataset, filename, rows=4, cols=4):
        fig, axarr = plt.subplots(rows, cols)
        dataset = dataset.take(rows * cols)

        for ax, im in zip(axarr.ravel(), dataset):
            ax.imshow(im, cmap='gray', aspect='auto')
        for ax in axarr.ravel():
            ax.set_axis_off()
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)

    def _ds_dtype_correction(self, dataset):
        sample = next(iter(dataset))
        if np.any(sample.numpy().max() > 1):
            dataset = dataset.map(lambda x: tf.cast(x, tf.int32))
        return dataset

    def _ds_inverse(self, dataset):
        return dataset.map(lambda x: x[..., ::-1])
