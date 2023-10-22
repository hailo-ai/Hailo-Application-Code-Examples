import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inspectors.base_inspector import BaseInspector, InspectorPriority


class ImageInspector(BaseInspector):
    PRIORITY = InspectorPriority.LOW

    def _run(self):
        if self._dataset is None:
            self._logger.warning("Skipping image inspector, no dataset was provided")
            return
        if self._dataset.element_spec[0].shape[-1] != 3:
            self._logger.warning("Skipping image inspector, last dim doesn't have 3 channels")
            return
        sample = next(iter(self._dataset))[0].numpy()
        if np.any(sample < 0):
            self._logger.warning("Skipping image inspector, data has negative values")
            return
        ds = self._dataset.map(lambda x, y: x)
        ds = self._ds_dtype_correction(ds)
        ds_inv = self._ds_inverse(ds)
        self._preview_images(ds, 'standard.jpg')
        self._preview_images(ds_inv, 'inverse.jpg')
        self._logger.warning("Please check `standard.jpg` and `inverse.jpg`, "
                             "if the images in `inverse.jpg` look more natural - "
                             "the calibration data might be stored as bgr.")

    def _preview_images(self, dataset, filename, rows=4, cols=4):
        fig, axarr = plt.subplots(rows, cols)
        dataset = dataset.take(rows * cols)

        for ax, im in zip(axarr.ravel(), dataset):
            ax.imshow(im, cmap='gray')
        fig.savefig(filename)

    def _ds_dtype_correction(self, dataset):
        sample = next(iter(dataset))
        if np.any(sample.numpy().max() > 1):
            dataset = dataset.map(lambda x: tf.cast(x, tf.int32))
        return dataset

    def _ds_inverse(self, dataset):
        return dataset.map(lambda x: x[..., ::-1])
