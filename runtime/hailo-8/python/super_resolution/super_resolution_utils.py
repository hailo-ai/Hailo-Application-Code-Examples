from PIL import Image
import numpy as np
from pathlib import Path
import cv2

RGB2YUV_mat = [
    [0.25678824, -0.14822353, 0.43921569],
    [0.50412941, -0.29099216, -0.36778824],
    [0.09790588, 0.43921569, -0.07142745],
]
YUV2RGB_mat = [
    [1.16438355, 1.16438355, 1.16438355],
    [0.0, -0.3917616, 2.01723105],
    [1.59602715, -0.81296805, 0.0],
]
RGB2YUV_offset = [16, 128, 128]

class SuperResolutionUtils:
    """
    Base class for super-resolution utility functions.

    Methods:
        pre_process(image: Image.Image, model_w: int, model_h: int) -> np.ndarray:
            Preprocesses an image for the super-resolution model.

        post_process(infer_result: np.ndarray, input_image: np.ndarray) -> Image.Image:
            Post-processes the model's output into a final image.
    """

    def pre_process(self, image: Image.Image, model_w: int, model_h: int) -> np.ndarray:
        pass

    def post_process(self, infer_result: np.ndarray, input_image: np.ndarray) -> Image.Image:
        pass

class SrganUtils(SuperResolutionUtils):
    """
    Utility class for SRGAN-specific preprocessing and postprocessing.

    Methods:
        pre_process(image: Image.Image, model_w: int, model_h: int) -> np.ndarray:
            Resizes the input image to the model dimensions.

        post_process(infer_result: np.ndarray, input_image: np.ndarray) -> Image.Image:
            Converts the inference result back into an image.
    """

    def pre_process(self, image: Image.Image, model_w: int, model_h: int) -> np.ndarray:
        image = image.resize((model_w, model_h), Image.BICUBIC)
        image = np.array(image)
        return image

    def post_process(self, infer_result: np.ndarray, input_image: np.ndarray) -> Image.Image:
        return Image.fromarray(infer_result)

class Espcnx4Utils(SuperResolutionUtils):
    """
    Utility class for ESPCNx4-specific preprocessing and postprocessing.

    Attributes:
        model_w (int): Model width for postprocessing.
        model_h (int): Model height for postprocessing.

    Methods:
        pre_process(image: Image.Image, model_w: int, model_h: int) -> np.ndarray:
            Converts an image to YUV format and extracts the Y channel.

        post_process(infer_result: np.ndarray, input_image: np.ndarray) -> Image.Image:
            Combines the SR Y-channel with resized UV channels and converts back to RGB.
    """

    def __init__(self):
        self.model_w = None
        self.model_h = None

    def pre_process(self, image: Image.Image, model_w: int, model_h: int) -> np.ndarray:
        # Save the model width and height for post-processing
        self.model_w = model_w
        self.model_h = model_h

        # RGB --> YUV
        image = np.array(image.resize((model_w, model_h), Image.BILINEAR)).astype(np.float32)
        image = (np.matmul(image, RGB2YUV_mat) + RGB2YUV_offset) / 255
        image = image.astype(np.float32)
        y_channel, _, _ = cv2.split(image)
        y_channel = np.expand_dims(y_channel, axis=-1)
        return y_channel

    def post_process(self, infer_result: np.ndarray, input_image: np.ndarray) -> Image.Image:
        input_image = np.array(input_image.resize((self.model_w, self.model_h), Image.BILINEAR)).astype(np.float32)
        infer_result = infer_result * 255
        infer_result = infer_result.astype(np.uint8)

        # Input RGB --> YUV
        img_yuv = np.matmul(input_image, RGB2YUV_mat) + RGB2YUV_offset

        # Resizing naively to get the resized U and V channels
        img_yuv_resized = np.clip(cv2.resize(
            img_yuv, (infer_result.shape[1], infer_result.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)

        # Concatenate the super-resolution Y channel with the resized U, V channels
        img_out = np.concatenate([infer_result, img_yuv_resized[..., 1:]], axis=2)

        # YUV --> RGB
        img_out_rgb = np.clip(np.matmul(img_out - RGB2YUV_offset, YUV2RGB_mat), 0, 255).astype(np.uint8)
        srgan_image = Image.fromarray(img_out_rgb.astype(np.uint8))

        return srgan_image