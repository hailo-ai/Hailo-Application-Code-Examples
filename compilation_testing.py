from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode
import matplotlib.pyplot as plt
from PIL import Image

from hailo_sdk_client import ClientRunner, NNFramework

# naming the model
model_name = "orientedObjectDetection"
# path to the model
onnx_path = "objectDetection_onnx/vision.onnx"
# path for the parsed model
hailo_model_har_name = f"hailo_files/{model_name}_hailo_model.har"
# path to optimized model
quantized_model_har_path = f"hailo_files/{model_name}_quantized_model.har"

# create first runner
runner = ClientRunner(hw_arch="hailo8")
# translate onnx model
hn, npz = runner.translate_onnx_model(
    onnx_path,
    model_name,
    start_node_names=["Conv_1"],
    end_node_names=["heatmap", "offset", "boxparams"],
    net_input_shapes={"Conv_1": [1, 3, 720, 960]},
)
# save translated model
runner.save_har(hailo_model_har_name)

# initialise new runner
# I do not know if this is necessary
runner = ClientRunner(hw_arch="hailo8", har_path=hailo_model_har_name)

# rescale the image to 6/8 of its original size and normalize the input values to be in [-.5, .5]
def preproc(image, scale=6 / 8):
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        resized_image = tf.compat.v1.image.resize_bilinear(
            tf.expand_dims(image, 0), [int(h * scale), int(w * scale)]
        )
        squeezed_image = tf.squeeze(resized_image)

    out_image = squeezed_image.numpy() / 255 - 0.5
    return out_image


images_path = Path("data")
images_list = list(images_path.glob("*.png"))

calib_dataset = np.zeros((len(images_list), 720, 960, 3), dtype=np.float32)
for idx, image_path in enumerate(sorted(images_list)):
    image = np.array(Image.open(image_path))
    image = np.repeat(image[..., None], 3, 2)
    preprocessed_image = preproc(image)
    calib_dataset[idx, :, :, :] = preprocessed_image

plt.imshow(image, interpolation="nearest")
plt.title("Original image")
plt.savefig("original.png")
plt.close()
plt.imshow(calib_dataset[idx] + 0.5, interpolation="nearest")
plt.title("Preprocessed image")
plt.savefig("prepocessed.png")

alls_lines = ["model_optimization_config(calibration, batch_size=1)\n"]
# Save the commands in an .alls file, this is the Model Script
with open("hailo_files/script.alls", "w") as file:
    file.writelines(alls_lines)

# Load the model script to ClientRunner so it will be considered on optimization
runner.load_model_script("hailo_files/script.alls")

# For a single input layer, could use the shorter version - just pass the dataset to the function
# runner.optimize(calib_dataset)
# For multiple input nodes, the calibration dataset could also be a dictionary with the format:
# {input_layer_name_1_from_hn: layer_1_calib_dataset, input_layer_name_2_from_hn: layer_2_calibdataset}
hn_layers = runner.get_hn_dict()["layers"]
input_layers = [
    layer for layer in hn_layers if hn_layers[layer]["type"] == "input_layer"
]

calib_dataset_dict = {input_layers[0]: calib_dataset}
runner.optimize(calib_dataset_dict)
runner.save_har(quantized_model_har_path)

# do compilation
runner = ClientRunner(hw_arch="hailo8", har_path=quantized_model_har_path)
hef = runner.get_hw_representation()
file_name = model_name + ".hef"
with open("hailo_files/" + file_name, "wb") as file:
    file.write(hef)
