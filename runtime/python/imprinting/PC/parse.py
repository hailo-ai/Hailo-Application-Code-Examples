# Description: This script parses an ONNX model and saves it as a HAR file.
#           It also parses only the feature extractor part of the model and saves it as a HAR file.
#           It also extracts the FC layer from the entire model and saves it as an ONNX sub-model.
#           This is useful for reference purpose.

from hailo_sdk_client import ClientRunner
import onnx, numpy as np

model_name = 'resnet_v1_18'
onnx_path = '../onnx/resnet_v1_18.onnx'

chosen_hw_arch = 'hailo8'

# Optional: Parse entire model for reference purpose and save it as a HAR file
runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(onnx_path, model_name)
model_har_name = f'../har/{model_name}_parsed.har'
runner.save_har(model_har_name)

# Parse only the feature extractor part of the model and save it as a HAR file
runner = ClientRunner(hw_arch=chosen_hw_arch)
end_node= 'Flatten_47' # This is the last node of the feature extractor part of the model
hn, npz = runner.translate_onnx_model(onnx_path, model_name, end_node_names= end_node)
model_har_name = f'../har/{model_name}_featext_parsed.har'
runner.save_har(model_har_name)

# Extract the FC layer from the entire model and save it as an ONNX sub-model
input_path = onnx_path
output_path = f'../onnx/{model_name}_fc.onnx'
input_names = ['190']
output_names = ['191']

# Also save the FC W&B as NPZ
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

onnx_model = onnx.load(output_path)
initializers=onnx_model.graph.initializer
onnx_weights = {}
for initializer in initializers:
    W = onnx.numpy_helper.to_array(initializer)
    onnx_weights[initializer.name] = W

np.savez(f'../onnx/{model_name}_fc', **onnx_weights)