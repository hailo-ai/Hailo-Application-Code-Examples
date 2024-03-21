from hailo_sdk_client import ClientRunner, InferenceContext

import os
import numpy as np

model_name = 'resnet_v1_18'
calib_file = 'calib_set.npy'

# Optional: Optimize the reference model
hailo_model_har_name = f'../har/{model_name}_parsed.har'

if (os.path.isfile(hailo_model_har_name)== True):
    print("\nOptimize the reference model")
    runner = ClientRunner(har=hailo_model_har_name)
        
    # Batch size is 8 by default
    alls = 'normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n'

    # Load the model script to ClientRunner so it will be considered on optimization
    runner.load_model_script(alls)

    # Call Optimize to perform the optimization process
    runner.optimize(calib_file)

    # Save the result state to a Quantized HAR file
    quantized_model_har_path = f'../har/{model_name}_quantized.har'
    runner.save_har(quantized_model_har_path)

# Optimzie the feature extractor model
hailo_model_har_name = f'../har/{model_name}_featext_parsed.har'

if (os.path.isfile(hailo_model_har_name)== True):

    print("\nOptimize the feature extractor model")
    runner = ClientRunner(har=hailo_model_har_name)

    # Batch size is 8 by default
    alls = 'normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n'

    # Load the model script to ClientRunner so it will be considered on optimization
    runner.load_model_script(alls)

    # Call Optimize to perform the optimization process
    runner.optimize(calib_file)

    # Save the result state to a Quantized HAR file
    quantized_model_har_path = f'../har/{model_name}_featext_quantized.har'
    runner.save_har(quantized_model_har_path)