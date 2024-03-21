import os
from hailo_sdk_client import ClientRunner

model_name = 'resnet_v1_18'

# Optional: generate HEF file for the reference model
quantized_model_har_path = f'../har/{model_name}_quantized.har'

if False: # (os.path.isfile(quantized_model_har_path)== True):
    print("\nCompile the reference model")
    runner = ClientRunner(har=quantized_model_har_path)
    # hw_arch='hailo15h'
    hef = runner.compile()

    file_name = f'../hef/{model_name}.hef'
    with open(file_name, 'wb') as f:
        f.write(hef)

    har_path = f'../har/{model_name}_compiled.har'
    runner.save_har(har_path)
    os.system(f"hailo profiler {har_path}") 


# Generate HEF file for the feature extractor model
quantized_model_har_path = f'../har/{model_name}_featext_quantized.har'

if (os.path.isfile(quantized_model_har_path)== True):
    print("\nCompile the feature extractor model")
    runner = ClientRunner(har=quantized_model_har_path, hw_arch='hailo15h')

    hef = runner.compile()

    file_name = f'../hef/{model_name}_featext.hef'
    with open(file_name, 'wb') as f:
        f.write(hef)

    har_path = f'../har/{model_name}_featext_compiled.har'
    runner.save_har(har_path)
    os.system(f"hailo profiler {har_path}") 
