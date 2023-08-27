#!/usr/bin/env python3

import argparse
import time
import numpy as np
from PIL import Image
import os
from zenlog import log
import psutil

from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, HailoSchedulingAlgorithm, FormatType)

parser = argparse.ArgumentParser(description='Running a Hailo + ONNXRUntime inference')
parser.add_argument('hefs', nargs='*', help="Path to the HEF files to be inferenced")
parser.add_argument('--input-images', help="Images path to perform inference on. Could be either a single image or a folder containing the images. In case the input path is not defined, the input will be a 300 randomly generated tensors.")
parser.add_argument('--output-images-path', help="Inferenced output images folder path. If no input images were defined this will have no effect.")
parser.add_argument('--use-multi-process', action='store_true', help="Use the Multi-Process service of HailoRT along with the Model Scheduler.")
args = parser.parse_args()

# ---------------- Post-processing functions ----------------- #

def post_processing(inference_output, i):
    print(f'Image {i} was processed!')

# ------------------------------------------------------------ #

# ---------------- Inferences threads functions -------------- #

def infer(network_group, input_vstreams_params, output_vstreams_params, images):
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        for idx, image in enumerate(images):
            infer_results = infer_pipeline.infer(np.expand_dims(image, axis=0).astype(np.float32))
            post_processing(infer_results, idx)

# ----------------------------------------------------------- #

# ---------------- Pre-processing functions ------------------ #

def set_resized_input(resize, width=640, height=640, do_normalization=False):
    result = resize((width, height))
    # Change normalization if needed
    if do_normalization:
        return (np.array(result, np.float32) - 127) / 128
    return np.array(result, np.float32)

# ----------------------------------------------------------- #

# --------------- Hailo Scheduler service functions ---------- #

def check_if_service_enabled(process_name):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            if process_name.lower() in proc.name().lower():
                print('HailoRT Scheduler service is enabled!')
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print('HailoRT Scheduler service is disabled. Enabling service...')
    os.system('sudo systemctl daemon-reload && sudo systemctl enable --now hailort.service')
    

def create_vdevice_params():
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    if args.use_multi_process:
        params.group_id = "SHARED"
    return params

# ----------------------------------------------------------- #

# ---------------- Start of the example --------------------- #

check_if_service_enabled('hailort_service')

hefs = []
all_images = []
for hef_path in args.hefs:
    hef_name = hef_path.split('/')[-1].split('.')[0]
    hef = HEF(hef_path)
    hefs.append(hef)
    
    log.info(f'HEF name: {hef_name}')
    [log.info('Input  layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_input_vstream_infos()]
    [log.info('Output layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_output_vstream_infos()]
    print()
    
    height, width, channels = hef.get_input_vstream_infos()[0].shape

    images_path = args.input_images

    images = []
    if not images_path:
        images = np.zeros((300, height, width, channels), dtype=np.float32)
    else:
        if (images_path.endswith('.jpg') or images_path.endswith('.png') or images_path.endswith('.bmp')):
            images.append(Image.open(images_path))
        if (os.path.isdir(images_path)):
            for image in os.listdir(images_path):
                if (image.endswith(".jpg") or image.endswith(".png") or images_path.endswith('.bmp')):
                    images.append(Image.open(os.path.join(images_path, image)))
    
    all_images.append(images)

params = create_vdevice_params()

with VDevice(params) as target:
    for i, hef in enumerate(hefs):
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        
        network_group = target.configure(hef, configure_params)[0]
        
        infer_processes = []
        
        input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        # Note: If you need to normalize the image, choose and change the set_resized_input function to right values
        if images_path:
            height, width, _ = hef.get_input_vstream_infos()[0].shape
            resized_images = [set_resized_input(lambda size: image.resize(size, Image.LANCZOS), width=width, height=height) for image in all_images[i]]
        else:
            resized_images = all_images[i]
        
        infer_processes.append(Process(target=infer, args=(network_group, input_vstreams_params, output_vstreams_params, resized_images)))
        
    start_time = time.time()
    
    for i in range(len(infer_processes)):
        infer_processes[i].start()
    for i in range(len(infer_processes)):    
        infer_processes[i].join()

        end_time = time.time()
print('Inference was successful!\n')

log.info('-------------------------------------')
log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
log.info(' Average FPS:     {:.3f}'.format(len(all_images[0])/(end_time - start_time)))
log.info('-------------------------------------')
