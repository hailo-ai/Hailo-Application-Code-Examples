#!/usr/bin/env python3

import argparse
import time
import numpy as np
from PIL import Image
import os
from zenlog import log
import psutil

from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, InputVStreams,
                            OutputVStreamParams, OutputVStreams, HailoSchedulingAlgorithm, FormatType)

parser = argparse.ArgumentParser(description='Running a Hailo + ONNXRUntime inference')
parser.add_argument('hef', help="HEF file path")
parser.add_argument('--input-images', help="Images path to perform inference on. Could be either a single image or a folder containing the images. In case the input path is not defined, the input will be a 300 randomly generated tensors.")
parser.add_argument('--output-images-path', help="Inferenced output images folder path. If no input images were defined this will have no effect.")
parser.add_argument('--use-multi-process', action='store_true', help="Use the Multi-Process service of HailoRT along with the Model Scheduler.")
args = parser.parse_args()

# ---------------- Post-processing functions ----------------- #

def post_processing(inference_output, i):
    print(f'Image {i} was processed!')

# ------------------------------------------------------------ #

# ---------------- Inferences threads functions -------------- #

def send(configured_network, images_list, num_images):
    vstreams_params = InputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    print('Performing inference on input images...\n')
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in vstreams}
        for i in range(num_images):
            for vstream, _ in vstream_to_buffer.items():
                data = np.expand_dims(images_list[i], axis=0).astype(np.float32)
                vstream.send(data)

                
def recv(configured_network, num_images):
    vstreams_params = OutputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for i in range(num_images):
            data = []
            for vstream in vstreams:
                data.append(vstream.recv())
            post_processing(data, i)

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
                print('HailoRT Schduler service is enabled!')
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print('HailoRT Schduler service is disabled. Enabling service...')
    os.system('sudo systemctl daemon-reload & sudo systemctl enable --now hailort.service')
    

def create_vdevice_params():
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    if args.use_multi_process:
        params.group_id = "SHARED"
    return params

# ----------------------------------------------------------- #

# ---------------- Start of the example --------------------- #

check_if_service_enabled('hailort_service')

hef = HEF(args.hef)
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


num_images = len(images)

params = create_vdevice_params()
hef = HEF(args.hef)

with VDevice(params) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        
        [log.info('Input  layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_input_vstream_infos()]
        [log.info('Output layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_output_vstream_infos()]
        
        # Note: If you need to normalize the image, choose and change the set_resized_input function to right values
        if images_path:
            resized_images = [set_resized_input(lambda size: image.resize(size, Image.LANCZOS), width=width, height=height) for image in images]
        else:
            resized_images = images
        
        send_process = Process(target=send, args=(network_group, resized_images, num_images))
        recv_process = Process(target=recv, args=(network_group, num_images))
        start_time = time.time()
        recv_process.start()
        send_process.start()
        
        recv_process.join()
        send_process.join()

        end_time = time.time()
print('Inference was successful!\n')
log.info('-------------------------------------')
log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
log.info(' Average FPS:     {:.3f}'.format(num_images/(end_time - start_time)))
log.info('-------------------------------------')
