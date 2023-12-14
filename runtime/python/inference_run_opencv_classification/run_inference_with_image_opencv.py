#!/usr/bin/env python3

import numpy as np
from hailo_platform import __version__
from multiprocessing import Process
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
from zenlog import log
import time
import os
import argparse
import cv2


parser = argparse.ArgumentParser(description='Running a Hailo inference with OpenCV')
parser.add_argument('hef', help="HEF file path")
parser.add_argument('images', help="Images path to perform inference on. Could be either a single image or a folder containing the images")
parser.add_argument('--labels', default='imagenet1000_clsidx_to_labels.txt', help="Path to lables .txt file. Defualt to imagenet1000_clsidx_to_labels")
args = parser.parse_args()


# ---------------- Post-processing functions ----------------- #

def post_processing(image_name, inference_output, labels):
    # Create your relevant post-processing functions
    print(f"{image_name}: {labels[np.argmax(inference_output)]}")

# ------------------------------------------------------------ #

# ---------------- Inferences threads functions -------------- #

def send(configured_network, images_list, num_images):
    vstreams_params = InputVStreamParams.make_from_network_group(configured_network, quantized=True, format_type=FormatType.UINT8)
    configured_network.wait_for_activation(100)
    log.info('\nPerforming inference on input images...\n')
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in vstreams}
        for i in range(num_images):
            for vstream, _ in vstream_to_buffer.items():
                data = np.expand_dims(images_list[i], axis=0).astype(np.float32)
                vstream.send(data)

                
def recv(configured_network, image_names, num_images, labels):
    vstreams_params = OutputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for _ in range(num_images):
            values = []
            image_name= image_names.pop(0)
            for vstream in vstreams:
                data = vstream.recv()
                values.append(data)
            post_processing(image_name, values, labels)

# ------------------------------------------------------------ #

# ----------------Pre-processing functions ------------------- #

def prepare_image(images):
    for i, image in enumerate(images):
        cv2.normalize(image, image, 0, 255, norm_type=cv2.NORM_MINMAX)
        processed_image = np.array(image, np.float32)
        images[i] = processed_image


# ------------------------------------------------------------ #


# ---------------- Start of the example --------------------- #

hef = HEF(args.hef)
height, width, channels = hef.get_input_vstream_infos()[0].shape

images_path = args.images

images = []
# if running inference on a single image:
if images_path.endswith('.jpg') or images_path.endswith('.png') or images_path.endswith('.bmp') or images_path.endswith('.jpeg'):
    images.append(cv2.cvtColor(cv2.imread(images_path), cv2.COLOR_BGR2RGB))
# if running inference on an images directory:
elif os.path.isdir(images_path):
    for img in os.listdir(images_path):
        if (img.endswith(".jpg") or img.endswith(".png") or img.endswith('.bmp') or img.endswith('.jpeg')):
            images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
else:
    raise ValueError('You must define input images path to a specific image or to a folder containing images. Run with -h for additional info')


num_images = len(images)

labels = []
with open(args.labels, 'r') as f:
    labels = eval(f.read())

devices = Device.scan()

inputs = hef.get_input_vstream_infos()
outputs = hef.get_output_vstream_infos()

with VDevice(device_ids=devices) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    
    [log.info('Input  layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in inputs]
    [log.info('Output layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in outputs]

    height, width, channels = hef.get_input_vstream_infos()[0].shape
    
    # Note: If you need to normalize the image, choose and change the set_resized_input function to right values
    resized_images = [cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA) for img in images]
    
    send_process = Process(target=send, args=(network_group, resized_images, num_images))
    recv_process = Process(target=recv, args=(network_group, num_images))
    start_time = time.time()
    recv_process.start()
    send_process.start()
    with network_group.activate(network_group_params):
        recv_process.join()
        send_process.join()

    end_time = time.time()
    print('Inference was successful!\n')
    # NOTICE: The avrage FPS can only be achieved by a large enough number of frames. The FPS that will be recieved from
    # one image does not reflect the average FPS of the model
    
    log.info('-------------------------------------')
    log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
    log.info(' Average FPS:     {:.3f}'.format(num_images/(end_time - start_time)))
    log.info('-------------------------------------')







