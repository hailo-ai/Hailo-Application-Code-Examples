#!/usr/bin/env python3

import numpy as np
from hailo_platform import __version__
from multiprocessing import Process
from hailo_platform import (HEF, PcieDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
from zenlog import log
import time
from PIL import Image
import os
import argparse
import cv2


parser = argparse.ArgumentParser(description='Running a Hailo inference with OpenCV')
parser.add_argument('--hef', help="HEF file path")
parser.add_argument('--input-images', help="Images path to perform inference on. Could be either a single image or a folder containing the images")
args = parser.parse_args()


# ---------------- Post-processing functions ----------------- #

def post_processing(inference_output):
   print('Here create your relevant post-processing functions. In the case, it is a classification post-processing:')
   labels = []
   with open('imagenet1000_clsidx_to_labels.txt','r') as f:
    labels = eval(f.read()) 
    print(labels[np.argmax(inference_output)])

# ------------------------------------------------------------ #

# ---------------- Inferences threads functions -------------- #

def send(configured_network, images_list, num_images):
    vstreams_params = InputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    log.info('\nPerforming inference on input images...\n')
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in vstreams}
        for i in range(num_images):
            for vstream, _ in vstream_to_buffer.items():
                data = np.expand_dims(images_list[i], axis=0).astype(np.float32)
                vstream.send(data)

                
def recv(configured_network, num_images):
    vstreams_params = OutputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for _ in range(num_images):
            values = []
            for vstream in vstreams:
                data = vstream.recv()
                values.append(data)
            post_processing(values)

# ------------------------------------------------------------ #

# ----------------Pre-processing functions ------------------- #

def prepare_image(image_path, images):
    raw_image = cv2.imread(image_path)
    image_RGB = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    cv2.normalize(image_RGB, image_RGB, 0, 255, norm_type=cv2.NORM_MINMAX)
    image = np.array(image_RGB, np.float32)
    images.append(image)


# ------------------------------------------------------------ #


# ---------------- Start of the example --------------------- #

if (not args.hef or not args.input_images):
    raise ValueError('You must define hef path and input images path in the command line. Run with -h for additional info')

images_path = args.input_images

images = []
if (images_path.endswith('.jpg') or images_path.endswith('.png')):
    prepare_image(images_path, images)
if (os.path.isdir(images_path)):
    for img in os.listdir(images_path):
        if (img.endswith(".jpg") or img.endswith(".png")):
            prepare_image(images_path + img, images)

num_images = len(images)

hef = HEF(args.hef)

with PcieDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        [log.info('Input  layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_input_vstream_infos()]
        [log.info('Output layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_output_vstream_infos()]

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
# NOTICE: The avrage FPS can onlt be achieved by a large enough number of frames. The FPS that will be recieved from
# one image does not reflect the average FPS of the model
 
log.info('-------------------------------------')
log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
log.info(' Average FPS:     {:.3f}'.format(num_images/(end_time - start_time)))
log.info('-------------------------------------')
