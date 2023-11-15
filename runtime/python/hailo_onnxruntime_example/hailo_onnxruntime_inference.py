#!/usr/bin/env python3

from locale import normalize
import numpy as np
import onnxruntime
from hailo_platform import __version__
from multiprocessing import Process, Queue
from hailo_platform import (HEF, PcieDevice, HailoStreamInterface, ConfigureParams,
 InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
from zenlog import log
import time
from PIL import Image
import os
import argparse

##############################################################################
# This example is a generic Hailo + ONNXRuntime inference detection example.   
# In order for this example to work properly, please create the relevant pre & 
# post-processing functions. 
##############################################################################


parser = argparse.ArgumentParser(description='Running a Hailo + ONNXRUntime inference')
parser.add_argument('hef', help="HEF file path")
parser.add_argument('onnx', help="ONNX file of path, when the output of the HEF is the input of the ONNX")
parser.add_argument('--input-images', help="Images path to perform inference on. Could be either a single image or a folder containing the images. In case the input path is not defined, the input will be a 300 randomly generated tensors.")
parser.add_argument('--output-images-path', help="Inferenced output images folder path. If no input images were defined this will have no effect.")
args = parser.parse_args()


# ---------------- Post-processing functions ----------------- #

def post_processing(inference_output, index):
    # Create your relevant post-processing functions
    print(f'Recieved image number: {index}')

# ------------------------------------------------------------ #

# ---------------- Inferences threads functions -------------- #

def send(configured_network, images_list, num_images):
    vstreams_params = InputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    print('Performing inference on input images...\n')
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in vstreams}
        for i in range(num_images):
            for vstream, _ in vstream_to_buffer.items():
                data = np.expand_dims(images_list[i], axis=0).astype(np.float32)
                vstream.send(data)

                
def recv(configured_network, write_q, num_images):
    vstreams_params = OutputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for _ in range(num_images):
            curr_vstream_data_dict = {}
            values = []
            for vstream in vstreams:
                data = vstream.recv()
                switched_tensor = np.swapaxes(np.swapaxes(np.expand_dims(data, axis=0), 1, 3), 2, 3) # 1 <--> 2, then 2 <--> 3
                values.append(switched_tensor)
            values.sort(key=lambda x: x.shape)
            inp = nms.get_inputs()
            inp.sort(key=lambda x: x.shape)
            curr_vstream_data_dict = {i.name : v for i, v in zip(inp, values)}
            write_q.put(curr_vstream_data_dict)
            
def ort_inference(read_q, images, num_images):
    i = 0
    while (i < num_images):
        if(read_q.empty() == False):
            inference_dict = read_q.get(0)
            inference_output = nms.run(None, inference_dict)
            print(inference_output[0].shape)
            image = images[i]
            post_processing(inference_output, i)
            i = i + 1

# ------------------------------------------------------------ #

# ---------------- Pre-processing functions ------------------ #

def set_resized_input(resize, width=320, height=320, normalize=False):
    result = resize((width, height))
    if normalize:
        return (np.array(result, np.float32) - 127) / 128
    return np.array(result, np.float32)

# ----------------------------------------------------------- #


# ---------------- Start of the example --------------------- #

hef = HEF(args.hef)
height, width, channels = hef.get_input_vstream_infos()[0].shape

images_path = args.input_images
nms_onnx_path = args.onnx
nms = onnxruntime.InferenceSession(nms_onnx_path)

images = []
if not images_path:
    images = np.zeros((300, height, width, channels), dtype=np.float32)
else:
    if (images_path.endswith('.jpg') or images_path.endswith('.png')):
        images.append(Image.open(images_path))
    if (os.path.isdir(images_path)):
        for image in os.listdir(images_path):
            if (image.endswith(".jpg") or image.endswith(".png")):
                images.append(Image.open(os.path.join(images_path, image)))

num_images = len(images)

with PcieDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        queue = Queue()
        
        # Note: If you need to normalize the image, choose and change the set_resized_input function to right values
        if images_path:
            resized_images = [set_resized_input(lambda size: image.resize(size, Image.LANCZOS), width=width, height=height) for image in images]
        else:
            resized_images = images
        
        send_process = Process(target=send, args=(network_group, resized_images, num_images))
        recv_process = Process(target=recv, args=(network_group, queue, num_images))
        nms_process = Process(target=ort_inference, args=(queue, images, num_images)) 
        start_time = time.time()
        recv_process.start()
        send_process.start()
        nms_process.start()
        with network_group.activate(network_group_params):
            recv_process.join()
            send_process.join()
            nms_process.join()

        end_time = time.time()
print('Inference was successful!\n')
log.info('-------------------------------------')
log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
log.info(' Average FPS:     {:.3f}'.format(num_images/(end_time - start_time)))
log.info('-------------------------------------')
