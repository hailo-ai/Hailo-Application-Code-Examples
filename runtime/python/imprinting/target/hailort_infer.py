#!/usr/bin/env python3

import numpy as np
from feat_extractor_class import FeatureExtractor
from hailo_platform import __version__
import onnxruntime
from multiprocessing import Process, Queue
from hailo_platform import (HEF, PcieDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
from zenlog import log
import time
from PIL import Image
import os
import argparse
import cv2

# Example of usage: python3 ./infer.py --hef ../hef/resnet_v1_18_featext.hef --onnx ../onnx/resnet_v1_18_fc.onnx --input-images ../data/ > results.txt

parser = argparse.ArgumentParser(description='Running a Hailo inference with OpenCV')
parser.add_argument('--hef', help="HEF file path")
parser.add_argument('--onnx', help="Path of ONNX file implementing the post-processing (ususally fully connected layer), that takes the output of the HEF as input")
parser.add_argument('--npz', help="Path of numpy arrays with weights&biases of the fully connected layer, that takes the output of the HEF as input")
parser.add_argument('--input-images', help="Images path to perform inference on. Could be either a single image or a folder containing the images")
args = parser.parse_args()

# ----------------Pre-processing functions ------------------- #

def prepare_image(image_path, images):
    raw_image = cv2.imread(image_path)
    image_RGB = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    cv2.normalize(image_RGB, image_RGB, 0, 255, norm_type=cv2.NORM_MINMAX)
    image = np.array(image_RGB, np.uint8)
    images.append(image)

# ---------------- Start of the example --------------------- #

if (not args.hef or not args.input_images):
    raise ValueError('You must define hef path and input images path in the command line. Run with -h for additional info')

# Create an instance of FeatureExtractor class

feat_extractor = FeatureExtractor()

images_path = args.input_images

images = []
image_names= []
if (images_path.endswith('.jpg') or images_path.endswith('.png')):
    prepare_image(images_path, images)
if (os.path.isdir(images_path)):
    for img in os.listdir(images_path):
        if (img.endswith(".jpg") or img.endswith(".png")):
            image_names.append(img)
            prepare_image(images_path + img, images)

num_images = len(images)

labels = []
with open('imagenet1000_clsidx_to_labels.txt','r') as f:
    labels = eval(f.read()) 

hef = HEF(args.hef)

with PcieDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        queue = Queue()

        [log.info('Input  layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_input_vstream_infos()]
        [log.info('Output layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_output_vstream_infos()]

        height, width, channels = hef.get_input_vstream_infos()[0].shape

        postproc_onnx_path = args.onnx
        postproc_model = onnxruntime.InferenceSession(postproc_onnx_path) \
                            if postproc_onnx_path is not None else None
        postproc_npz_path = args.npz                            
        postproc_npdict = dict(np.load(postproc_npz_path)) if postproc_npz_path is not None else None  # npy: load(..).item()
        
        # Note: If you need to normalize the image, choose and change the set_resized_input function to right values
        resized_images = [cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA) for img in images]
        
        send_process = Process(target=feat_extractor.send, args=(network_group, resized_images, num_images))
        recv_process = Process(target=feat_extractor.recv, args=(network_group, num_images, queue, postproc_model))
        post_process = Process(target=feat_extractor.post_processing, args=(postproc_npdict, postproc_model, queue, image_names, num_images, labels)) 
        start_time = time.time()
        recv_process.start()
        send_process.start()
        post_process.start()
        with network_group.activate(network_group_params):
            recv_process.join()
            send_process.join()
            post_process.join()

        end_time = time.time()
print('Inference was successful!\n')
# NOTICE: The avrage FPS can onlt be achieved by a large enough number of frames. The FPS that will be recieved from
# one image does not reflect the average FPS of the model
 
log.info('-------------------------------------')
log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
log.info(' Average FPS:     {:.3f}'.format(num_images/(end_time - start_time)))
log.info('-------------------------------------')







