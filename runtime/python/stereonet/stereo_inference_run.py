#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from multiprocessing import Process
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, ConfigureParams,
 InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
from zenlog import log
import time
from PIL import Image
import os
import argparse


##############################################################################
# This example is a StereoNet Hailo inference example.   
##############################################################################


parser = argparse.ArgumentParser(description='Running a Hailo inference for multi-inputs models')
parser.add_argument('hef', help="HEF file path.")
parser.add_argument('--right', help="The right side image.")
parser.add_argument('--left', help="The right side image")
parser.add_argument('--output-path', help="Inferenced output images folder path. If no path defined it the output would be save to current directory.")
args = parser.parse_args()

# ---------------- Post-processing functions ----------------- #

def post_processing(i, logits):
    im = Image.fromarray(np.squeeze(logits).astype(np.uint8), mode='L')
    output_path = args.output_path if args.output_path else '.'
    if not output_path[-1] == '/':
        output_path = output_path + '/'
    im.save(f'{output_path}output_image{i}.jpg')

# ------------------------------------------------------------ #

# ---------------- Inferences threads functions -------------- #

def send(configured_network, preprocessed_images, num_images):
    vstreams_params = InputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    print('Performing inference on input images...\n')
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        for i in range(num_images):
            images_dict = preprocessed_images[i]
            for j, vstream in enumerate(vstreams):
                data = np.expand_dims(images_dict[vstream.name], axis=0)
                vstream.send(data)

                
def recv(configured_network, num_images):
    vstreams_params = OutputVStreamParams.make_from_network_group(configured_network, quantized=False, format_type=FormatType.FLOAT32)
    configured_network.wait_for_activation(100)
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for i in range(num_images):
            for vstream in vstreams:
                data = vstream.recv()
                post_processing(i, data)

# ---------------- Pre-process functions ------------------ #

def pad_and_crop_tensor(image, target_height=368, target_width=1232):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    # Calculate the amount of padding required
    pad_height = tf.maximum(target_height - height, 0)
    pad_width = tf.maximum(target_width - width, 0)

    # Pad the tensor symmetrically on all sides
    paddings = tf.constant([[0, 0], [0, 0], [0, 0]], dtype=tf.int32)
    paddings = tf.tensor_scatter_nd_update(paddings, [[0, 1]], [pad_height])
    paddings = tf.tensor_scatter_nd_update(paddings, [[1, 1]], [pad_width])

    padded_image = tf.pad(image, paddings)

    # Crop or pad the tensor to the target shape
    cropped_image = tf.image.crop_to_bounding_box(padded_image, 0, 0, target_height, target_width)

    return cropped_image

def preprocess_stereonet(images, input_shape):
    image_l = images['image_l']
    image_r = images['image_r']
    crop_h, crop_w, _ = input_shape
    image_l = pad_and_crop_tensor(image_l, crop_h, crop_w)
    image_l = tf.ensure_shape(image_l, [crop_h, crop_w, 3])
    image_r = pad_and_crop_tensor(image_r, crop_h, crop_w)
    image_r = tf.ensure_shape(image_r, [crop_h, crop_w, 3])
    image = {'stereonet/input_layer1': image_l, 'stereonet/input_layer2': image_r}
    return image

def create_images_list(images_path, images):
    if (images_path.endswith('.jpg') or images_path.endswith('.png') or images_path.endswith('.bmp') or images_path.endswith('.jpeg')):
        images.append(Image.open(images_path))
    # if running inference on an images directory:
    if (os.path.isdir(images_path)):
        for img in os.listdir(images_path):
            if (img.endswith(".jpg") or img.endswith(".png") or img.endswith('.bmp') or img.endswith('.jpeg')):
                images.append(Image.open(os.path.join(images_path, img)))


def load_input_images(right_images, left_images, images):
    right = []
    left = []
    create_images_list(right_images, right)
    create_images_list(left_images, left)
    
    if len(right) != len(left):
        raise ValueError(f'Number of images from left and right must be the same. There are {len(right)} and {len(left)} left images.')

    for right_img, left_img in zip(right, left):
        images.append({'image_r' : np.array(right_img).astype(np.float32), 'image_l' : np.array(left_img).astype(np.float32)})

# ------------------------------------------------------------ #

# ---------------- Start of the example --------------------- #

if not args.right or not args.left:
    raise ValueError('Plese supply right and left image paths or images folders paths.')

hef = HEF(args.hef)

input_vstream_infos = hef.get_input_vstream_infos()
input_shape = input_vstream_infos[0].shape

images = []
preprocessed_images = []

load_input_images(args.right, args.left, images)

num_images = len(images)

for img in images:
    image = preprocess_stereonet(img, input_shape)
    preprocessed_images.append(image)

devices = Device.scan()

with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        [log.info('Input  layer: {:20.20} {}'.format(li.name, li.shape)) for li in input_vstream_infos]
        [log.info('Output layer: {:20.20} {}'.format(li.name, li.shape)) for li in hef.get_output_vstream_infos()]
        
        send_process = Process(target=send, args=(network_group, preprocessed_images, num_images))
        recv_process = Process(target=recv, args=(network_group, num_images))
        start_time = time.time()
        recv_process.start()
        send_process.start()
        with network_group.activate(network_group_params):
            recv_process.join()
            send_process.join()

        end_time = time.time()

print('Inference was successful!\n')
log.info('-------------------------------------')
log.info(' Infer Time:      {:.3f} sec'.format(end_time - start_time))
log.info(' Average FPS:     {:.3f}'.format(num_images/(end_time - start_time)))
log.info('-------------------------------------')



