#!/usr/bin/env python3

import numpy as np
from PIL import Image
from pathlib import Path
import os
from loguru import logger
import argparse
import time

from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)


def configure_and_get_network_group(hef, target):
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    return network_group


def create_input_output_vstream_params(network_group):
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=FormatType.UINT8)
    return input_vstreams_params, output_vstreams_params


def print_input_output_vstream_info(hef):
    input_vstream_info = hef.get_input_vstream_infos()
    output_vstream_info = hef.get_output_vstream_infos()

    for layer_info in input_vstream_info:
        logger.info('Input layer: {} {}'.format(layer_info.name, layer_info.shape))
    for layer_info in output_vstream_info:
        logger.info('Output layer: {} {}'.format(layer_info.name, layer_info.shape))
    
    return input_vstream_info, output_vstream_info


def run_inference(images, hef, output_path):
    """Run inference on hailo-8

    Args:
        images (_type_): images to run inference on
        hef (HEF): hef file
        output_path (Path): output path (folder) to save output images
    """
    # create folder for output images if doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # configuration
    devices = Device.scan()
    with VDevice(device_ids=devices) as target:
        network_group = configure_and_get_network_group(hef, target)
        network_group_params = network_group.create_params()
        input_vstreams_params, output_vstreams_params = create_input_output_vstream_params(network_group)
        
        # print info of input & output
        input_vstream_info, output_vstream_info = print_input_output_vstream_info(hef)
        
        output_images = list()
        total_inference_time = 0
        images_num = 0
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            for idx, image in enumerate(images):
                # assuming one input to the model              
                input_data = {input_vstream_info[0].name: np.expand_dims(image, axis=0)}
                
                with network_group.activate(network_group_params):
                    start_time = time.time()
                    raw_image = infer_pipeline.infer(input_data)
                    end_time = time.time()
                    total_inference_time += (end_time - start_time)
                    
                    raw_image = raw_image[next(iter(output_vstream_info)).name][0]
                    image_name = str(idx) + '_srgan.png'
                    output_image = Image.fromarray(raw_image)
                    output_images.append(output_image)
                    output_image.save(output_path/image_name, format='png')
                    images_num = images_num + 1
        logger.info("Total inference time: {} sec, {} images", total_inference_time, images_num)
        return output_images


def parser_init():
    """
    Function to declare an argument parser for all the possible command line arguments used in this script
    @return: instance of ArgumentParser
    """
    parser = argparse.ArgumentParser(description="SRgan inference")

    parser.add_argument(
        "hef",
        help="Path of srgan.hef",
        default="srgan.hef"
    )

    parser.add_argument(
        "-i",
        "--images",
        default="test_images",
        help="Path of images to perform inference on. \
            Could be either a single image or a folder containing the images",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        default="output_images",
        help="Path of folder for output images",
    )
    
    parser.add_argument(
        "-s",
        "--show",
        action='store_true',
        default=False,
        help="Show example output",
    )

    return parser


def is_image(path):
    """return True if path is a image of type: bmp, png, jpg, jpeg."""
    return (path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png') or path.endswith('.bmp'))


def load_input_images(images_path, height=None, width=None):
    """load input images and resize to (height, width)

    Args:
        images_path (str): Path of images to perform inference on. \
            Could be either a single image or a folder containing the images
        height (int): height to resize. default no resize
        width (int): width to resize. default no resize
    Returns:
        list: list of PIL images
    """
    input_images = list()
    dir_path = Path()
    if (os.path.isdir(images_path)):
        # running inference on a single image
        dir_path = Path(images_path)
        input_images = [img for img in os.listdir(images_path) if is_image(img)]
    else: 
        # running inference on a single image
        if is_image(images_path):
          input_images = [str(images_path)]
    
    images_for_infer = list()
    for image in input_images:
        image = Image.open(dir_path/image)
        if height & width:
            image = np.array(image.resize((width,height), Image.BILINEAR)).astype(np.float32)
        images_for_infer.append(image)
    
    # logger.info("running inference on images: {}", input_images)
    return images_for_infer


def create_and_save_enlarged_images(images, height, width, output_path):
    enlarged_images = []
    # create folder for output images if doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, image in enumerate(images):
        enlarged_image = Image.fromarray((image).astype(np.uint8)).resize((width,height), Image.BILINEAR)
        image_name = str(idx) + '_enlarged.png'
        enlarged_image.save(output_path/image_name, format='png')
        enlarged_images.append(enlarged_image)
    
    return enlarged_images


def show_enlarged_vs_srgan(srgan_image, enlarged_image):
    """shows image enlarged (without srgan) vs srgan image.
       right: Image enlarged with SRGAN, left: Image enlarged without SRGAN 
    """
    if(srgan_image and enlarged_image):
        Image.fromarray(np.hstack((np.array(srgan_image),np.array(enlarged_image)))).show()


if __name__ == "__main__":
    
    # parse args
    args = parser_init().parse_args()
    
    # get info on expected height & width. Set to first [0] since there is only one input / output.
    hef = HEF(args.hef)
    input_height, input_width, input_channels = hef.get_input_vstream_infos()[0].shape
    output_height, output_width, output_channels = hef.get_output_vstream_infos()[0].shape
    
    # load input image(s) and resize
    images_for_infer = load_input_images(args.images, input_height, input_width)
    
    # create and save enlarged images to compare to inference results
    output_path = Path(args.output)
    enlarged_images = create_and_save_enlarged_images(images_for_infer, output_height, output_width, output_path)
    
    # run inference
    output_images = run_inference(images_for_infer, hef, output_path)
    
    if args.show:
        enlarged_image = next(iter(enlarged_images), None)
        srgan_image = next(iter(output_images), None)
        show_enlarged_vs_srgan(srgan_image, enlarged_image)
    