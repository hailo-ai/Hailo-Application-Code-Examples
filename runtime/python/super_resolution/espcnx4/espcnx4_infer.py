#!/usr/bin/env python3

import numpy as np
from PIL import Image
from pathlib import Path
import os
from loguru import logger
import argparse
import cv2



from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)


YUV2RGB_mat = [[1.16438355, 1.16438355, 1.16438355],
                [0., -0.3917616, 2.01723105],

                [1.59602715, -0.81296805, 0.]]
        
RGB2YUV_mat = [[0.25678824, -0.14822353, 0.43921569],

                [0.50412941, -0.29099216, -0.36778824],

                [0.09790588, 0.43921569, -0.07142745]]

RGB2YUV_offset = [16, 128, 128]


def configure_and_get_network_group(hef, target):
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    return network_group


def create_input_output_vstream_params(network_group):
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    return input_vstreams_params, output_vstreams_params


def print_input_output_vstream_info(hef):
    input_vstream_info = hef.get_input_vstream_infos()
    output_vstream_info = hef.get_output_vstream_infos()

    for layer_info in input_vstream_info:
        logger.info('Input layer: {} {}'.format(layer_info.name, layer_info.shape))
    for layer_info in output_vstream_info:
        logger.info('Output layer: {} {}'.format(layer_info.name, layer_info.shape))
    
    return input_vstream_info, output_vstream_info


def preproc(image):
    # input RGB --> YUV 
    image = (np.matmul(image, RGB2YUV_mat ) + RGB2YUV_offset )/255
    # split channels
    y_channel, u_channel, v_channel = cv2.split(image)
    y_channel = np.expand_dims(y_channel, axis=-1)
    return y_channel


def postproc(image, input_image):
    image = image*255
    image = image.astype(np.uint8)
    # input RGB --> YUV 
    img_yuv = np.matmul(input_image, RGB2YUV_mat) + RGB2YUV_offset

    # Resizing naively to get the resized U and V channels
    img_yuv_resized = np.clip(cv2.resize(img_yuv,
                                       (image.shape[1],image.shape[0]),
                                     interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
                                     
    # concatenate the Super-Resolution Y channel with the resized U, V channels
    img_out = np.concatenate([image, img_yuv_resized[..., 1:]], axis=2)

    # YUV -->  RGB 
    img_out_rgb = np.clip(np.matmul(img_out - RGB2YUV_offset, YUV2RGB_mat), 0, 255).astype(np.uint8) 
    srgan_image = Image.fromarray(img_out_rgb.astype(np.uint8))
    
    return srgan_image


def run_inference_and_save_sr_images(images, hef, output_path):
    """Run inference on hailo-8

    Args:
        images (_type_): images to run inference on
        hef (HEF): hef file
        output_path (Path): output path (folder) to save output images
    """
    # Create folder for output images if doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    devices = Device.scan()

    # Configuration
    with VDevice(device_ids=devices) as target:
        network_group = configure_and_get_network_group(hef, target)
        network_group_params = network_group.create_params()
        input_vstreams_params, output_vstreams_params = create_input_output_vstream_params(network_group)
        
        # Print info of input & output
        input_vstream_info, output_vstream_info = print_input_output_vstream_info(hef)
        
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            for idx, image in enumerate(images):
                y_channel = preproc(image)
                input_data = {input_vstream_info[0].name: np.expand_dims(y_channel, axis=0)}
                with network_group.activate(network_group_params):
                    raw_image = infer_pipeline.infer(input_data)
                raw_image = raw_image[next(iter(output_vstream_info)).name][0]
                output_image = postproc(raw_image, image)
                image_name = str(idx) + '_sr.png'
                output_image.save(output_path/image_name, format='png')



def parser_init():
    """
    Function to declare an argument parser for all the possible command line arguments used in this script
    @return: instance of ArgumentParser
    """
    parser = argparse.ArgumentParser(description="SRgan inference")

    parser.add_argument(
        "hef",
        help="Path of espcn_x4_540_960.hef",
        default="espcn_x4_540_960.hef"
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
    

    return parser


def is_image(path):
    """Return True if path is a image of type: bmp, png, jpg, jpeg."""
    return (path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png') or path.endswith('.bmp'))


def load_input_images(images_path, height=None, width=None):
    """Load input images and resize to (height, width)

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
        # Running inference on a single image
        dir_path = Path(images_path)
        input_images = [img for img in os.listdir(images_path) if is_image(img)]
    else: 
        # Running inference on a single image
        if is_image(images_path):
          input_images = [str(images_path)]
    
    images_for_infer = list()
    for image in input_images:
        image = Image.open(dir_path/image)
        if height & width:
            image = np.array(image.resize((width,height), Image.BILINEAR)).astype(np.float32)
            images_for_infer.append(image)
    
    logger.info("running inference on images: {}", input_images)
    return images_for_infer


def save_org_images(images, output_path):
    # Create folder for output images if doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, image in enumerate(images):
        org_image = Image.fromarray((image).astype(np.uint8))
        image_name = str(idx) + '_org.png'
        org_image.save(output_path/image_name, format='png')
        
    
# Resize the image using Lanczos interpolation  
def improve_resolution_lanczos(image, height, width):
        resized_image = cv2.resize(image.astype(np.uint8), (width, height), interpolation=cv2.INTER_LANCZOS4)
        resized_image = Image.fromarray(resized_image)
        return resized_image


# Create and save open cv images using Lanczos4
def create_and_save_opencv_images(images, height, width):   
    for idx, image in enumerate(images):
        opencv_image = improve_resolution_lanczos(image, height, width)
        image_name = str(idx) + '_opencv.png'
        opencv_image.save(output_path/image_name, format='png')
    
    
if __name__ == "__main__":
    
    # Parse args
    args = parser_init().parse_args()
    
    # Get info on expected height & width. Set to first [0] since there is only one input / output.
    hef = HEF(args.hef)
    input_height, input_width, input_channels = hef.get_input_vstream_infos()[0].shape
    output_height, output_width, output_channels = hef.get_output_vstream_infos()[0].shape
    
    # Load input image(s) and resize
    images_for_infer = load_input_images(args.images, input_height, input_width)
    
    # save orginal images to compare to inference results
    output_path = Path(args.output)
    save_org_images(images_for_infer,  output_path)
    
    # Create and save opencv images to compare to inference results
    create_and_save_opencv_images(images_for_infer, output_height, output_width)
    
    # Run inference save super resolution image
    run_inference_and_save_sr_images(images_for_infer, hef, output_path)
    