from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm)
from functools import partial
from loguru import logger
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Generator, Optional, Tuple
import queue

IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')


class HailoAsyncInference:
    def __init__(
        self, hef_path: str, input_queue: queue.Queue,
        output_queue: queue.Queue, batch_size: int = 1,
        input_type: Optional[str] = None, output_type: Optional[str] = None
    ) -> None:
        """
        Initialize the HailoAsyncInference class with the provided HEF model 
        file path and input/output queues.

        Args:
            hef_path (str): Path to the HEF model file.
            input_queue (queue.Queue): Queue from which to pull input frames 
                                       for inference.
            output_queue (queue.Queue): Queue to hold the inference results.
            batch_size (int): Batch size for inference. Defaults to 1.
            input_type (Optional[str]): Format type of the input stream. 
                                        Possible values: 'UINT8', 'UINT16'.
            output_type (Optional[str]): Format type of the output stream. 
                                         Possible values: 'UINT8', 'UINT16', 'FLOAT32'.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        params = VDevice.create_params()
        
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        self.output_type = output_type

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type: Optional[str] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type of all of them.

        Args:
            output_type (Optional[str]): Format type of the output stream.
        """
        self.infer_model.output().set_format_type(getattr(FormatType, output_type))

    def callback(
        self, completion_info, bindings_list: list, processed_batch: list
    ) -> None:
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the 
                             inference task.
            bindings_list (list): List of binding objects containing input 
                                  and output buffers.
            processed_batch (list): The processed batch of images.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: bindings.output(name).get_buffer() 
                        for name in bindings._output_names
                    }
                self.output_queue.put((processed_batch[i], result))

    def _get_vstream_info(self) -> Tuple[list, list]:
        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

    def run(self) -> None:
        """
        Run asynchronous inference on the Hailo device, processing batches 
        from the input queue.

        Batches are fetched from the input queue until a sentinel value 
        (None) is encountered.
        """
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_frames = self.input_queue.get()  
                # Get the tuple (processed_batch, batch_array) from the queue
                if batch_frames is None:
                    break  # Sentinel value to stop the inference loop

                bindings_list = []
                for frame in batch_frames:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list, partial(
                        self.callback, processed_batch=batch_frames, 
                        bindings_list=bindings_list
                    )
                )

                job.wait(10000)  # Wait for the last job

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            hef_output_type = str(
                self.hef.get_output_vstream_infos()[0].format.type
            ).split(".")[1].lower()
            output_type = getattr(np, hef_output_type)
        else:
            output_type = getattr(np, self.output_type.lower())

        output_buffers = {
            name: np.empty(
                self.infer_model.output(name).shape, dtype=output_type
            )
            for name in self.infer_model.output_names
        }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )

class HailoInference:
    def __init__(self, hef_path, output_type='FLOAT32'):
        """
        Initialize the HailoInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
        """
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        """
        Configure the Hailo device and get the network group.

        Returns:
            NetworkGroup: Configured network group.
        """
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.target.configure(self.hef, configure_params)[0]
        return network_group

    def _create_vstream_params(self, output_type):
        """
        Create input and output stream parameters.

        Args:
            output_type (str): Format type of the output stream.

        Returns:
            tuple: Input and output stream parameters.
        """
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        """
        Get and print information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        for layer_info in input_vstream_info:
            logger.info(f'Input layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')
        for layer_info in output_vstream_info:
            logger.info(f'Output layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')

        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info[0].shape  # Assumes that the model has one input

    def run(self, input_data):
        """
        Run inference on Hailo-8 device.

        Args:
            input_data (np.ndarray, dict, list, tuple): Input data for inference.

        Returns:
            np.ndarray: Inference output.
        """
        input_dict = self._prepare_input_data(input_data)

        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)[self.output_vstream_info[0].name]

        return output

    def _prepare_input_data(self, input_data):
        """
        Prepare input data for inference.

        Args:
            input_data (np.ndarray, dict, list, tuple): Input data for inference.

        Returns:
            dict: Prepared input data.
        """
        input_dict = {}
        if isinstance(input_data, dict):
            return input_data
        elif isinstance(input_data, (list, tuple)):
            for layer_info in self.input_vstream_info:
                input_dict[layer_info.name] = input_data
        else:
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)
            input_dict[self.input_vstream_info[0].name] = input_data

        return input_dict

    def release_device(self):
        """
        Release the Hailo device.
        """
        self.target.release()
        
def load_input_images(images_path: str) -> List[Image.Image]:
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[Image.Image]: List of PIL.Image.Image objects.
    """
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path)]
    elif path.is_dir():
        return [
            Image.open(img) for img in path.glob("*") 
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []


def validate_images(images: List[Image.Image], batch_size: int) -> None:
    """
    Validate that images exist and are properly divisible by the batch size.

    Args:
        images (List[Image.Image]): List of images.
        batch_size (int): Number of images per batch.

    Raises:
        ValueError: If images list is empty or not divisible by batch size.
    """
    if not images:
        raise ValueError(
            'No valid images found in the specified path.'
        )
    
    if len(images) % batch_size != 0:
        raise ValueError(
            'The number of input images should be divisible by the batch size '
            'without any remainder.'
        )


def divide_list_to_batches(
    images_list: List[Image.Image], batch_size: int
) -> Generator[List[Image.Image], None, None]:
    """
    Divide the list of images into batches.

    Args:
        images_list (List[Image.Image]): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        Generator[List[Image.Image], None, None]: Generator yielding batches 
                                                  of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]
