from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm
)
from functools import partial
from loguru import logger
import numpy as np


class HailoInference:
    def __init__(self, hef_path: str, output_type: str = 'FLOAT32'):
        """
        Initialize the HailoInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
            output_type (str): Format type of the output stream. Defaults to 'FLOAT32'.
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

    def _create_vstream_params(self, output_type: str):
        """
        Create input and output stream parameters.

        Args:
            output_type (str): Format type of the output stream.

        Returns:
            tuple: Input and output stream parameters.
        """
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group, format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group, format_type=getattr(FormatType, output_type))
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

    def get_input_shape(self) -> tuple:
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


class HailoAsyncInference:
    def __init__(self, hef_path: str, input_queue, output_queue, batch_size: int = 1,
                 input_type: str = None, output_type: str = None):
        """
        Initialize the HailoAsyncInference class with the provided HEF model file path and input/output queues.

        Args:
            hef_path (str): Path to the HEF model file.
            input_queue (queue.Queue): Queue from which to pull input frames for inference.
            output_queue (queue.Queue): Queue to hold the inference results.
            batch_size (int): Batch size for inference. Defaults to 1.
            input_type (str): Format type of the input stream. Possible values: 'UINT8', 'FLOAT32'.
            output_type (str): Format type of the output stream. Possible values: 'UINT8', 'FLOAT32'.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        if output_type is not None or input_type is not None:
            self._set_input_output(input_type, output_type)
        self.output_type = output_type

        self.input_vstream_info, self.output_vstream_info = self._get_vstream_info()
        self.configured_infer_model = self.infer_model.configure()

    def _set_input_output(self, input_type: str = None, output_type: str = None):
        """
        Set the input and output layer information for the HEF model.

        Args:
            input_type (str): Format type of the input stream.
            output_type (str): Format type of the output stream.
        """
        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))
        if output_type is not None:
            self.infer_model.output().set_format_type(getattr(FormatType, output_type))

    def callback(self, completion_info, bindings_list, processed_batch):
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the inference task.
            bindings_list: List of binding objects containing input and output buffers.
            processed_batch: The processed batch of images.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {name: bindings.output(name).get_buffer() for name in bindings._output_names}
                self.output_queue.put((processed_batch[i], result))  # Add the result to the output queue

    def _get_vstream_info(self):
        """
        Get information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        return self.hef.get_input_vstream_infos(), self.hef.get_output_vstream_infos()

    def get_input_shape(self) -> tuple:
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info[0].shape  # Assumes that the model has one input

    def run(self):
        """
        Run asynchronous inference on the Hailo-8 device, processing batches from the input queue.

        Batches are fetched from the input queue until a sentinel value (None) is encountered.

        Returns:
            None: Results are added to the output queue.
        """
        while True:
            batch_frames = self.input_queue.get()  # Get the tuple (processed_batch, batch_array) from the queue
            if batch_frames is None:
                break  # Sentinel value to stop the inference loop

            bindings_list = []
            for frame in batch_frames:
                bindings = self._create_bindings()
                bindings.input().set_buffer(np.array(frame))
                bindings_list.append(bindings)

            self.configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            job = self.configured_infer_model.run_async(
                bindings_list, partial(self.callback, processed_batch=batch_frames, bindings_list=bindings_list)
            )

        job.wait(10000)  # Wait for the last job

    def _create_bindings(self):
        """
        Create bindings for input and output buffers.

        Returns:
            bindings: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            hef_output_type = str(self.hef.get_output_vstream_infos()[0].format.type).split(".")[1].lower()
            output_type = getattr(np, hef_output_type)
        else:
            output_type = getattr(np, self.output_type.lower())

        output_buffers = {name: np.empty(self.infer_model.output(name).shape, dtype=output_type)
                          for name in self.infer_model.output_names}
        return self.configured_infer_model.create_bindings(output_buffers=output_buffers)

    def release_device(self):
        """
        Release the Hailo device.
        """
        del self.configured_infer_model
        self.target.release()
