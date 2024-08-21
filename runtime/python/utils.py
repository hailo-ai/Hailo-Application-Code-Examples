from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm
)
from functools import partial
from loguru import logger
import numpy as np


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

    def _set_input_type(self, input_type: str = None) -> None:
        """
        Set the input type for the HEF model.

        Args:
            input_type (str): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type: str = None) -> None:
        """
        Set the output type for the HEF model.

        Args:
            output_type (str): Format type of the output stream.
        """
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
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes that the model has one input

    def run(self):
        """
        Run asynchronous inference on the Hailo device, processing batches from the input queue.

        Batches are fetched from the input queue until a sentinel value (None) is encountered.

        Returns:
            None: Results are added to the output queue.
        """
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_frames = self.input_queue.get()  # Get the tuple (processed_batch, batch_array) from the queue
                if batch_frames is None:
                    break  # Sentinel value to stop the inference loop

                bindings_list = []
                for frame in batch_frames:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list, partial(self.callback, processed_batch=batch_frames, bindings_list=bindings_list)
                )

                job.wait(10000)  # Wait for the last job

    def _create_bindings(self, configured_infer_model):
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
        return configured_infer_model.create_bindings(output_buffers=output_buffers)
