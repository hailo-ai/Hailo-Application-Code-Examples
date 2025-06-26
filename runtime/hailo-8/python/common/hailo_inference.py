from typing import Tuple, Dict
from typing import Callable, Optional
from functools import partial
import queue
from loguru import logger
import numpy as np
from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)
from hailo_platform.pyhailort.pyhailort import FormatOrder


class HailoAsyncInference:
    def __init__(
        self, hef_path: str, input_queue: queue.Queue, callback: Callable, batch_size: int = 1,
            input_type: Optional[str] = None, output_type: Optional[str] = None,
            send_original_frame: bool = False) -> None:

        """
        Initialize the HailoAsyncInference class with the provided HEF model 
        file path and input/output queues.

        Args:
            hef_path (str): Path to the HEF model file for inference.
            input_queue (queue.Queue): Queue containing preprocessed frames or data.
            callback (Callable): Function to be called with inference results.
            batch_size (int, optional): Number of inputs processed in a single batch.
                Defaults to 1.
            input_type (Optional[str], optional): Input data type format. Common
                values: 'UINT8', 'UINT16', 'FLOAT32'.
            output_type (Optional[str], optional): Output data type format. Common
                values: 'UINT8', 'UINT16', 'FLOAT32'.
            send_original_frame (bool, optional): If True, passes the original input
                frame to the callback. Defaults to False.
        """

        self.input_queue = input_queue
        params = VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        self._set_input_type(input_type)
        self._set_output_type(output_type)
        self.send_original_frame = send_original_frame
        self.callback_fn = callback

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """

        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type: Optional[str] = None) -> None:
        """
        Set the output type for each model output.

        Args:
            output_type (Optional[str]): Desired output data type. Common values:
                'UINT8', 'UINT16', 'FLOAT32'.
        """

        self.nms_postprocess_enabled = False

        # If the model uses HAILO_NMS_WITH_BYTE_MASK format (e.g.,instance segmentation),
        if self.infer_model.outputs[0].format.order == FormatOrder.HAILO_NMS_WITH_BYTE_MASK:
            # Use UINT8 and skip setting output formats
            self.nms_postprocess_enabled = True
            self.output_type = self._output_data_type2dict("UINT8")
            return

        # Otherwise, set the format type based on the provided output_type argument
        self.output_type = self._output_data_type2dict(output_type)

        # Apply format to each output layer
        for name, dtype in self.output_type.items():
            self.infer_model.output(name).set_format_type(getattr(FormatType, dtype))



    def default_callback(
            self,
            completion_info,
            bindings_list: list,
            input_batch: list,
            output_queue: queue.Queue
    ) -> None:
        """
        Default callback to handle inference results and push them to a queue.

        Args:
            completion_info: Hailo inference completion info.
            bindings_list (list): Output bindings for each inference.
            input_batch (list): Original input frames.
            output_queue (queue.Queue): Queue to push output results to.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(
                            bindings.output(name).get_buffer(), axis=0
                        )
                        for name in bindings._output_names
                    }
                output_queue.put((input_batch[i], result))


    def get_vstream_info(self) -> Tuple[list, list]:

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

    def get_hef(self) -> HEF:
        """
        Get a HEF instance
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

    def run(self) -> None:
        """
        Main inference loop. Continuously pulls batches from the input queue,
        runs async inference, and triggers the callback with results.
        """

        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_data = self.input_queue.get()
                if batch_data is None:
                    break  # Sentinel value to stop the inference loop

                # Unpack original and preprocessed batch if needed
                if self.send_original_frame:
                    original_batch, preprocessed_batch = batch_data
                else:
                    preprocessed_batch = batch_data

                bindings_list = []
                for frame in preprocessed_batch:
                    # Create bindings for each frame in the batch
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000)

                # Run inference asynchronously and attach the callback
                job = configured_infer_model.run_async(
                    bindings_list,
                    partial(
                        self.callback_fn,
                        input_batch=(
                            original_batch if self.send_original_frame
                            else preprocessed_batch
                        ),
                        bindings_list=bindings_list,
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
        output_buffers = {
            name: np.empty(
                self.infer_model.output(name).shape,
                dtype=(getattr(np, self.output_type[name].lower()))
            )
        for name in self.output_type
        }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )


    def is_nms_postprocess_enabled(self) -> bool:
        """
        Returns True if the HEF model includes an NMS postprocess node.
        """
        return self.nms_postprocess_enabled

    def _output_data_type2dict(self, data_type: Optional[str]) -> Dict[str, str]:
        """
        Generate a dictionary mapping each output layer name to its corresponding
        data type. If no data type is provided, use the type defined in the HEF.

        Args:
            data_type (Optional[str]): The desired data type for all output layers.
                                       Valid values: 'float32', 'uint8', 'uint16'.
                                       If None, uses types from the HEF metadata.

        Returns:
            Dict[str, str]: A dictionary mapping output layer names to data types.
        """
        valid_types = {"float32", "uint8", "uint16"}
        data_type_dict = {}

        for output_info in self.hef.get_output_vstream_infos():
            name = output_info.name
            if data_type is None:
                # Extract type from HEF metadata
                hef_type = str(output_info.format.type).split(".")[-1]
                data_type_dict[name] = hef_type
            else:
                if data_type.lower() not in valid_types:
                    raise ValueError(f"Invalid data_type: {data_type}. Must be one of {valid_types}")
                data_type_dict[name] = data_type

        return data_type_dict