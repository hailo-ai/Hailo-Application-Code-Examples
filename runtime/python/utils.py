from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)

from loguru import logger
import numpy as np

class HailoInference():
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

        Args:
            hef (HEF): HEF model object.
            target (VDevice): Hailo device target.

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
            network_group (NetworkGroup): Configured network group.

        Returns:
            InputVStreamParams, OutputVStreamParams: Input and output stream parameters.
        """
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=self.hef.get_input_vstream_infos()[0].format.type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params
    
    def _get_and_print_vstream_info(self):
        """
        Get and print information about input and output stream layers.

        Args:
            hef (HEF): HEF model object.

        Returns:
            list, list: List of input stream layer information, List of output stream layer information.
        """
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        for layer_info in input_vstream_info:
            logger.info('Input layer: {} {} {}'.format(layer_info.name, layer_info.shape, layer_info.format.type))
        for layer_info in output_vstream_info:
            logger.info('Output layer: {} {} {}'.format(layer_info.name, layer_info.shape, layer_info.format.type))
        
        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape # Assumes that the model has one input
    
    def run(self, input_data):
        """
        Run inference on Hailo-8 device.

        Args:
            input data : Accepts multiple formats - dictionary, list, tuple or array.

        Returns:
            numpy.ndarray: Inference output.
        """
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:            
            input_dict = {}
        
            if isinstance(input_data, dict):
            # Input data is already a dictionary
                input_dict = {k: v.copy() for k, v in input_data.items()}
            elif isinstance(input_data, (list, tuple)):
            # Input data is a list or tuple
                for i, layer_info in enumerate(self.input_vstream_info):
                    input_dict[layer_info.name] = np.expand_dims(input_data[i].copy(), axis=0)
            else:
            # Input data is a single array
                input_dict[self.input_vstream_info[0].name] = np.expand_dims(input_data.copy(), axis=0)
                
            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)

        return output

    def release_device(self):
        """
        Release the Hailo device.
        """
        self.target.release()