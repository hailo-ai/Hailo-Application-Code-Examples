#!/usr/bin/bash
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/yolov5m_wo_spp_60p_nms_on_hailo.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/onnxs/yolov5m_wo_spp_postprocess.onnx

if [ -d "/opt/onnxruntime-linux-x64-1.13.1" ]; then
	echo "ONNXRuntime folder already exists in /opt. No need to download and extract."
else
	wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/external_tar_files/onnxruntime-linux-x64-1.13.1.tgz
	sudo tar -xvf onnxruntime-linux-x64-1.13.1.tgz
	sudo cp -r onnxruntime-linux-x64-1.13.1 /opt && sudo rm -rf onnxruntime-linux-x64-1.13.1
fi
