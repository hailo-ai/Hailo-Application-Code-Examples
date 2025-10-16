#!/usr/bin/bash
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/yolov5m_wo_spp.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/onnxs/yolov5m_wo_spp_postprocess_v1.onnx

if [ -d "/opt/onnxruntime-linux-x64" ]; then
	echo "ONNXRuntime folder already exists in /opt. No need to download and extract."
else
	wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/external_tar_files/onnxruntime-linux-x64-1.18.1.tgz
	sudo cp onnxruntime-linux-x64-1.18.1.tgz /opt
	cd /opt
       	sudo tar -xvf onnxruntime-linux-x64-1.18.1.tgz
	sudo rm onnxruntime-linux-x64-1.18.1.tgz
       	cd -
	sudo rm onnxruntime-linux-x64-1.18.1.tgz
fi
