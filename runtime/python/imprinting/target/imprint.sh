bash gst_hailo_infer_folder.sh ../training_data/001*
bash gst_hailo_infer_folder.sh ../training_data/002.*

IMPRINT_CMD="python3 ./imprint_weights_from_txt.py  \
                  --hef ../hef/resnet_v1_18_featext.hef \
                  --npz_input ../onnx/resnet_v1_18_fc.npz \
                  --npz_output fc \
                  --input-images ../training_data/ "

eval ${IMPRINT_CMD}
