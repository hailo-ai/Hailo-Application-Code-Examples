bash gst_hailo_infer_folder.sh ../training_data/001*
bash gst_hailo_infer_folder.sh ../training_data/002.*

IMPRINT_CMD="python3 ./imprint_weights_from_txt.py  \                  
                  --npz-input ../onnx/resnet_v1_18_fc.npz \
                  --npz-output fc \
                  --input-images ../training_data/ "

eval ${IMPRINT_CMD}
