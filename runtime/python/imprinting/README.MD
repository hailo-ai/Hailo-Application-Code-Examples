## Weight imprinting POC ## 

This is a small POC of upgrading a classifier deployed on H15 with additional classes, by providing a few dozen images from each class.
The deployed HEF is feature-extractor (FH) only, while the fully-connected (FC) layer aka linear classifier is left on the host, performed in python using an NPZ. The FC matrix columns is the centroid vectors for classes in the space of feature vectors that FE creates. So we create centroids for new classes by averaging the FE outputs for the mini-train images, and concat them to the FC matrix, e.g. 512x1000 --> 512x1002. 


### PC preparations
1. Invoking DFC (parse, calib, optimize, compile (for H15)) to create a decapitated ResNet-18 feature extractor HEF and FC NPZ.
2. Cross-compile a postproc-filter .so to be used within hailo-gst pipeline to print a FE output vectors into text file, line per image.
   See runtime/cpp/embedding/H15_img2vec

Copy to H15 the "target" folder, adding the .so, HEF, NPZ
Alongside that, put the "training_data" folder with contents formatted as: 001/0000.jpg, 001/0001.jpg, .., 002/0000.jpg etc. The subfolders correspond to the new classes. Images need to be in a standard format, e.g. 640x480 . The training_data has a couple scripts that can help make an image collection conform


### Running imprint on H15 
Just launch 'bash imprint.sh' 

#### testing
..incomplete, the imprint.py retests on the training data itself. maybe can try a "held-out set" test..
Currently my inference is through the script of jpgs2txt anyways, so a separate "test script" would be a boring duplication
Old infer (hailort_infer.py) can't run on H15, but can on a RPi maybe. 
    

