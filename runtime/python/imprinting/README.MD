# create compilation docker
sudo docker run --gpus all --name finkel_hrt_dfc -it -p 8147:8147 -v /local:/local_host -v /data:/data -v /work:/work repo-docker.int.hailo.ai/hailo_repo_init_cpu:hrt_dfc /bin/bash --login

# in docker:
bash build.sh /opt/poky/4.0.2 

# put built .so on H15:
scp build/aarch64/libfeature_extractor_cross_compilation_h15.so root@192.168.241.118:/home/root/pocs/online_training/target/

# on PC:
create_calib, parse, optimize, 
compile (for H15!): ClientRunner(har=quantized_model_har_path, hw_arch='hailo15h')

# move to target:
FE hef, fc.npz

# on target:
bash imprint.sh

# testing
..incomplete, the imprint.py retests on the training data itself. maybe can try held-out test..
   Currently my inference is through the script of jpgs2txt anyways, so a separate "test script" would be a boring duplication
  Old infer (hailort_infer.py) can't run on H15, but can on a RPi maybe. 
    

