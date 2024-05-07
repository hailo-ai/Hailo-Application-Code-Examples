# Clip Inference Example

This is an example application to run a CLIP inference on a video in realtime.
The image embeddings are accelerated by Hailo-8 AI processor.
The text embeddings are running on the host. Text embeddings are sparse and should be calculated only once per text. If they do not need to be updated in real time they can be saved to a JSON file and loaded on the next run.
As default the app starts w/o enabling online test embeddings. This is done to speed up load time and save memory. It also allows to run on low memory hosts like the RPi.

## Prerequisites
This example was tested on Hailo's TAPPAS rpi_v3.27.0
You'll need it installed on your system.

#### These packages are required for the CLIP
```
sudo apt-get -y install libblas-dev \
nlohmann-json3-dev nlohmann-json3-dev
```
### hailo_tappas pkg_config
This applications is using hailo_tapps pkg_config. 
To test installation and setup envirounment run:
```
source setup_env.sh
```
If you get response which looks like this you're good to go.

```bash
Setting up the environment...
TAPPAS_WORKSPACE set to /home/giladn/TAPPAS/tappas/
Activating virtual environment...
```
If output is not as expected make sure:
- You have read rights to /opt/hailo/tappas/pkgconfig/hailo_tappas.pc
- PKG_CONFIG_PATH includes this path.
To add it to PKG_CONFIG_PATH run:
```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/hailo/tappas/pkgconfig/
```

## CPP code compilation
Some CPP code is used in this app for post processes and cropping. These should be compiled before running the example. It uses Hailo pkg-config to find the required libraries.

The compilation script is **compile_postprocess.sh** you can run it manually but it will be ran automatically when installing the package.
The postprocess so files will be installed under resources dir.


## installation
##### Make sure you run `source setup_env.sh` before running install.
To install the application run in the application root dir:
```bash 
python3 -m pip install -e .
```
This will install the app as a python package in "Editable" mode. It will also compile the CPP code and download required HEF files.

## Usage
##### Make sure you run `source setup_env.sh` before running.

Run the example:
```bash
clip_app
```
It can also be ran directly from the source (from the application root dir):
```bash
python3 -m clip_app.clip_app
```

On the first run clip will download the required models. This will happen only once.

### Arguments
```bash
clip_app -h
usage: clip_app [-h] [--sync] [--input INPUT] [--dump-dot] [--detection-threshold DETECTION_THRESHOLD] [--detector {person,none}] [--onnx-runtime] [--clip-runtime]
                [--json-path JSON_PATH] [--multi-stream]

Hailo online clip app

options:
  -h, --help            show this help message and exit
  --sync                Enable display sink sync.
  --input INPUT, -i INPUT
                        URI of the input stream.
  --dump-dot            Dump the pipeline graph to a dot file.
  --detection-threshold DETECTION_THRESHOLD
                        Detection threshold
  --detector {person,none}, -d {person,none}
                        Which detection pipeline to use.
  --onnx-runtime        Not supported. Requires ONNX runtime for text embedding.
  --clip-runtime        When set app will use clip pythoch runtime for text embedding.
  --json-path JSON_PATH
                        Path to json file to load and save embeddings. If not set embeddings.json will be used.
  --multi-stream        When set app will use multi stream pipeline. In this mode detector is set to person.
```

### Modes
- The default mode (--detector none) will run only the clip inference on the entire frame. This is the mode CLIP is trained for, and will give the best results. In this mode CLIP is acting as a classifier describing the entire frame. CLIP will be ran on every frame.
- Person mode (--detector person) will run the clip inference only on the detected persons. In this mode we first run a person detector and then run clip on the detected persons. This mode does not exaclty fit the data base CLIP was trained on but gives a good use case for using CLIP as a detector. In this mode CLIP is acting as a person classifier. In this mode CLIP will be ran only on detected persons. To reduce the number of clip inferences we run clip only every second per tracked person. This can be changed in the code.
- Multi stream mode (--multi-stream) will run clip on 4 streams. This mode is presenting a use case of using CLIP to monitor multiple streams. In this mode we are running with a person detector. The App will bring the most relevant stream to the foreground. If the searched person is not found in any of the streams a random stream will be selected.
  
**Note Multi stream mode requires a strong machine.**

### Online text embeddings
- To run with online text embeddings run with the --clip-runtime flag. 
- This will run the text embeddings on the host. You will be able to change the text on the fly. This mode might not work on weak machines. It requires a host with enough memory to run the text embeddings model (run on CPU).
- You can set which json file to use for saving and loading embeddings using the --json-path flag. If not set embeddings.json will be used.
- If you wish to load / save your JSON, use --json-path flag explicitly.


**As default the online text embeddings are disabled. This is done to speed up load time and save memory. If changing the classes is not required in the final application you can just load the embeddings from JSON. It also allows to run on low memory hosts like the RPi 4.** 

### Offline text embeddings
- You can save the embeddings to a JSON file and load them on the next run. This will not require to run the text embeddings on the host.
- If you need to prepare text embeddings on a weak machine you can use the 'text_image_matcher' tool. This tool will run the text embeddings on the host and save them to a JSON file. Without running the intire pipeline.
``` bash
text_image_matcher -h
usage: text_image_matcher [-h] [--output OUTPUT] [--interactive] [--image-path IMAGE_PATH] [--texts-list TEXTS_LIST [TEXTS_LIST ...]]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       output file name default=text_embeddings.json
  --interactive         input text from interactive shell
  --image-path IMAGE_PATH
                        Optional, path to image file to match. Note image embeddings are not running on Hailo here.
  --texts-list TEXTS_LIST [TEXTS_LIST ...]
                        A list of texts to add to the matcher, the first one will be the searched text, the others will be considered negative
                        prompts. Example: --texts-list "cat" "dog" "yellow car"
```

## Tips for good prompt usage
- Keep in mind that the network was trained on image + caption pairs. You text description should be somewhat similar. For example a text description of " A photo of a cat" will give better score than "cat".
- The app got a "pefix" pre defined to "A photo of a " you can change it in the "TextImageMatcher" class.
- The pipeline output will select on of the classes as "the best one". There is no 'background' class. You should define a "negative" prompt (or prompts) to be used as "backgroud". When set as negative the class will be used in the "best match" algorithm but will not be showed in the output.
- You can also use "threshold" to fine tune detection sensitivity. However, using "negative" prompts are better for detecting specific classes.
- Negative prompts should be used to "peal off" similar classifications to your target. For example "a man with red shirt" will have a high score also for just a man, or a shirt with a different color. Add negative prompts like "a man with blue shirt" to make sure you do not get lots of false classification.
- Play around with prompts to see what works best for you application.