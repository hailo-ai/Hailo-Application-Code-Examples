# Automatic Speech Recognition with OpenAI Whisper model

This application performs a speech-to-text transcription using OpenAI's *Whisper-tiny* model on the Hailo-8/8L AI accelerator.

## Prerequisites

Ensure your system matches the following requirements before proceeding:

- Platforms tested: x86, Raspberry Pi 5
- OS: Ubuntu 22 (x86) or Raspberry OS.
- **HailoRT 4.20 or 4.21** and the corresponding **PCIe driver** must be installed. You can download them from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- **ffmpeg** and **libportaudio2** installed for audio processing.
  ```
  sudo apt update
  sudo apt install ffmpeg
  sudo apt install libportaudio2
  ```
- **Python 3.10 or 3.11** installed.

## Installation - Inference only

Follow these steps to set up the environment and install dependencies for inference:

1. Clone this repository:

  ```sh
  git clone git@github.com:topper-ai/WhisperingHeights.git
  cd WhisperingHeights
  ```
  If you have any authentization issues, add your SSH key or download the zip.

2. Run the setup script to install dependencies:

  ```sh
  python3 setup.py
  ```

3. Activate the virtual environment from the repository root folder:

  ```sh
  source whisper_env/bin/activate
  ```

4. Install pyHailoRT inside the virtual environment (must be downloaded from the Hailo Developer Zone), for example:
  ```sh
  pip install hailort-4.20.0-cp310-cp310-linux_x86_64.whl
  ```
  The pyHailoRT version must match the installed HailoRT version.

## Before running the app

- Make sure you have a microphone connected to your system. If you have multiple microphones connected, please make sure the proper one is selected in the system configuration, and that the input volume is set to a medium/high level.  
  A good quality microphone (or a USB camera) is suggested to acquire the audio.
- The application allows the user to acquire and process an audio sample up to 10 seconds long.
- The current pipeline supports **English language only**.

## Usage from CLI
1. Activate the virtual environment from the repository root folder:

  ```sh
  source whisper_env/bin/activate
  ```
2. Run the command line app (from the root folder)
  ```sh
  python3 -m app.app_hailo_whisper
  ```

### Command line arguments
Use the `python3 -m app.app_hailo_whisper --help` command to print the helper.

The following command line options are available:

- **--reuse-audio**: Reloads the audio from the previous run.
- **--hw-arch**: Selects the correct Whisper models from Hailo-8 / 8L.
- **--multi-process-service**: Enables the multi-process service, to run other models on the same chip in addition to Whisper


## Additional notes

- This application is just an example to show how to run a Whisper-based pipeline on the Hailo-8/8L AI accelerator, and it is not focused on optimal pre/post-processing.
- Torch is still required for pre-processing. It will be removed in the next release.
- We are considering future improvements, like:
  - Release scripts for model conversion
  - Optimized post-processing to improve transcription's accuracy
  - Additional models support
  - Dedicated C++ implementation  

  Feel free to share your suggestions in the [Hailo Community](https://community.hailo.ai/) regarding how this application can be improved.

## Troubleshooting

- Make sure that the microphone is connected to your host and that it can be detected by the system.
- Post-processing is being applied to improve the quality of the transcription, e.g. applying peanlty on repeated tokens and removing model's hallucinations (if any). These methods can be modified by the user to find an optimal solution.
- the `--reuse-audio` flag can be used to load the audio acquired during the previous run, for debugging purposes.


## Disclaimer
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
