"""Main app for Hailo Whisper"""

import time
import argparse
import os
import sys
from app.hailo_whisper_pipeline import HailoWhisperPipeline
from common.audio_utils import load_audio
from common.preprocessing import preprocess, improve_input_audio
from common.postprocessing import clean_transcription
from common.record_utils import record_audio
from app.whisper_hef_registry import HEF_REGISTRY


DURATION = 5  # recording duration in seconds


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Whisper Hailo Pipeline")
    parser.add_argument(
        "--reuse-audio", 
        action="store_true", 
        help="Reuse the previous audio file (sampled_audio.wav)"
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        choices=["hailo8", "hailo8l"],
        help="Hardware architecture to use (default: hailo8)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "tiny"],
        help="Whisper variant to use (default: base)"
    )
    parser.add_argument(
        "--multi-process-service", 
        action="store_true", 
        help="Enable multi-process service to run other models in addition to Whisper"
    )
    return parser.parse_args()


def get_hef_path(model_variant: str, hw_arch: str, component: str) -> str:
    """
    Method to retrieve HEF path.

    Args:
        model_variant (str): e.g. "tiny", "base"
        hw_arch (str): e.g. "hailo8", "hailo8l"
        component (str): "encoder" or "decoder"

    Returns:
        str: Absolute path to the requested HEF file.
    """
    try:
        hef_path = HEF_REGISTRY[model_variant][hw_arch][component]
    except KeyError as e:
        raise FileNotFoundError(
            f"HEF not available for model '{model_variant}' on hardware '{hw_arch}'."
        ) from e

    if not os.path.exists(hef_path):
        raise FileNotFoundError(f"HEF file not found at: {hef_path}\nIf not done yet, please run ./download_resources.sh from the app/ folder to download the required HEF files.")
    return hef_path


def main():
    """
    Main function to run the Hailo Whisper pipeline.
    """
    # Get command line arguments
    args = get_args()

    variant = args.variant
    print(f"Selected variant: Whisper {variant}")
    encoder_path = get_hef_path(variant, args.hw_arch, "encoder")
    decoder_path = get_hef_path(variant, args.hw_arch, "decoder")

    whisper_hailo = HailoWhisperPipeline(encoder_path, decoder_path, variant, multi_process_service=args.multi_process_service)
    print("Hailo Whisper pipeline initialized.")
    audio_path = "sampled_audio.wav"
    is_nhwc = True

    chunk_length = 10 if variant == "tiny" else 5

    while True:
        if args.reuse_audio:
            # Reuse the previous audio file
            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found. Please record audio first.")
                break
        else:
            user_input = input("\nPress Enter to start recording, or 'q' to quit: ")
            if user_input.lower() == "q":
                break
            # Record audio
            sampled_audio = record_audio(DURATION, audio_path=audio_path)

        # Process audio
        sampled_audio = load_audio(audio_path)

        sampled_audio, start_time = improve_input_audio(sampled_audio, vad=True)
        chunk_offset = start_time - 0.2
        if chunk_offset < 0:
            chunk_offset = 0

        mel_spectrograms = preprocess(
            sampled_audio,
            is_nhwc=is_nhwc,
            chunk_length=chunk_length,
            chunk_offset=chunk_offset
        )

        for mel in mel_spectrograms:
            whisper_hailo.send_data(mel)
            time.sleep(0.2)
            transcription = clean_transcription(whisper_hailo.get_transcription())
            print(f"\n{transcription}")

        if args.reuse_audio:
            break  # Exit the loop if reusing audio

    whisper_hailo.stop()


if __name__ == "__main__":
    main()
