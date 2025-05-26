"""Main app for Hailo Whisper"""

import time
import argparse
import os
from app.hailo_whisper_pipeline import HailoWhisperPipeline
from common.audio_utils import load_audio
from common.preprocessing import preprocess, improve_input_audio
from common.postprocessing import clean_transcription
from common.record_utils import record_audio


DURATION = 10  # recording duration in seconds


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
        "--multi-process-service", 
        action="store_true", 
        help="Enable multi-process service to run other models in addition to Whisper"
    )
    return parser.parse_args()


def get_encoder_hef_path(hw_arch):
    """
    Get the HEF path for the encoder based on the Hailo hardware architecture.

    Args:
        hw_arch (str): Hardware architecture ("hailo8" or "hailo8l").

    Returns:
        str: Path to the encoder HEF file.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    if hw_arch == "hailo8l":
        hef_path = os.path.join(base_path, 'hefs', 'h8l', 'tiny', 'tiny-whisper-encoder-10s_15dB_h8l.hef')
    else:
        hef_path = os.path.join(base_path, 'hefs', 'h8', 'tiny', 'tiny-whisper-encoder-10s_15dB.hef')
    if not os.path.exists(hef_path):
        raise FileNotFoundError(f"Encoder HEF file not found: {hef_path}. Please check the path.")
    return hef_path


def get_decoder_hef_path(hw_arch):
    """
    Get the HEF path for the decoder based on the Hailo hardware architecture and host type.

    Args:
        hw_arch (str): Hardware architecture ("hailo8" or "hailo8l").
        host (str): Host type ("x86" or "arm64").

    Returns:
        str: Path to the decoder HEF file.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    if hw_arch == "hailo8l":
        hef_path = os.path.join(base_path, "hefs", "h8l", "tiny", "tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef")
    else:
        hef_path = os.path.join(base_path, "hefs", "h8", "tiny", "tiny-whisper-decoder-fixed-sequence-matmul-split.hef")
    if not os.path.exists(hef_path):
        raise FileNotFoundError(f"Decoder HEF file not found: {hef_path}. Please check the path.")
    return hef_path


def main():
    """
    Main function to run the Hailo Whisper pipeline.
    """
    # Get command line arguments
    args = get_args()

    encoder_path = get_encoder_hef_path(args.hw_arch)
    decoder_path = get_decoder_hef_path(args.hw_arch)

    variant = "tiny"  # only tiny model is available for now

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
