import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import select
import sys
import queue
import time


# Whisper expects 16kHz mono audio
SAMPLE_RATE = 16000
CHANNELS = 1

def enter_pressed():
    return select.select([sys.stdin], [], [], 0.0)[0]

def record_audio(duration, audio_path):
    """
    Record audio from the microphone and save it as a WAV file. The user has the possibility to stop the recording earlier by pressing Enter on the keyboard.

    Args:
        duration (int): Duration of the recording in seconds.

    Returns:
        np.ndarray: Recorded audio data.
    """
    q = queue.Queue()
    recorded_frames = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)
        q.put(indata.copy())

    print(f"Recording for up to {duration} seconds. Press Enter to stop early...")

    start_time = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype="float32",
                        callback=audio_callback):
        # Set stdin to non-blocking line-buffered mode
        sys.stdin = open('/dev/stdin')
        while True:
            if time.time() - start_time >= duration:
                print("Max duration reached.")
                break
            if enter_pressed():
                sys.stdin.read(1)  # consume the newline
                print("Early stop requested.")
                break
            try:
                frame = q.get(timeout=0.1)
                recorded_frames.append(frame)
            except queue.Empty:
                continue

    print("Recording finished. Processing...")

    audio_data = np.concatenate(recorded_frames, axis=0)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    wav.write(audio_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    return audio_data
