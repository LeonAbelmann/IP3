"""
IP3 - M1: Load and play a heart sound recording.

Plays 'heart_single_channel_physionet_49829_TV.wav' over the default
audio output (notebook loudspeaker).

Requires: numpy, scipy, sounddevice  (pip install sounddevice)
"""

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Audio file next to this script
AUDIO_FILE = Path(__file__).parent / "heart_single_channel_physionet_49829_TV.wav"


def load_audio(path: Path):
    """Load a wav file, return (data as float array, sample rate)."""
    fs, data = wavfile.read(path)

    # Convert to float in [-1, 1] regardless of original dtype
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:  # 8-bit wav is unsigned, centered at 128
        data = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype != np.float32 and data.dtype != np.float64:
        raise ValueError(f"Unsupported wav dtype: {data.dtype}")

    # If stereo, take the first channel
    if data.ndim > 1:
        data = data[:, 0]

    print(f"Loaded {path.name}: {len(data)} samples @ {fs} Hz "
          f"({len(data) / fs:.1f} s)")
    return data, fs


def play_audio(data, fs):
    """Play audio over the default output device (blocking)."""
    print("Playing... (Ctrl-C to stop)")
    try:
        sd.play(data, fs)
        sd.wait()  # block until playback is finished
    except KeyboardInterrupt:
        sd.stop()
    print("Done.")


def main():
    if not AUDIO_FILE.exists():
        sys.exit(f"Audio file not found: {AUDIO_FILE}\n"
                 f"Place it in the same directory as filter.py.")
    data, fs = load_audio(AUDIO_FILE)
    play_audio(data, fs)


if __name__ == "__main__":
    main()
