"""
IP3 - M1: Load, play and visualize a heart sound recording.

Loads 'heart_single_channel_physionet_49829_TV.wav', shows the amplitude
as a function of time, and plays the sound over the default audio
output while the plot is open (with a moving cursor tracking playback).

Requires: numpy, scipy, matplotlib, sounddevice
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def play_and_plot(data, fs):
    """Play audio while showing the waveform with a moving playback cursor."""
    t = np.arange(len(data)) / fs  # time axis in seconds

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, data, lw=0.5, color="tab:blue")
    line, = ax.plot([], [], color="tab:red", lw=1.5)  # playback cursor
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [-]")
    ax.set_title("heart_single_channel_physionet_49829_TV.wav")
    ax.set_xlim(0, t[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    print("Playing (close the plot window to stop)...")
    sd.play(data, fs)  # non-blocking: playback starts in the background

    try:
        while sd.get_stream().active:
            # Current playback position in seconds
            current = (len(data) - sd.get_stream().read_available) / fs
            line.set_data([current, current], [np.min(data), np.max(data)])
            plt.pause(0.05)  # redraw while letting the GUI breathe
        sd.wait()
    except Exception:
        sd.stop()
        raise
    finally:
        print("Done.")
    plt.show()  # keep the plot open after playback finishes


def main():
    if not AUDIO_FILE.exists():
        sys.exit(f"Audio file not found: {AUDIO_FILE}\n"
                 f"Place it in the same directory as filter.py.")
    data, fs = load_audio(AUDIO_FILE)
    play_and_plot(data, fs)


if __name__ == "__main__":
    main()
