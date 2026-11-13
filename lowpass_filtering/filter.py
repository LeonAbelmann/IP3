"""
IP3 - M1: Load, play and visualize a heart sound recording.

Loads 'heart_single_channel_physionet_49829_TV.wav', shows the amplitude
as a function of time, and plays the sound over the default audio
output while the plot is open (with a moving cursor tracking playback).

The display focuses on the clean section (12-25 s) and clips the
short impulsive artifacts described in audio_description.md, so the
S1/S2 heart sounds are displayed at a proper scale.

Requires: numpy, scipy, matplotlib, sounddevice
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Audio file next to this script
AUDIO_FILE = Path(__file__).parent / "heart_single_channel_physionet_49829_TV.wav"

# --- Display / playback settings -------------------------------------------
# Clean section of the recording (seconds). The audio outside this
# window contains short impulsive artifacts that are several times
# larger than the heart sounds (see audio_description.md).
SECTION = (12.0, 25.0)   # None to show/play the whole file

# Amplitude clip for *display only* (data itself is not modified).
# Artifacts are clipped; S1/S2 remain at a readable scale.
CLIP_DISPLAY = True
CLIP_PERCENTILE = 99.9   # robust amplitude estimate, ignores artifact spikes

# Play only the displayed section (True) or the whole file (False)
PLAY_SECTION_ONLY = True


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
    n = len(data)
    t = np.arange(n) / fs  # time axis in seconds

    # Select the section to display
    if SECTION is not None:
        i0, i1 = int(SECTION[0] * fs), min(int(SECTION[1] * fs), n)
        t_plot, x_plot = t[i0:i1], data[i0:i1]
    else:
        t_plot, x_plot = t, data

    # Robust display limits: percentile ignores the artifact spikes
    if CLIP_DISPLAY:
        a_clip = np.percentile(np.abs(x_plot), CLIP_PERCENTILE)
        y_margin = 1.2 * a_clip
    else:
        y_margin = 1.05 * np.max(np.abs(x_plot))
    print(f"Display amplitude limit: +/-{y_margin:.3f} "
          f"({'clipped' if CLIP_DISPLAY else 'full scale'})")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_plot, x_plot, lw=0.5, color="tab:blue")
    line, = ax.plot([], [], color="tab:red", lw=1.5)  # playback cursor
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [-]")
    ax.set_title("heart_single_channel_physionet_49829_TV.wav")
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_ylim(-y_margin, y_margin)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Choose what to play
    if PLAY_SECTION_ONLY and SECTION is not None:
        audio, t_start = data[i0:i1].copy(), SECTION[0]
        print(f"Playing section {SECTION[0]}-{SECTION[1]} s only...")
    else:
        audio, t_start = data.copy(), 0.0
        print("Playing whole file...")
    duration = len(audio) / fs

    sd.play(audio, fs)  # non-blocking
    t0 = time.perf_counter()  # wall-clock reference for the cursor

    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= duration or not sd.get_stream().active:
                break
            # Current playback position in seconds (absolute time axis)
            current = t_start + elapsed
            line.set_data([current, current], [-y_margin, y_margin])
            plt.pause(0.05)
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
