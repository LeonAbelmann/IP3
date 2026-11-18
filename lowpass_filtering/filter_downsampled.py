"""
IP3 - M1 (section 4.3): Downsampling experiment.

Duplicate of filter.py with a downsampling stage added: after bandpass
filtering (10-800 Hz, zero-phase) at the original Fs, the signal is
decimated by M = 4: Fs 4 kHz -> 1 kHz.

Deliberately kept: the 800 Hz upper cutoff. Compare with filter.py to
see what happens:

1. The 500-800 Hz part of the filtered signal lies ABOVE the new
   Nyquist frequency (500 Hz) and is aliased down by naive decimation.
2. Designing the same 800 Hz filter directly at Fs = 1 kHz is
   impossible (cutoff > Nyquist). The script demonstrates this error.

Requires: numpy, scipy, matplotlib, sounddevice
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile

# Audio file next to this script
AUDIO_FILE = Path(__file__).parent / "heart_single_channel_physionet_49829_TV.wav"

# --- Display / playback settings -------------------------------------------
SECTION = (12.0, 25.0)   # seconds; None to show/play the whole file
CLIP_DISPLAY = True
CLIP_PERCENTILE = 99.9
PLAY_ORIGINAL = True     # True: play unfiltered original; False: filtered+downsampled

# --- Filter settings --------------------------------------------------------
ORDER = 2
F_LOW = 10.0              # lower cutoff [Hz]
F_HIGH = 800.0            # upper cutoff [Hz] (deliberately NOT adapted)

# --- Downsampling settings --------------------------------------------------
DECIMATION = 4            # M: keep every M-th sample (4 kHz -> 1 kHz)


def load_audio(path: Path):
    """Load a wav file, return (data as float array, sample rate)."""
    fs, data = wavfile.read(path)

    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype != np.float32 and data.dtype != np.float64:
        raise ValueError(f"Unsupported wav dtype: {data.dtype}")

    if data.ndim > 1:
        data = data[:, 0]

    print(f"Loaded {path.name}: {len(data)} samples @ {fs} Hz "
          f"({len(data) / fs:.1f} s)")
    return data, fs


def butterworth_bandpass(data, fs, order=ORDER, f_low=F_LOW, f_high=F_HIGH):
    """Zero-phase Butterworth bandpass. Cutoffs normalized by Nyquist."""
    nyquist = fs / 2.0
    low, high = f_low / nyquist, f_high / nyquist

    if high >= 1.0:
        # This is what happens if you design the filter directly at
        # the downsampled rate without adapting the cutoff:
        raise ValueError(
            f"Cannot design a {f_high} Hz cutoff at Fs = {fs} Hz: "
            f"cutoff must be below the Nyquist frequency ({nyquist} Hz). "
            f"Normalized cutoff = {high:.2f} >= 1.")

    b, a = signal.butter(order, [low, high], btype='band')
    print(f"Butterworth bandpass @ Fs={fs:.0f}: order={order}, "
          f"cutoffs [{f_low}, {f_high}] Hz -> normalized [{low}, {high}]")
    return signal.filtfilt(b, a, data)


def demonstrate_design_error(fs_new):
    """Show what happens if you design the filter at the new rate."""
    print(f"\n--- Attempting to design a 10-800 Hz filter directly at "
          f"Fs = {fs_new} Hz ---")
    try:
        butterworth_bandpass(np.zeros(10), fs_new)
    except ValueError as e:
        print(f"ERROR: {e}")
    print()


def downsample(x, m):
    """Naive decimation: keep every m-th sample (no anti-alias filter)."""
    print(f"Decimating by M={m}: {len(x)} -> {len(x[::m])} samples")
    return x[::m]


def plot_and_play(data, fs_orig, filtered_ds, fs_new):
    """Show original vs downsampled signal; play with a moving cursor."""
    n = len(data)
    t_orig = np.arange(n) / fs_orig
    t_new = np.arange(len(filtered_ds)) / fs_new

    if SECTION is not None:
        i0, i1 = int(SECTION[0] * fs_orig), min(int(SECTION[1] * fs_orig), n)
        j0, j1 = int(SECTION[0] * fs_new), min(int(SECTION[1] * fs_new),
                                               len(filtered_ds))
    else:
        i0, i1 = 0, n
        j0, j1 = 0, len(filtered_ds)

    x_plot = data[i0:i1]
    t1, t2 = t_orig[i0:i1], t_new[j0:j1]
    y_plot = filtered_ds[j0:j1]

    if CLIP_DISPLAY:
        a_clip = np.percentile(np.abs(x_plot), CLIP_PERCENTILE)
        y_margin = 1.2 * a_clip
    else:
        y_margin = 1.05 * np.max(np.abs(x_plot))
    print(f"Display amplitude limit: +/-{y_margin:.3f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t1, x_plot, lw=0.5, color="tab:blue",
            label=f"original @ {fs_orig:.0f} Hz")
    ax.plot(t2, y_plot, lw=0.7, color="tab:red",
            label=f"filtered + decimated x{DECIMATION} @ {fs_new:.0f} Hz")
    line, = ax.plot([], [], color="black", lw=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [-]")
    ax.set_title("Downsampling experiment: Fs 4 kHz -> 1 kHz, "
                 "cutoff kept at 800 Hz (aliased above 500 Hz)")
    ax.set_xlim(SECTION[0] if SECTION else 0,
                SECTION[1] if SECTION else t_orig[-1])
    ax.set_ylim(-y_margin, y_margin)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Play: original at fs_orig, or the downsampled signal at fs_new
    if PLAY_ORIGINAL:
        audio, fs_play, t_start = data[i0:i1].copy(), fs_orig,             SECTION[0] if SECTION else 0.0
        print("Playing ORIGINAL...")
    else:
        audio, fs_play, t_start = filtered_ds[j0:j1].copy(), fs_new,             SECTION[0] if SECTION else 0.0
        print(f"Playing filtered+downsampled @ {fs_new:.0f} Hz...")
    duration = len(audio) / fs_play

    sd.play(audio, fs_play)  # non-blocking
    t0 = time.perf_counter()

    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= duration or not sd.get_stream().active:
                break
            current = t_start + elapsed
            line.set_data([current, current], [-y_margin, y_margin])
            plt.pause(0.05)
        sd.wait()
    except Exception:
        sd.stop()
        raise
    finally:
        print("Done.")
    plt.show()


def main():
    if not AUDIO_FILE.exists():
        sys.exit(f"Audio file not found: {AUDIO_FILE}\n"
                 f"Place it in the same directory as filter.py.")
    data, fs = load_audio(AUDIO_FILE)

    # Lesson 1: designing the 800 Hz filter at the new rate fails
    fs_new = fs / DECIMATION
    # demonstrate_design_error(fs_new)

    # Downsample
    data_ds = downsample(data, DECIMATION)

    # Filter with new cutoff frequency
    filtered_ds = butterworth_bandpass(data_ds, fs_new,
                                       order= 2,
                                       f_low=F_LOW, f_high= 499)
 
    # Lesson 2: content between 500 and 800 Hz is aliased by the
    # decimation (new Nyquist = 500 Hz).

    plot_and_play(data, fs, filtered_ds, fs_new)


if __name__ == "__main__":
    main()
