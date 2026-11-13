"""
IP3 - M1: Load, play, filter and visualize a heart sound recording.

Loads 'heart_single_channel_physionet_49829_TV.wav', applies a 2nd-order
Butterworth bandpass filter (10-800 Hz, zero-phase via filtfilt), shows
original and filtered amplitude as a function of time, and plays the
sound over the default audio output while the plot is open.

Bode plot mode (assignment 4.2):
    python filter.py --bode
generates bode.pdf with the filter's amplitude (top) and phase (bottom),
from 0 to 4 kHz.

Requires: numpy, scipy, matplotlib, sounddevice
"""

import argparse
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
# Clean section of the recording (seconds)
SECTION = (12.0, 25.0)   # None to show/play the whole file

# Amplitude clip for *display only* (data itself is not modified).
CLIP_DISPLAY = True
CLIP_PERCENTILE = 99.9   # robust amplitude estimate, ignores artifact spikes

# Play the original (unfiltered) audio, not the filtered one
PLAY_ORIGINAL = True

# --- Filter settings (assignment 4.2) ---------------------------------------
ORDER = 2                 # Butterworth filter order
F_LOW = 10.0              # lower cutoff frequency [Hz]
F_HIGH = 800.0            # upper cutoff frequency [Hz]


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


def butterworth_bandpass_coefficients(fs, order=ORDER, f_low=F_LOW, f_high=F_HIGH):
    """Design a Butterworth bandpass filter h(t) (manual section 4.2).

    Cutoffs are normalized by the Nyquist frequency (fs/2).
    """
    nyquist = fs / 2.0
    low = f_low / nyquist     # e.g. 10 / 2000  = 0.005
    high = f_high / nyquist   # e.g. 800 / 2000 = 0.4
    b, a = signal.butter(order, [low, high], btype='band')
    print(f"Butterworth bandpass: order={order}, "
          f"cutoffs [{f_low}, {f_high}] Hz -> normalized [{low}, {high}]")
    return b, a


def butterworth_bandpass(data, fs, order=ORDER, f_low=F_LOW, f_high=F_HIGH):
    """Zero-phase Butterworth bandpass filtering of a signal.

    filtfilt applies the filter forwards and backwards -> zero phase,
    and doubles the effective filter order.
    """
    b, a = butterworth_bandpass_coefficients(fs, order, f_low, f_high)
    return signal.filtfilt(b, a, data)


def make_bode_plot(fs, order=ORDER, f_low=F_LOW, f_high=F_HIGH,
                   f_max=4000.0, n_freqs=4001, pdf_path="bode.pdf"):
    """Bode plot (amplitude top, phase bottom) of the filter h(t).

    Also shows the forward-backward filter g(t) = |H|^2 (filtfilt),
    whose phase is exactly zero. Frequencies run from 0 to f_max.

    Exported to pdf_path (PDF, vector graphics).
    """
    b, a = butterworth_bandpass_coefficients(fs, order, f_low, f_high)

    # Frequency axis 0..f_max Hz. Note: above the Nyquist frequency
    # (fs/2) the digital filter response mirrors (aliasing).
    f = np.linspace(0, f_max, n_freqs)

    # Response of the one-pass filter h(t)
    _, h = signal.freqz(b, a, worN=f, fs=fs, include_nyquist=True)
    mag_h = np.abs(h)
    phase_h = np.unwrap(np.angle(h))

    # Response of the forward-backward filter g(t) = |H|^2
    # (amplitude squared, phase exactly zero)
    mag_g = mag_h ** 2

    mag_db = 20 * np.log10(np.maximum(mag_h, 1e-12))
    mag_g_db = 20 * np.log10(np.maximum(mag_g, 1e-12))
    phase_deg = np.degrees(phase_h)

    nyquist = fs / 2.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # --- Top: amplitude ---
    ax1.plot(f, mag_db, color="tab:blue", label=f"h(t): order {order}")
    ax1.plot(f, mag_g_db, color="tab:red", ls="--",
             label=f"g(t) = filtfilt: order {2 * order}")
    ax1.axvline(f_low, color="grey", ls=":", lw=1)
    ax1.axvline(f_high, color="grey", ls=":", lw=1)
    ax1.axvline(nyquist, color="black", ls="-.", lw=1,
                label=f"Nyquist = {nyquist:.0f} Hz")
    ax1.axhline(-3, color="grey", lw=0.5)
    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_title(f"Butterworth bandpass {f_low:.0f}-{f_high:.0f} Hz, "
                  f"order {order} (filtfilt: {2 * order})")
    ax1.set_ylim(-80, 5)
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    # --- Bottom: phase ---
    ax2.plot(f, phase_deg, color="tab:blue", label="h(t) phase")
    ax2.axvline(nyquist, color="black", ls="-.", lw=1)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Phase [degrees]")
    ax2.set_xlim(0, f_max)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(pdf_path)  # vector PDF export
    print(f"Bode plot exported to {pdf_path}")

    # Readout at frequencies of interest
    for f_check in (f_low, f_high, nyquist):
        idx = np.argmin(np.abs(f - f_check))
        print(f"  h(t) @ {f[idx]:6.0f} Hz: {mag_db[idx]:6.1f} dB, "
              f"phase {phase_deg[idx]:7.1f} deg   "
              f"| g(t): {mag_g_db[idx]:6.1f} dB")


def plot_and_play(data, filtered, fs):
    """Show original + filtered waveform; play audio with a moving cursor."""
    n = len(data)
    t = np.arange(n) / fs  # time axis in seconds

    # Select the section to display
    if SECTION is not None:
        i0, i1 = int(SECTION[0] * fs), min(int(SECTION[1] * fs), n)
        t_plot, x_plot, f_plot = t[i0:i1], data[i0:i1], filtered[i0:i1]
    else:
        t_plot, x_plot, f_plot = t, data, filtered

    # Robust display limits: percentile ignores the artifact spikes
    if CLIP_DISPLAY:
        a_clip = np.percentile(np.abs(x_plot), CLIP_PERCENTILE)
        y_margin = 1.2 * a_clip
    else:
        y_margin = 1.05 * np.max(np.abs(x_plot))
    print(f"Display amplitude limit: +/-{y_margin:.3f} "
          f"({'clipped' if CLIP_DISPLAY else 'full scale'})")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_plot, x_plot, lw=0.5, color="tab:blue", label="original")
    ax.plot(t_plot, f_plot, lw=0.7, color="tab:red", label="filtered")
    line, = ax.plot([], [], color="black", lw=1.5)  # playback cursor
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [-]")
    ax.set_title("Heart sound: original vs Butterworth bandpass 10-800 Hz")
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_ylim(-y_margin, y_margin)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Choose what to play
    audio = data if PLAY_ORIGINAL else filtered
    if SECTION is not None:
        audio_seg, t_start = audio[i0:i1].copy(), SECTION[0]
        print(f"Playing section {SECTION[0]}-{SECTION[1]} s "
              f"({'original' if PLAY_ORIGINAL else 'filtered'})...")
    else:
        audio_seg, t_start = audio.copy(), 0.0
        print("Playing whole file...")
    duration = len(audio_seg) / fs

    sd.play(audio_seg, fs)  # non-blocking
    t0 = time.perf_counter()  # wall-clock reference for the cursor

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
    plt.show()  # keep the plot open after playback finishes


def main():
    parser = argparse.ArgumentParser(
        description="Load, play, filter and visualize a heart sound.")
    parser.add_argument("--bode", action="store_true",
                        help="Generate bode.pdf (amplitude + phase, 0-4 kHz) "
                             "and exit.")
    parser.add_argument("--fs", type=float, default=4000.0,
                        help="Sample rate used for the --bode plot [Hz].")
    args = parser.parse_args()

    if args.bode:
        make_bode_plot(fs=args.fs)
        return

    if not AUDIO_FILE.exists():
        sys.exit(f"Audio file not found: {AUDIO_FILE}\n"
                 f"Place it in the same directory as filter.py.")
    data, fs = load_audio(AUDIO_FILE)
    filtered = butterworth_bandpass(data, fs)
    plot_and_play(data, filtered, fs)


if __name__ == "__main__":
    main()
