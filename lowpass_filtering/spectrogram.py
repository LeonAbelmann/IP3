"""
IP3 - M1 (section 4.4): Spectrogram analysis.

Extends filter.py: loads the heart sound recording, applies the zero-phase
Butterworth bandpass (10-800 Hz), and shows the spectrograms of the
original and the filtered signal, aligned above each other.

Uses scipy.signal.ShortTimeFFT (requires SciPy >= 1.12), following the
example in the course manual section 4.4: Gaussian window with 0.01 s
standard deviation, nperseg=256, noverlap=125 (hop = 131 samples),
centered FFT, PSD scaling.

Amplitude is shown on a logarithmic scale (dB), frequency range 0..fs/2.

Requires: numpy, scipy (>= 1.12), matplotlib
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

# Audio file next to this script
AUDIO_FILE = Path(__file__).parent / "heart_single_channel_physionet_49829_TV.wav"

# --- Display settings -------------------------------------------------------
# Time window to analyze (seconds). None = whole file.
# (12, 25) is the clean section; the impulsive artifacts elsewhere
# dominate the color scale of a spectrogram even more than in the
# time-domain plot.
SECTION = (12.0, 25.0)

# --- Filter settings (as in filter.py) --------------------------------------
ORDER = 2
F_LOW = 10.0
F_HIGH = 800.0

# --- Spectrogram settings (manual section 4.4) ------------------------------
NPERSEG = 256        # window length in samples
NOVERLAP = 125       # window overlap -> hop = 256 - 125 = 131 samples
FFT_MODE = 'centered'
SCALE_TO = 'psd'     # power spectral density scaling


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
    """Zero-phase Butterworth bandpass (10-800 Hz)."""
    nyquist = fs / 2.0
    b, a = signal.butter(order, [f_low / nyquist, f_high / nyquist],
                         btype='band')
    return signal.filtfilt(b, a, data)


def make_spectrogram(x, fs):
    """Spectrogram via ShortTimeFFT (manual section 4.4 recipe).

    Returns (Sxx_dB in dB, t_lo, t_hi) with the frequency axis restricted
    to 0..fs/2 (the 'centered' FFT produces -fs/2..fs/2; for a real
    signal the halves are symmetric, so we keep f >= 0).
    """
    N = len(x)
    win = ('gaussian', 1e-2 * fs)   # Gaussian window, 0.01 s std dev

    SFT = signal.ShortTimeFFT.from_window(
        win, fs, nperseg=NPERSEG, noverlap=NOVERLAP,
        fft_mode=FFT_MODE, scale_to=SCALE_TO, phase_shift=None)

    Sxx = SFT.spectrogram(x)        # shape (mfft, k): PSD per time slot
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of the plot

    # Keep only f >= 0 half of the centered spectrum
    f = SFT.f
    keep = f >= 0
    Sxx = Sxx[keep]

    # Logarithmic amplitude scale (dB), floor to avoid log(0)
    Sxx_dB = 10 * np.log10(np.fmax(Sxx, 1e-10))
    return Sxx_dB, t_lo, t_hi


def plot_spectrograms(x_orig, x_filt, fs):
    """Two spectrograms aligned above each other, same axes and color scale."""
    S1, t_lo, t_hi = make_spectrogram(x_orig, fs)
    S2, _, _ = make_spectrogram(x_filt, fs)

    # Shared color limits: robust against the impulsive artifacts
    # (a few very loud pixels would otherwise wash out everything else)
    vmax = max(np.percentile(S1, 99.9), np.percentile(S2, 99.9))
    vmin = vmax - 80.0   # 80 dB dynamic range
    print(f"Color scale: {vmin:.1f} to {vmax:.1f} dB")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   sharey=True)

    im1 = ax1.imshow(S1, origin='lower', aspect='auto',
                     extent=(t_lo, t_hi, 0, fs / 2),
                     vmin=vmin, vmax=vmax, cmap='magma')
    ax1.set_title("Spectrogram: original signal")
    ax1.set_ylabel("Frequency [Hz]")

    ax2.imshow(S2, origin='lower', aspect='auto',
               extent=(t_lo, t_hi, 0, fs / 2),
               vmin=vmin, vmax=vmax, cmap='magma')
    ax2.set_title("Spectrogram: bandpass filtered "
                  f"({F_LOW:.0f}-{F_HIGH:.0f} Hz, zero-phase)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")

    if SECTION is not None:
        ax1.set_xlim(*SECTION)
        ax1.set_ylim(0, fs / 2)

    fig.subplots_adjust(right=0.9)
    cbar = fig.colorbar(im1, ax=[ax1, ax2])
    cbar.set_label("PSD [dB]")

    fig.suptitle("heart_single_channel_physionet_49829_TV.wav", y=0.995)
    fig.tight_layout()
    plt.show()


def main():
    if not AUDIO_FILE.exists():
        sys.exit(f"Audio file not found: {AUDIO_FILE}\n"
                 f"Place it in the same directory as spectrogram.py.")
    data, fs = load_audio(AUDIO_FILE)

    if SECTION is not None:
        i0, i1 = int(SECTION[0] * fs), min(int(SECTION[1] * fs), len(data))
        data = data[i0:i1]
        print(f"Analyzing section {SECTION[0]}-{SECTION[1]} s "
              f"({len(data)} samples)")

    filtered = butterworth_bandpass(data, fs)
    plot_spectrograms(data, filtered, fs)


if __name__ == "__main__":
    main()
