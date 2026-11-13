# Lowpass filtering — Assignment 4.2 workspace

This directory contains `filter.py`, a self-contained script that loads a real
heart sound recording (phonocardiogram, PCG), applies a Butterworth bandpass
filter, and helps you see, hear and understand what the filter does.

## The data

`heart_single_channel_physionet_49829_TV.wav` (not committed; see the
repository `.gitignore`) is a real PCG from the PhysioNet CirCor DigiScope
dataset (patient 49829), recorded with a digital stethoscope at the tricuspid
valve (TV) auscultation location. Mono, 16-bit, Fs = 4 kHz, 29.3 s.
See `audio_description.md` for details and things to be aware of.

## What the script does

- **Load**: reads the wav file, converts to float amplitude in [-1, 1],
  reports sample rate and duration.
- **Filter**: designs an *order-2 Butterworth bandpass* (10–800 Hz) using
  `signal.butter`. Cutoffs are normalized by the Nyquist frequency
  (Fs/2 = 2 kHz): 10/2000 = 0.005 and 800/2000 = 0.4.
- **Zero-phase filtering**: applies the filter with `signal.filtfilt`
  (forward + backward), so the phase response cancels and S1/S2 peaks
  keep their exact positions.
- **Display**: plots original (blue) and filtered (red) amplitude vs time,
  focused on the clean 12–25 s section, with artifact spikes clipped via a
  robust percentile scale (see `audio_description.md`).
- **Play**: plays the audio while a moving cursor tracks playback on the
  plot. Play the original or the filtered signal (A/B listening).

## Usage

```bash
conda activate IP3
python filter.py           # plot + play: original vs filtered signal
python filter.py --bode    # generate bode.pdf (filter frequency response)
```

Settings live as named constants at the top of the script:

| Constant | Meaning |
|---|---|
| `ORDER` | Butterworth filter order (assignment: 2) |
| `F_LOW`, `F_HIGH` | Cutoff frequencies 10 and 800 Hz |
| `SECTION` | Time window shown/played, e.g. (12, 25) s |
| `CLIP_PERCENTILE` | Robust display scale that ignores artifact spikes |
| `PLAY_ORIGINAL` | Play original (True) or filtered (False) audio |

Outputs: an interactive plot window, and with `--bode` the file `bode.pdf`
(amplitude top, phase bottom, 0–4 kHz, vector PDF).

## What you can learn from it

1. **Nyquist normalization.** `signal.butter` expects cutoffs divided by
   Fs/2. With Fs = 4 kHz: [10, 800] Hz -> [0.005, 0.4].
2. **Zero-phase filtering.** `filtfilt` runs the filter forwards and
   backwards, so the phase cancels exactly and peaks stay in place - compare
   the red and blue curves. The effective filter order doubles (2 -> 4),
   as does the damping. Peak positions matter because Modules 3-4
   localize S1/S2 in time.
3. **Reading a Bode plot.** `--bode` shows the maximally flat passband,
   the transition band, and the phase non-linearity of the one-pass filter
   near the top of the passband - the reason pulse shapes distort without
   `filtfilt`.
4. **Zeros at DC and Nyquist.** A digital *bandpass* Butterworth has zeros
   exactly at z = 1 (0 Hz) and z = -1 (Nyquist, 2 kHz), so its damping at
   those frequencies is (numerically about -240 dB) infinite. A *lowpass*
   Butterworth does not share this at DC.
5. **Robust scaling.** The recording contains impulsive artifacts several
   times larger than the heart sounds. Naive normalization on max|x| lets
   the artifacts set the scale; a percentile-based scale (or clipping /
   median filtering) keeps S1/S2 readable. Same lesson applies to the
   bandpass: filtering also suppresses out-of-band artifacts.
6. **Digital filter periodicity.** In the Bode plot the response above
   2 kHz mirrors the response below it: a digital filter's response is
   periodic in Fs. Nothing above Nyquist exists in the recording.

## Next steps

- Vary `ORDER` (2 vs 4) and observe passband flatness vs roll-off steepness.
- Add the filter's impulse response (g = filtfilt(b, a, impulse)).
- Continue with downsampling (4.3), spectrogram (4.4) and energy
  detection (4.5) on the filtered signal.
