# IP3: Heart Sound Localization

TU Delft — EE2L1 IP3 project. Analysis and localization of heart sounds
(phonocardiograms) from a multi-microphone digital stethoscope, using
filtering, signal modeling, and array processing.

## Contents

| Directory | About |
|---|---|
| `lowpass_filtering/` | Assignment 4.2 workspace: load, play, filter (Butterworth bandpass 10–800 Hz, zero-phase `filtfilt`) and visualize a real PhysioNet PCG recording; includes a Bode-plot mode. See its [README](lowpass_filtering/README.md) for usage and learning points. |

More module directories (M1 single-channel analysis, M2 heart sound
modeling, M3/M4 direction finding) will be added as the project progresses.

## Conventions

- Standard processing sample rate: **Fs = 4 kHz** (Nyquist = 2 kHz)
- Python 3 + NumPy/SciPy/Matplotlib; recommended conda environment `IP3`:
  `pip install numpy scipy matplotlib sounddevice jupyter`
- Audio recordings and large data files are **never committed** (see `.gitignore`);
  place data files in the directory where you use them
- Each module folder has its own README describing what the code does,
  how to run it, and what can be learned from it

## Getting started

```bash
git clone https://github.com/LeonAbelmann/IP3.git
cd IP3
```

Start in `lowpass_filtering/`: put the recording
`heart_single_channel_physionet_49829_TV.wav` there, then

```bash
python filter.py           # plot + play: original vs filtered signal
python filter.py --bode    # filter frequency response -> bode.pdf
```
