HEART SOUND RECORDING — Module 1 (single channel signal analysis)
===================================================
PURPOSE
A real heart sound recording (phonocardiogram, PCG) to get familiar with heart sounds: listen to it, plot it, make spectrograms, and test your preprocessing (bandpass filter) on it. See manual Ch. 4.

1. SOURCE
- Recording from Public dataset from PhysioNet (CirCor DigiScope, Patient ID: 49829) [Oliveira et al. (2022)]
- Recorded with a digital stethoscope at ONE location on the chest, namely the "TV" = tricuspid valve area (manual §2.3, Fig. 2.6). NB: this is an auscultation location, not an isolated valve signal. All four valves are audible in this recording; the location only determines which valve dominates.

2. RECORDED SIGNAL
- File:    heart_single_channel_physionet_49829_TV.wav
- Format:  WAV, Fs = 4 kHz, 16-bit integer, mono, 29.3 s.

3. THINGS TO BE AWARE OF
- The recording contains a few short impulsive artifacts that are several times larger than the heart sounds themselves. If you normalise with x / max|x|, these spikes will set the scale and the actual S1/S2 sounds will look very small. Consider clipping or median filtering first, or normalise on a high percentile.
- At Fs = 4 kHz the highest representable frequency is 2 kHz. Filter specifications at or above 2 kHz cannot be checked on this file.
