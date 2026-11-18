"""
IP3 - M3: Spatial spectrum of the matched (delay-and-sum) beamformer.

Py(theta) = |a^H(theta) a(theta0)|^2     (manual section 6.2.5)

for a Uniform Linear Array of M microphones, evaluated over
theta = -90 .. +90 degrees. The source direction theta0 and the
microphone spacing d can be changed (command line or constants below).

Delta = d/lambda (the spacing in wavelengths) is shown inside the plot:
Delta > 0.5 leads to spatial aliasing (grating lobes).

Usage:
    python spatial_spectrum.py                     # theta0=0, d=10 cm
    python spatial_spectrum.py --theta0 30         # degrees
    python spatial_spectrum.py --d 0.85            # meters
    python spatial_spectrum.py --M 5 --f0 800

Requires: numpy, matplotlib
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

# --- Default settings ------------------------------------------------------
M = 6            # number of microphones (IP3 stethoscope array)
v = 340.0        # speed of sound [m/s]
F0 = 500.0       # center frequency [Hz]
THETA0 = 0.0     # source angle [degrees] (command line overrides)
D = 0.10         # microphone spacing [m]   (command line overrides)

# --- Beamforming grid ------------------------------------------------------
THETA_GRID_DEG = np.arange(-90.0, 90.0 + 0.25, 0.25)   # evaluation angles


def a_lin(theta, M, d, v, f0):
    """ULA array response vectors, shape (M, len(theta)); mic 0 is reference."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    m = np.arange(M)[:, None]
    phase = -1j * m * (d / v) * np.sin(theta[None, :]) * 2 * np.pi * f0
    return np.exp(phase)


def spatial_spectrum(theta_deg, theta0_deg, M, d, v, f0):
    """Matched-beamformer spatial spectrum Py(theta) for one source at theta0.

    Py(theta) = |a^H(theta) a(theta0)|^2, normalized to unit source power.
    The returned spectrum is normalized to peak 1 for interpretability
    (the normalization does not move the peaks).
    """
    a = a_lin(np.deg2rad(theta_deg), M, d, v, f0)
    a0 = a_lin(np.deg2rad(theta0_deg), M, d, v, f0)
    Py = np.abs(a.conj().T @ a0) ** 2          # (Q,1)
    return Py[:, 0] / Py.max()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--theta0", type=float, default=THETA0,
                   help="source angle of incidence [deg] (default 0)")
    p.add_argument("--d", type=float, default=D,
                   help="microphone spacing [m] (default 0.10)")
    p.add_argument("--M", type=int, default=M,
                   help="number of microphones (default 6)")
    p.add_argument("--f0", type=float, default=F0,
                   help="center frequency [Hz] (default 500)")
    p.add_argument("--v", type=float, default=v,
                   help="speed of sound [m/s] (default 340)")
    args = p.parse_args()

    lam = args.v / args.f0
    Delta = args.d / lam
    aliasing = Delta > 0.5

    Py = spatial_spectrum(THETA_GRID_DEG, args.theta0, args.M, args.d,
                          args.v, args.f0)

    peak_deg = THETA_GRID_DEG[np.argmax(Py)]
    print(f"Delta = d/lambda = {args.d:.3f} / {lam:.3f} = {Delta:.4f}"
          f"{'  (>0.5: expect grating lobes!)' if aliasing else '  (<0.5: no spatial aliasing)'}")
    print(f"Spectrum peak at {peak_deg:+.2f} deg "
          f"(source at {args.theta0:+.1f} deg)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(THETA_GRID_DEG, Py, color="tab:blue")
    ax.set_xlabel(" incidence angle theta [deg]   (0 = broadside)")
    ax.set_ylabel("Normalized output power Py(theta) [-]")
    ax.set_title(f"Matched-beamformer spatial spectrum, M={args.M}, "
                 f"f0={args.f0:.0f} Hz, v={args.v:.0f} m/s")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Mark the true source direction
    ax.axvline(args.theta0, color="tab:red", ls=":", lw=1,
              label=f"source: theta0 = {args.theta0:+.0f} deg")
    # Show Delta inside the plot
    ax.text(0.02, 0.97,
            f"$\\Delta = d/\\lambda = {Delta:.3f}$          \n"
            f"($\\Delta {'>' if aliasing else '\\leq'} 0.5$: "
            f"{'spatial aliasing' if aliasing else 'no spatial aliasing'}\n"
            f"\"grating lobes\" {'present' if aliasing else 'absent'})",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="0.5"))

    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
