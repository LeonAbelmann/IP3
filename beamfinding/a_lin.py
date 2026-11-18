"""
IP3 - M3 (Basic direction finding): array response vector a_lin(theta).

Implements eq. (6.1) of the course manual: the narrowband array response
vector of a Uniform Linear Array (ULA) of M microphones with spacing d,
for a plane wave at angle theta (measured from broadside), center
frequency f0, propagation speed v:

    a_m(theta) = exp(-j * m * (d/v) * sin(theta) * Omega_0),   m = 0..M-1

    Omega_0 = 2*pi*f0

Microphone 0 is the phase reference: a[0] = 1 for all theta.

Requires: numpy
"""

import numpy as np


def a_lin(theta, M, d, v, f0):
    """Array response vectors of a uniform linear array (manual eq. 6.1).

    Parameters
    ----------
    theta : float or 1D array_like
        Incidence angle(s) [radians], measured from broadside.
    M : int
        Number of (uniformly spaced) microphones.
    d : float
        Microphone spacing [m].
    v : float
        Speed of sound [m/s].
    f0 : float
        Center frequency of the wave [Hz].

    Returns
    -------
    a : complex ndarray, shape (M, Q)
        One column per angle: a[:, q] = a(theta_q). For scalar theta,
        the shape is (M, 1). Unit magnitude for every entry.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    m = np.arange(M)[:, None]                     # (M, 1) microphone index
    Omega0 = 2 * np.pi * f0
    phase = -1j * m * (d / v) * np.sin(theta[None, :]) * Omega0
    return np.exp(phase)


def main():
    # --- Test from assignment 6.2.3 / 6.3.2 ---------------------------------
    M, f0, v, d = 4, 1000.0, 300.0, 0.10
    thetas_deg = np.arange(0, 50, 5)             # 0 .. 45 deg, step 5
    thetas = np.deg2rad(thetas_deg)

    # Informational printout: spacing in wavelengths (spatial aliasing)
    lam = v / f0
    Delta = d / lam
    print(f"Speed of sound v = {v:.0f} m/s, f0 = {f0:.0f} Hz")
    print(f"Wavelength lambda = v/f0 = {lam:.3f} m")
    print(f"Delta = d/lambda = {Delta:.4f}"
          f"  ({'< 0.5: no spatial aliasing' if Delta <= 0.5 else '>= 0.5: spatial aliasing (grating lobes)!'})")

    a = a_lin(thetas, M, d, v, f0)               # (M, Q)

    assert np.allclose(np.abs(a), 1.0), "entries must have unit magnitude"
    assert np.allclose(a[0, :], 1.0), "mic 0 is the phase reference"

    # --- Table: rows = components, columns = theta -------------------------
    print(f"\na(theta), M = {M} mics, d = {d:.2f} m, f0 = {f0:.0f} Hz, "
          f"v = {v:.0f} m/s")
    print(f"{'component':>10} | " + " | ".join(f"th={t:>2.0f}deg" for t in thetas_deg))
    print("-" * (13 + 11 * len(thetas_deg)))
    for m in range(M):
        cells = []
        for q in range(len(thetas_deg)):
            z = a[m, q]
            cells.append(f"{z.real:+.3f}{z.imag:+.3f}j")
        print(f"{'a[%d]' % m:>10} | " + " | ".join(f"{c:>9}" for c in cells))

    # Phase (in degrees) makes the structure easier to read:
    print(f"\nPhase of a in degrees (angle -m * 2*pi*Delta*sin(theta)):")
    print(f"{'component':>10} | " + " | ".join(f"th={t:>2.0f}deg" for t in thetas_deg))
    print("-" * (13 + 11 * len(thetas_deg)))
    for m in range(M):
        phases = np.degrees(np.angle(a[m, :]))
        print(f"{'a[%d]' % m:>10} | " + " | ".join(f"{p:>9.1f}" for p in phases))


if __name__ == "__main__":
    main()
