"""
IP3 - M3 (assignment 6.3.3): data covariance matrix Rx.

Implements the recipe from the manual: the covariance matrix of the
received signals for Q unit-power, mutually uncorrelated sources at
angles theta0, plus independent identically distributed (iid) sensor
noise at a given SNR:

    Rx = R + Rn
    R  = A A^H                 (signal covariance, Q rank-1 terms)
    Rn = sigma_n^2 * I         (noise covariance: uncorrelated, equal
                                variance on every channel)
    sigma_n = 10^(-SNR/20)     (SNR in dB, amplitude conversion)

Requires: numpy, and beamfinding.a_lin (a_lin.py in this directory).
"""

import numpy as np

from a_lin import a_lin   # local import: run from the beamfinding directory


def generate_Rx(theta0, M, d, v, f0, SNR):
    """Data covariance Rx for Q unit-power uncorrelated sources in noise.

    Parameters
    ----------
    theta0 : float or 1D array_like
        True source angle(s) [radians] (Q = number of sources).
    M : int
        Number of microphones.
    d : float
        Microphone spacing [m].
    v : float
        Speed of sound [m/s].
    f0 : float
        Center frequency [Hz].
    SNR : float
        Signal-to-noise ratio per source [dB].

    Returns
    -------
    Rx : complex ndarray, shape (M, M)
        Hermitian data covariance matrix Rx = A A^H + sigma_n^2 I.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    Q = len(theta0)

    # Noise amplitude from SNR (dB): sigma_n = 10^(-SNR/20)
    sigma_n = 10 ** (-SNR / 20)

    # Source direction vectors, A has shape (M, Q)
    A = a_lin(theta0, M, d, v, f0)
    A_H = A.conj().T

    # Signal covariance: sum of Q rank-1 matrices a a^H (unit source power)
    R = A @ A_H                     # (M, M)

    # Noise covariance: iid noise -> uncorrelated (off-diagonal zeros),
    # equal variance (uniform diagonal sigma_n^2)
    Rn = np.eye(M, M) * sigma_n ** 2

    Rx = R + Rn
    print(f"generate_Rx: M = {M} mics, Q = {Q} source(s) at "
          f"{np.degrees(theta0)} deg, SNR = {SNR:.0f} dB "
          f"(sigma_n = {sigma_n:.4f})")
    return Rx


def main():
    # --- Test: two sources, as in the manual's recipe ----------------------
    M, d, v, f0 = 6, 0.10, 340.0, 500.0
    theta0 = np.deg2rad([-20.0, 30.0])       # two sources
    SNR = 20.0                                # dB

    Rx = generate_Rx(theta0, M, d, v, f0, SNR)

    # --- Sanity checks that make the structure visible ---------------------
    print("shape:", Rx.shape)
    print("Hermitian (Rx == Rx^H):", np.allclose(Rx, Rx.conj().T))

    # Eigenvalue spectrum: Q "signal" eigenvalues stand out above a
    # noise floor at sigma_n^2. This structure is what beamformers exploit.
    eigvals = np.sort(np.linalg.eigvalsh(Rx))[::-1]   # descending, real
    print("eigenvalues (descending):")
    for lam in eigvals:
        print(f"  {lam:8.4f}")
    print(f"noise floor sigma_n^2 = {10 ** (-SNR / 10):.4f}")
    print(f"number of eigenvalues clearly above the floor: "
          f"{np.sum(eigvals > 3 * 10 ** (-SNR / 10))} (expected: Q = 2)")

    # Rank of the signal part: Q (full-rank Rx because noise adds sigma_n^2 I)
    R_only = Rx - (10 ** (-SNR / 10)) * np.eye(M)
    print("rank of signal covariance R = A A^H:",
          np.linalg.matrix_rank(R_only, tol=1e-9), "(expected: Q = 2)")


if __name__ == "__main__":
    main()
