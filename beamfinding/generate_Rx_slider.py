"""
IP3 - M3: Interactive Rx explorer with a slider.

Shows the data covariance matrix Rx of two sources, with the direction
of the SECOND source (theta2) adjustable via a matplotlib slider.
Watch what happens as you drag theta2 from far away towards theta1:

- the eigenvalue spectrum splits (M +/- |rho|) and finally collapses
  (lambda_2 -> noise floor) when both sources coincide,
- the |Rx| image loses structure (rank-1: a a^H is rank one),
- the phase image goes from "two ramps superposed" to one clean ramp.

Layout: 2x2 of images (|Rx|, phase) + (spatial spectrum of the
eigenvalues, i.e. the eigenbasis strength), plus the slider below.
The second source position marker in the spatial spectrum.

Requires: numpy, matplotlib, and beamfinding.a_lin (a_lin.py in this
directory).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from a_lin import a_lin   # local import: run from the beamfinding directory
from generate_Rx import generate_Rx

# --- Fixed settings ---------------------------------------------------------
M = 6
d = 0.10
v = 340.0
f0 = 500.0
SNR = 20.0             # dB
THETA1_DEG = -20.0     # first source (fixed)
THETA2_INIT_DEG = 30.0  # second source (slider)


def plot_spectrum(ax, eigvals, sigma_n_sq):
    """Eigenvalue spectrum with noise floor, as stem/bar plot."""
    ax.clear()
    m = np.arange(1, M + 1)
    ax.bar(m - 0.2, eigvals, width=0.38, color="tab:blue",
           label="eigenvalues")
    ax.axhline(sigma_n_sq, color="tab:red", ls=":",
               label=f"noise floor sigma_n^2 = {sigma_n_sq:.3f}")
    ax.set_xlabel("eigenvalue index")
    ax.set_xticks(m)
    ax.set_ylabel("lambda")
    ax.set_ylim(0, 2 * M * 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def update(theta2_deg):
    """Recompute and redraw everything when the slider moves."""
    theta0 = np.deg2rad([THETA1_DEG, theta2_deg])
    Rx = generate_Rx(theta0, M, d, v, f0, SNR, verbose=False)

    absR = np.abs(Rx)
    phaseR = np.degrees(np.angle(Rx))
    eigvals = np.sort(np.linalg.eigvalsh(Rx))[::-1]
    sigma_n_sq = 10 ** (-SNR / 10)

    im_abs.set_data(absR)
    im_abs.autoscale()
    im_phase.set_data(phaseR)

    plot_spectrum(ax_eig, eigvals, sigma_n_sq)

    # Annotate: steering-vector overlap rho and the eigenvalue split
    A = a_lin(theta0, M, d, v, f0)
    rho = A[:, 0].conj() @ A[:, 1]
    split_ratio = abs(rho) / M
    ax_settitle = f"theta1 = {THETA1_DEG:+.0f} deg, theta2 = {theta2_deg:+.0f} deg   " \
                  f"|rho|/M = {split_ratio:.2f}   " \
                  f"lambda = {eigvals[0]:.2f}, {eigvals[1]:.2f}, rest ~ {sigma_n_sq:.2f}"
    fig.suptitle(ax_settitle)

    fig.canvas.draw_idle()


def main():
    global fig, ax_abs, ax_phase, ax_eig, im_abs, im_phase

    fig = plt.figure(figsize=(12, 8.5))

    # 2x2 grid: two images on the left column, spectrum right column
    ax_abs = fig.add_axes([0.06, 0.55, 0.32, 0.35])
    ax_phase = fig.add_axes([0.06, 0.11, 0.32, 0.35])
    ax_eig = fig.add_axes([0.48, 0.33, 0.45, 0.55])
    ax_slider = fig.add_axes([0.48, 0.15, 0.45, 0.03])
    ax_help = fig.add_axes([0.48, 0.04, 0.45, 0.06])
    ax_help.axis("off")

    theta0_init = np.deg2rad([THETA1_DEG, THETA2_INIT_DEG])
    Rx = generate_Rx(theta0_init, M, d, v, f0, SNR, verbose=False)
    absR = np.abs(Rx)
    phaseR = np.degrees(np.angle(Rx))

    im_abs = ax_abs.imshow(absR, cmap="gray", origin="upper",
                           vmin=0, vmax=absR.max() * 1.05)
    ax_abs.set_title("|Rx(i,j)| (grayscale)")
    ax_abs.set_xlabel("mic j")
    ax_abs.set_ylabel("mic i")
    fig.colorbar(im_abs, ax=ax_abs, fraction=0.046)

    im_phase = ax_phase.imshow(phaseR, cmap="twilight", origin="upper",
                               vmin=-180, vmax=180)
    ax_phase.set_title("phase Rx(i,j) [deg]")
    ax_phase.set_xlabel("mic j")
    ax_phase.set_ylabel("mic i")
    fig.colorbar(im_phase, ax=ax_phase, fraction=0.046)

    eigvals_init = np.sort(np.linalg.eigvalsh(Rx))[::-1]
    sigma_n_sq = 10 ** (-SNR / 10)
    plot_spectrum(ax_eig, eigvals_init, sigma_n_sq)
    ax_eig.set_title("Eigenvalues of Rx")

    ax_help.text(0.0, 0.5,
                 "Drag theta2 to theta1: watch lambda_2 sink to the noise\n"
                 "floor (sources merge into one rank-1 contribution).",
                 fontsize=9, va="center")

    s = Slider(ax_slider, "theta2 [deg]", -90.0, 90.0,
               valinit=THETA2_INIT_DEG, valstep=1.0)
    s.on_changed(update)

    update(THETA2_INIT_DEG)
    plt.show()


if __name__ == "__main__":
    main()
