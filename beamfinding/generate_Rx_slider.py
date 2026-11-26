"""
IP3 - M3: Interactive Rx explorer with sliders.

Shows the data covariance matrix Rx of two sources, with BOTH source
directions (theta1 and theta2) adjustable via matplotlib sliders.
Watch what happens as you drag the sources towards each other:

- the eigenvalue spectrum splits (M +/- |rho|) and finally collapses
  (lambda_2 -> noise floor) when both sources coincide,
- the |Rx| image loses structure (rank-1: a a^H is rank one),
- the phase image goes from "two ramps superposed" to one clean ramp.

Layout: |Rx| and phase images (left column), eigenvalue spectrum
(right column, top) and two sliders (right column, bottom).

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
THETA1_INIT_DEG = -20.0  # first source (slider 1)
THETA2_INIT_DEG = 30.0   # second source (slider 2)


def plot_spectrum(ax, eigvals, sigma_n_sq):
    """Eigenvalue spectrum with noise floor, as bar plot."""
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


def update(theta1_deg, theta2_deg):
    """Recompute and redraw everything when a slider moves."""
    theta0 = np.deg2rad([theta1_deg, theta2_deg])
    Rx = generate_Rx(theta0, M, d, v, f0, SNR, verbose=False)

    absR = np.abs(Rx)
    phaseR = np.degrees(np.angle(Rx))
    eigvals = np.sort(np.linalg.eigvalsh(Rx))[::-1]
    sigma_n_sq = 10 ** (-SNR / 10)

    im_abs.set_data(absR)
    im_abs.autoscale()
    im_phase.set_data(phaseR)

    plot_spectrum(ax_eig, eigvals, sigma_n_sq)

    # Running commentary in the title: steering-vector overlap rho
    A = a_lin(theta0, M, d, v, f0)
    rho = A[:, 0].conj() @ A[:, 1]
    split_ratio = abs(rho) / M
    fig.suptitle(
        f"theta1 = {theta1_deg:+.0f} deg, theta2 = {theta2_deg:+.0f} deg   "
        f"|rho|/M = {split_ratio:.2f}   "
        f"lambda = {eigvals[0]:.2f}, {eigvals[1]:.2f}, "
        f"rest ~ {sigma_n_sq:.2f}")

    fig.canvas.draw_idle()


def main():
    global fig, ax_abs, ax_phase, ax_eig, im_abs, im_phase

    fig = plt.figure(figsize=(12, 8.5))

    # Images on the left column, spectrum + sliders on the right column
    ax_abs = fig.add_axes([0.06, 0.55, 0.32, 0.35])
    ax_phase = fig.add_axes([0.06, 0.11, 0.32, 0.35])
    ax_eig = fig.add_axes([0.48, 0.42, 0.45, 0.46])
    ax_slider1 = fig.add_axes([0.48, 0.30, 0.45, 0.03])
    ax_slider2 = fig.add_axes([0.48, 0.22, 0.45, 0.03])
    ax_help = fig.add_axes([0.48, 0.06, 0.45, 0.10])
    ax_help.axis("off")

    theta0_init = np.deg2rad([THETA1_INIT_DEG, THETA2_INIT_DEG])
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
                 "Drag theta1 or theta2. Move the sources together: watch\n"
                 "lambda_2 sink to the noise floor (sources merge into a\n"
                 "single rank-1 contribution). Move them to symmetric\n"
                 "positions (e.g. +/- same angle) for eigenvalue surprises.",
                 fontsize=9, va="center")

    s1 = Slider(ax_slider1, "theta1 [deg]", -90.0, 90.0,
                valinit=THETA1_INIT_DEG, valstep=1.0)
    s2 = Slider(ax_slider2, "theta2 [deg]", -90.0, 90.0,
                valinit=THETA2_INIT_DEG, valstep=1.0)

    s1.on_changed(lambda val: update(val, s2.val))
    s2.on_changed(lambda val: update(s1.val, val))

    update(THETA1_INIT_DEG, THETA2_INIT_DEG)
    plt.show()


if __name__ == "__main__":
    main()
