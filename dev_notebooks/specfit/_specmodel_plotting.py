"""Small plotting/geometry helpers shared by the tutorial and batch diagnostic.

No physical model evaluation or reference preparation is hidden here.
"""

import numpy as np


def display_wavelength(model_wavelength, requested_wavelength, minimum=1200):
    """Resolve model-scale features on an evaluation grid, without regridding it."""
    native_step = np.min(np.diff(np.asarray(model_wavelength), axis=1), axis=1)
    requested = np.asarray(requested_wavelength)
    pixels = max(minimum, int(np.ceil(2 * np.max(np.ptp(requested, axis=1) / native_step))) + 1)
    return np.stack([np.linspace(row.min(), row.max(), pixels) for row in requested])


def plot_spectra(wavelength, curves, *, titles=None, observation=None, flux_limits=None):
    """Plot already evaluated curves, optionally with an unchanged observation."""
    import matplotlib.pyplot as plt

    count = len(wavelength)
    fig, axes = plt.subplots(count, 1, figsize=(10, 3 * count), squeeze=False)
    for region, axis in enumerate(axes[:, 0]):
        if observation is not None:
            wave, flux = observation
            axis.plot(wave[region], flux[region], ".", color=".65",
                      label="sample spectrum (not fitted)")
        for label, flux in curves.items():
            axis.plot(wavelength[region], np.asarray(flux)[region], label=label)
        axis.set(xlabel="Wavelength [Å]", ylabel="Flux")
        if titles is not None:
            axis.set_title(titles[region])
        if flux_limits is not None:
            axis.set_ylim(*flux_limits)
        axis.ticklabel_format(useOffset=False, style="plain", axis="x")
        axis.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_comparison(wavelength, frozen, current):
    """Overlay evaluated predictions and their difference, one column per region."""
    import matplotlib.pyplot as plt

    count = len(wavelength)
    fig, axes = plt.subplots(2, count, figsize=(10, 6), squeeze=False, sharex="col")
    for region in range(count):
        axes[0, region].plot(wavelength[region], frozen[region], label="frozen physical prediction")
        axes[0, region].plot(wavelength[region], current[region], "--", label="jaxstar physical prediction")
        axes[0, region].set(ylabel="Flux", title=f"Region row {region}")
        axes[0, region].legend(fontsize=8)
        axes[1, region].plot(wavelength[region], current[region] - frozen[region])
        axes[1, region].set(xlabel="Wavelength [Å]", ylabel="new − frozen")
        axes[1, region].ticklabel_format(useOffset=False, style="plain", axis="x")
    fig.tight_layout()
    return fig, axes
