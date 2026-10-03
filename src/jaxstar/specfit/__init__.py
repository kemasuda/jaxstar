"""Spectral grids, physical models, observations and Gaussian continuum helpers."""

from .libraries import (
    load_bosz,
    load_coelho,
    load_tlusty,
    prepare_bosz,
    prepare_coelho,
    prepare_tlusty,
)
from .storage import load_spectral_grid, save_spectral_grid
from .sampling import is_log_uniform, resample_spectral_grid
from .model import SpecModel, SpectralDecomposition
from .observation import Observation
from .continuum import (
    ContinuumPrior,
    ContinuumPosterior,
    chebyshev_basis,
    continuum_design_matrix,
    evaluate_continuum,
    apply_continuum,
    continuum_prior,
    marginalized_continuum_log_likelihood,
    continuum_posterior,
)

__all__ = [
    "load_coelho", "load_bosz", "load_tlusty",
    "prepare_coelho", "prepare_bosz", "prepare_tlusty",
    "load_spectral_grid", "save_spectral_grid",
    "is_log_uniform", "resample_spectral_grid",
    "SpecModel", "SpectralDecomposition",
    "Observation",
    "ContinuumPrior", "ContinuumPosterior",
    "chebyshev_basis", "continuum_design_matrix", "evaluate_continuum",
    "apply_continuum", "continuum_prior",
    "marginalized_continuum_log_likelihood", "continuum_posterior",
]
