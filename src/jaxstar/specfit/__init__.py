"""Spectral grids, physical models, observations and Gaussian likelihoods."""

from .libraries import (
    select_grid_paths,
    load_bosz,
    load_coelho,
    load_tlusty,
    prepare_bosz,
    prepare_coelho,
    prepare_tlusty,
)
from .storage import load_spectral_grid, save_spectral_grid
from .ip_calibration import IPCalibration, read_ip_calibration
from .results import FitResult, load_fit_result, save_fit_result
from .plotting import plot_mcmc_diagnostics
from .sampling import is_log_uniform, resample_spectral_grid
from .model import SpecModel, SpectralDecomposition
from .observation import Observation
from .specfit import SpecFit
from .configured_model import ConfiguredModel
from .reconstruction import reconstruct_single
from .numpyro_model import (
    make_model_single,
    model_single,
    single_star_params,
    physical_logg_max,
    empirical_vmic,
    empirical_vmacro_valenti_fischer2005,
)
from .continuum import (
    ContinuumPrior,
    ContinuumPosterior,
    chebyshev_basis,
    continuum_design_matrix,
    evaluate_continuum,
    apply_continuum,
    continuum_prior,
    continuum_posterior,
)
from .likelihood import (
    marginalized_continuum_log_likelihood,
    prepare_gp_observation,
    gp_marginalized_continuum_log_likelihood,
    gp_continuum_posterior,
    gp_conditional_mean,
)

__all__ = [
    "select_grid_paths",
    "load_coelho", "load_bosz", "load_tlusty",
    "prepare_coelho", "prepare_bosz", "prepare_tlusty",
    "load_spectral_grid", "save_spectral_grid",
    "IPCalibration", "read_ip_calibration",
    "FitResult", "load_fit_result", "save_fit_result",
    "plot_mcmc_diagnostics",
    "is_log_uniform", "resample_spectral_grid",
    "SpecModel", "SpectralDecomposition",
    "Observation", "SpecFit",
    "ConfiguredModel", "make_model_single", "reconstruct_single",
    "model_single", "single_star_params",
    "physical_logg_max", "empirical_vmic", "empirical_vmacro_valenti_fischer2005",
    "ContinuumPrior", "ContinuumPosterior",
    "chebyshev_basis", "continuum_design_matrix", "evaluate_continuum",
    "apply_continuum", "continuum_prior",
    "marginalized_continuum_log_likelihood", "continuum_posterior",
    "prepare_gp_observation", "gp_marginalized_continuum_log_likelihood",
    "gp_continuum_posterior", "gp_conditional_mean",
]
