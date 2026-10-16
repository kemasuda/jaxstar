# MIST v2.5 native track columns

The optional `include_structure=True` / `--include-structure` selection retains
all 66 native non-isotope columns listed below, with their original names and
units. The 102 individual surface/central isotope-abundance columns are excluded.
See the [grid guide](mistgrid.md#optional-v25-structural-columns) for building
and loading this grid. The default compact grid retains its existing fields.

To predict all 66 columns from an expanded `MistModel`, pass the input
coordinates (`mass_init`, `feh_init`, and `eep`) as `params`:

```python
prediction = model(params, keys=tuple(model.metadata["native_columns"]))
moment_of_inertia = prediction["total_moment_of_inertia"]
```

| Group | Contents | Native column names |
| --- | --- | --- |
| Basic stellar properties | Age, current mass, mass-loss rate, temperature, gravity, luminosity and radius | `star_age`, `star_mass`, `star_mdot`, `log_Teff`, `log_g`, `log_L`, `log_R` |
| Core masses | Helium-, carbon-, oxygen-rich cores and convective core | `he_core_mass`, `c_core_mass`, `o_core_mass`, `mass_conv_core` |
| Surface composition | Surface metal mass fraction and C/O number ratio | `log_surf_cell_z`, `surf_num_c12_div_num_o16` |
| Central conditions | Temperature, density, degeneracy, plasma coupling and rotation | `log_center_T`, `log_center_Rho`, `center_degeneracy`, `center_gamma`, `center_omega` |
| Convective-envelope boundaries | Top and bottom coordinates in mass and radius | `conv_env_top_mass`, `conv_env_bot_mass`, `conv_env_top_radius`, `conv_env_bot_radius` |
| Convective turnover times | Local at one mixing length (`l_t`), half a mixing length (`l_b`) above the base, and global (`g`) | `conv_env_turnover_time_l_t`, `conv_env_turnover_time_l_b`, `conv_env_turnover_time_g` |
| Inertia and internal structure | Total stellar moment of inertia and apsidal constant | `total_moment_of_inertia`, `apsidal_constant_k2` |
| Rotation and shape | Surface rotation, critical rotation, angular momentum and equatorial/polar radii | `surf_avg_omega`, `surf_avg_v_rot`, `surf_avg_omega_crit`, `surf_avg_omega_div_omega_crit`, `surf_avg_v_crit`, `surf_avg_v_div_v_crit`, `total_angular_momentum`, `surf_r_equatorial_div_r`, `surf_r_polar_div_r` |
| Gravity darkening | Pole-on and equator-on luminosities and effective temperatures | `grav_dark_L_polar`, `grav_dark_Teff_polar`, `grav_dark_L_equatorial`, `grav_dark_Teff_equatorial` |
| Asteroseismology | Large frequency separation, g-mode period spacing, peak-power frequency and acoustic cutoff | `delta_nu`, `delta_Pg`, `nu_max`, `acoustic_cutoff` |
| Luminosity and energy | Eddington ratio, burning/gravitational luminosities, thermal and envelope binding energies | `log_L_div_Ledd`, `log_LH`, `log_LHe`, `log_LZ`, `log_abs_Lgrav`, `e_thermal`, `envelope_binding_energy` |
| Nuclear-burning diagnostics | Contributions from reaction categories | `pp`, `cno`, `tri_alfa`, `burn_c`, `burn_n`, `burn_o`, `c12_c12` |
| Convection, radiation and pressure diagnostics | Velocity, temperature-gradient, pressure and Eddington-ratio diagnostics | `surf_avg_Lrad_div_Ledd`, `v_div_csound_surf`, `max_conv_vel_div_csound`, `max_gradT_div_grada`, `gradT_excess_alpha`, `min_Pgas_div_P`, `max_L_rad_div_Ledd` |
| Evolutionary phase | Numerical phase value; interpolated values are not categorical labels | `phase` |

Convective-envelope boundary masses and radii are in solar masses and solar radii. Turnover times are stored in seconds, total moment of inertia in g cm², and `apsidal_constant_k2` is dimensionless. The [v2.5 fitting notebook](../examples/mistfit/fit_track_v25.ipynb) converts times to days and scales inertia by 10^53 for its corner plot. Envelope thickness and mass are the differences between their top and bottom coordinates.

The native fields use linear interpolation in the stored quantity. Finite source zeros/sentinels are retained; zero envelope boundaries/times can indicate no surface convective envelope. See the [MIST column descriptions](https://mist.science/README_tables.pdf) and [local validation record](../benchmarks/mist_v25/README.md) for definitions and checks.
