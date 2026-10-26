# Called by compare.py; all wavelengths here are vacuum Angstroms.
using Korg
using TOML

length(ARGS) == 2 || error("Usage: synthesize.jl request.toml output_directory")
request = TOML.parsefile(ARGS[1])
outdir = ARGS[2]
teff, logg = request["teff"], request["logg"]
feh, alpha_fe = request["feh"], request["alpha_fe"]
lo, hi = request["wavelength_bounds_vacuum_A"]
step = request["step_A"]
isfinite(step) && 0 < step < hi - lo || error("Invalid wavelength step")

# format_A_X takes [metals/H], [alpha/H], NOT [alpha/Fe].
# Keep the solar reference explicit; it differs from Coelho's GS98 mixture.
# https://ajwheeler.github.io/Korg.jl/stable/API/#Korg.format_A_X
solar = Korg.bergemann_2025_solar_abundances
alphas = [8, 10, 12, 14, 16, 20, 22] # Coelho: O, Ne, Mg, Si, S, Ca, Ti
A_X = Korg.format_A_X(feh, feh + alpha_fe;
                     solar_abundances=solar, alpha_elements=alphas)
@assert isapprox(A_X[26] - solar[26], feh; atol=1e-12)
@assert all(isapprox(A_X[z] - solar[z] - feh, alpha_fe; atol=1e-12) for z in alphas)

# Pass A_X so Korg handles the MARCS atmosphere's different solar reference.
# No perturbation of the requested Teff/logg at atmosphere-grid boundaries.
println("Interpolating the MARCS atmosphere")
flush(stdout)
atm = Korg.interpolate_marcs(teff, logg, A_X; perturb_at_grid_values=false)
println("Loading the APOGEE DR17 line list (first call may take several minutes)")
flush(stdout)
lines = Korg.get_APOGEE_DR17_linelist() # H band, including molecules and water

# Pad by one numerical sample: Korg's upper bound need not be included.
# synthesize applies no rotation, macroturbulence or instrumental convolution.
# Thermal/pressure/microturbulent line formation is retained. Coelho uses
# vmic=1 km/s at this logg (Coelho et al. 2005, section 3).
println("Synthesizing one Korg spectrum at ", (teff, logg, feh, alpha_fe))
flush(stdout)
solution = Korg.synthesize(atm, lines, A_X, (lo - step, hi + step, step);
                          vmic=1.0, return_cntm=true,
                          hydrogen_lines=true, use_MHD_for_hydrogen_lines=false)
wavelength = collect(solution.wavelengths)
continuum = solution.cntm
@assert all(isfinite, continuum) && all(continuum .> 0)
normalized = solution.flux ./ continuum
@assert all(isfinite, normalized)
@assert first(wavelength) <= lo && last(wavelength) >= hi

mkpath(outdir)
open(joinpath(outdir, "korg_native.csv"), "w") do io
    println(io, "wavelength_vacuum_A,flux,continuum,normalized_flux")
    for i in eachindex(wavelength)
        println(io, join((wavelength[i], solution.flux[i], continuum[i], normalized[i]), ','))
    end
end
metadata = Dict(
    "julia_version" => string(VERSION), "korg_version" => string(pkgversion(Korg)),
    "teff" => teff, "logg" => logg, "feh" => feh, "alpha_fe" => alpha_fe,
    "format_A_X_arguments" => [feh, feh + alpha_fe],
    "solar_abundances" => "bergemann_2025_solar_abundances",
    "alpha_atomic_numbers" => alphas, "A_X" => A_X,
    "atmosphere" => "interpolate_marcs(Teff, logg, A_X; perturb_at_grid_values=false)",
    "linelist" => "get_APOGEE_DR17_linelist(include_water=true)",
    "vmic_km_s" => 1.0, "vsini_km_s" => 0.0, "vmacro_km_s" => 0.0,
    "instrumental_convolution" => false, "normalization" => "flux / cntm",
    "hydrogen_lines" => true, "use_MHD_for_hydrogen_lines" => false,
    "wavelength_medium" => "vacuum", "wavelength_unit" => "Angstrom",
    "step_A" => step, "synthesis_bounds_A" => [first(wavelength), last(wavelength)],
)
open(joinpath(outdir, "korg_metadata.toml"), "w") do io
    TOML.print(io, metadata)
end
