__all__ = ["create_mistgrid"]

import os
import pathlib
import re
import tarfile
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from astropy.constants import G, M_sun, R_sun
from scipy.interpolate import interp1d

logg_sun = np.log10((G * M_sun / R_sun**2).cgs.value)

keys = [
    "2MASS_J",
    "2MASS_H",
    "2MASS_Ks",
    "logT",
    "logg",
    "teff",
    "logage",
    "mass",
    "dmdeep",
    "logL",
    "radius",
    "mmin",
    "mmax",
    "eepmin",
    "eepmax",
]
keys += ["Gaia_G_DR2Rev", "Gaia_BP_DR2Rev", "Gaia_RP_DR2Rev"]
keys += ["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"]
keys += ["star_mass", "feh_photosphere"]
keys += ["Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I"]

HEADER = [
    "EEP",
    "log10_isochrone_age_yr",
    "initial_mass",
    "star_mass",
    "log_Teff",
    "log_R",  # v2.5
    "log_g",
    "log_L",
    "[Fe/H]_init",
    "[Fe/H]",
    "Bessell_U",
    "Bessell_B",
    "Bessell_V",
    "Bessell_R",
    "Bessell_I",
    "2MASS_J",
    "2MASS_H",
    "2MASS_Ks",
    "Kepler_Kp",
    "Kepler_D51",
    "Hipparcos_Hp",
    "Tycho_B",
    "Tycho_V",
    "Gaia_G_DR2Rev",
    "Gaia_BP_DR2Rev",
    "Gaia_RP_DR2Rev",
    "Gaia_G_MAW",
    "Gaia_BP_MAWb",
    "Gaia_BP_MAWf",
    "Gaia_RP_MAW",
    "TESS",
    "Gaia_G_EDR3",
    "Gaia_BP_EDR3",
    "Gaia_RP_EDR3",
    "Gemini_NIRI_BrG",
    "WIYN_NESSI_NB832",
    "phase",
]

MIST_URLS = {
    "v1.2": "http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit{vvcrit:.1f}_{phot_system}.txz",
    "v2.5": "https://mist.science/data/tarballs_v2.5/isos/{phot_system}.txz",
    "new": "https://mist.science/data/tarballs_v2.5/isos/{phot_system}.txz",
}


# Match both legacy and new file names, e.g.
#   MIST_v1.2_feh_m2.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd
#   feh_m025_afe_m2_vvcrit0.0_full.iso.UBVRIplus
_FEH_RE = re.compile(r"(?:^|_)feh_([^_]+)")
_AFE_RE = re.compile(r"(?:^|_)afe_([^_]+)")
_VVCRIT_RE = re.compile(r"(?:^|_)vvcrit([0-9.]+)")


def _decode_signed_token(token: str, *, scale_if_no_decimal: float) -> float:
    """Decode tokens like m2.00, p0.0, m025, p4, p0."""
    sign = 1.0
    body = token
    if token.startswith("m"):
        sign = -1.0
        body = token[1:]
    elif token.startswith("p"):
        sign = 1.0
        body = token[1:]

    if "." in body:
        value = float(body)
    else:
        value = int(body) * scale_if_no_decimal
    return sign * value


def _parse_mist_filename(path) -> dict:
    """Parse metadata from MIST file names."""
    name = pathlib.Path(path).name
    meta = {}

    m = _FEH_RE.search(name)
    if m:
        meta["feh_from_name"] = _decode_signed_token(
            m.group(1), scale_if_no_decimal=0.01
        )

    m = _AFE_RE.search(name)
    if m:
        meta["afe"] = _decode_signed_token(m.group(1), scale_if_no_decimal=0.1)

    m = _VVCRIT_RE.search(name)
    if m:
        meta["vvcrit"] = float(m.group(1))

    return meta


def _is_relevant_iso_file(path, phot_system="UBVRIplus") -> bool:
    name = pathlib.Path(path).name
    return phot_system in name and (
        name.endswith(".cmd")
        or name.endswith(f".iso.{phot_system}")
        or name.endswith(f".iso.{phot_system}.cmd")
    )


def _find_iso_files(mistdir_path, *, phot_system="UBVRIplus"):
    return sorted(
        str(p)
        for p in pathlib.Path(mistdir_path).rglob("*")
        if p.is_file() and _is_relevant_iso_file(p, phot_system=phot_system)
    )


def _normalized_mist_version(mist_version: str) -> str:
    version = str(mist_version).lower()
    if version in {"2.5", "v2.5", "new", "latest"}:
        return "v2.5"
    if version in {"1.2", "v1.2", "legacy"}:
        return "v1.2"
    raise ValueError(
        "mist_version must be one of 'v1.2' or 'v2.5' "
        f"(got {mist_version!r})"
    )


def _download_and_extract_if_needed(
    mistgriddir_path,
    *,
    mist_version="v1.2",
    phot_system="UBVRIplus",
    vvcrit=0.4,
):
    version = _normalized_mist_version(mist_version)
    url_template = MIST_URLS[version]
    url = url_template.format(phot_system=phot_system, vvcrit=vvcrit)

    extract_dir = mistgriddir_path / f"MIST_{version}_{phot_system}"
    if _find_iso_files(extract_dir, phot_system=phot_system):
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    tar_name = pathlib.Path(url).name
    tar_path = extract_dir / tar_name

    if not tar_path.exists():
        print(f"MIST files not found. Downloading {url} ...")
        try:
            with urllib.request.urlopen(url) as download_file:
                data = download_file.read()
            with open(tar_path, mode="wb") as save_file:
                save_file.write(data)
        except urllib.error.URLError as errormsg:
            raise RuntimeError(
                f"failed to download MIST tarball: {url}\n{errormsg}")

    print(f"extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:xz") as t:
        t.extractall(path=extract_dir)

    if not _find_iso_files(extract_dir, phot_system=phot_system):
        raise FileNotFoundError(
            "download/extraction finished but no matching MIST isochrone files were found "
            f"under {extract_dir}"
        )
    return extract_dir


def _resolve_mistdir(
    mistgriddir_path,
    *,
    mist_version="v1.2",
    mistdir=None,
    phot_system="UBVRIplus",
    vvcrit=0.4,
):
    if mistdir is not None:
        return pathlib.Path(mistdir)

    return _download_and_extract_if_needed(
        mistgriddir_path,
        mist_version=mist_version,
        phot_system=phot_system,
        vvcrit=vvcrit,
    )


def _eepderivative(y):
    dy = 0.5 * (y[2:] - y[:-2])
    return np.array([dy[0]] + list(dy) + [dy[-1]])


def create_mistgrid(
    mist_version="v2.5",
    *,
    mistdir=None,
    phot_system="UBVRIplus",
    vvcrit=0.4,
    afe=0.0,
):
    """Create mistgrid_iso.npz for jaxstar.mistfit.

    Parameters
    ----------
    mist_version : str, optional
        MIST version selector. Supported values are ``"v1.2"`` and ``"v2.5"``.
    mistdir : str or Path, optional
        Directory containing extracted MIST isochrone files. If provided,
        auto-download is skipped.
    phot_system : str, optional
        Photometric system suffix in the file name. Default is ``"UBVRIplus"``.
    vvcrit : float, optional
        Filter files by this rotation value. Default is 0.4.
    afe : float, optional
        Filter files by this [alpha/Fe] value. Default is 0.0.

    Notes
    -----
    This version supports both the legacy file names such as
    ``MIST_v1.2_feh_m2.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd``
    and newer file names such as
    ``feh_m025_afe_m2_vvcrit0.0_full.iso.UBVRIplus``.
    """
    mistgriddir_path = pathlib.Path(
        os.path.dirname(os.path.realpath(__file__)))
    mistgrid_path = mistgriddir_path / "mistgrid_iso.npz"
    mistdir_path = _resolve_mistdir(
        mistgriddir_path,
        mist_version=mist_version,
        mistdir=mistdir,
        phot_system=phot_system,
        vvcrit=vvcrit,
    )

    print(f"looking for MIST files in {mistdir_path}")
    filenames_all = _find_iso_files(mistdir_path, phot_system=phot_system)
    if not filenames_all:
        raise FileNotFoundError(
            f"no MIST isochrone files found under {mistdir_path} for phot_system={phot_system!r}"
        )

    filenames = []
    for filename in filenames_all:
        meta = _parse_mist_filename(filename)
        if "afe" in meta and not np.isclose(meta["afe"], afe):
            continue
        if "vvcrit" in meta and not np.isclose(meta["vvcrit"], vvcrit):
            continue
        filenames.append(filename)

    if not filenames:
        raise FileNotFoundError(
            f"no files matched afe={afe} and vvcrit={vvcrit} under {mistdir_path}"
        )

    print(f"found {len(filenames)} matching files")
    print("creating grid for mistfit...")

    df_all = pd.DataFrame(data={})
    for filename in filenames:
        meta = _parse_mist_filename(filename)
        d = pd.read_csv(filename, sep=r"\s+", comment="#",
                        header=None, names=HEADER)
        d["mass"] = d["initial_mass"]
        d["feh"] = d["[Fe/H]_init"]
        d["feh_photosphere"] = d["[Fe/H]"]
        d["teff"] = 10 ** d["log_Teff"]
        d["radius"] = np.sqrt(d["star_mass"] / 10 ** (d["log_g"] - logg_sun))
        d["logage"] = d["log10_isochrone_age_yr"]
        d["age"] = 10 ** d.logage / 1e9

        feh_from_name = meta.get("feh_from_name")
        if feh_from_name is not None:
            feh_from_table = float(np.unique(d["feh"])[0])
            if not np.isclose(feh_from_name, feh_from_table):
                raise ValueError(
                    f"filename/header [Fe/H] mismatch in {filename}: "
                    f"{feh_from_name} vs {feh_from_table}"
                )

        df_all = pd.concat([df_all, d], ignore_index=True)

    df_all = df_all.reset_index(drop=True)

    d = df_all.sort_values(["logage", "feh", "EEP"]).reset_index(drop=True)
    agrid = np.sort(list(set(d.logage)))
    fgrid = np.sort(list(set(d.feh)))[6:]
    eepgrid = np.sort(list(set(d.EEP)))

    print("logage(yr) grid:", agrid)
    print("feh grid:", fgrid)
    print("EEP grid:", eepgrid)

    d = d.rename({"log_Teff": "logT", "log_L": "logL",
                 "log_g": "logg"}, axis="columns")

    pgrids2d = []
    for key in keys:
        pgrid2d = np.zeros((len(agrid), len(fgrid), len(eepgrid)))
        for i, a in enumerate(agrid):
            for j, f in enumerate(fgrid):
                _d = d[(d.logage == a) & (d.feh == f)]
                '''
                eeparr = np.ones_like(eepgrid) * -np.inf
                eep0, eep1 = int(_d.EEP.min()), int(_d.EEP.max()) + 1
                if key == "dmdeep":
                    _marr = interp1d(_d.EEP, _d["mass"])(np.arange(eep0, eep1))
                    eeparr[eep0:eep1] = _eepderivative(_marr)
                elif key == "mmin":
                    eeparr = _d["mass"].min()
                elif key == "mmax":
                    eeparr = _d["mass"].max()
                elif key == "eepmin":
                    eeparr = eep0
                elif key == "eepmax":
                    eeparr = eep1 - 1
                else:
                    eeparr[eep0:eep1] = interp1d(
                        _d.EEP, _d[key])(np.arange(eep0, eep1))
                eeparr = np.ones_like(eepgrid, dtype=float) * -np.inf
                '''
                eeparr = np.ones_like(eepgrid, dtype=float) * -np.inf
                eepmin = int(_d.EEP.min())
                eepmax = int(_d.EEP.max())

                mask = (eepgrid >= eepmin) & (eepgrid <= eepmax)
                eep_eval = eepgrid[mask].astype(float)

                if key == "dmdeep":
                    _marr = interp1d(
                        _d.EEP, _d["mass"], bounds_error=False, fill_value=np.nan
                    )(eep_eval)
                    eeparr[mask] = _eepderivative(_marr)
                elif key == "mmin":
                    eeparr[:] = _d["mass"].min()
                elif key == "mmax":
                    eeparr[:] = _d["mass"].max()
                elif key == "eepmin":
                    eeparr[:] = eepmin
                elif key == "eepmax":
                    eeparr[:] = eepmax
                else:
                    y = pd.to_numeric(_d[key], errors="coerce")
                    eeparr[mask] = interp1d(
                        _d.EEP, y, bounds_error=False, fill_value=np.nan
                    )(eep_eval)
                pgrid2d[i][j] = eeparr
        pgrids2d.append(pgrid2d)

    np.savez(
        mistgrid_path,
        logagrid=agrid,
        fgrid=fgrid,
        eepgrid=eepgrid,
        jmag=pgrids2d[0],
        hmag=pgrids2d[1],
        kmag=pgrids2d[2],
        logt=pgrids2d[3],
        logg=pgrids2d[4],
        teff=pgrids2d[5],
        logage=pgrids2d[6],
        mass=pgrids2d[7],
        dmdeep=pgrids2d[8],
        logl=pgrids2d[9],
        radius=pgrids2d[10],
        mmin=pgrids2d[11],
        mmax=pgrids2d[12],
        eepmin=pgrids2d[13],
        eepmax=pgrids2d[14],
        gmag2=pgrids2d[15],
        bpmag2=pgrids2d[16],
        rpmag2=pgrids2d[17],
        gmag3=pgrids2d[18],
        bpmag3=pgrids2d[19],
        rpmag3=pgrids2d[20],
        star_mass=pgrids2d[21],
        feh_photosphere=pgrids2d[22],
        umag=pgrids2d[23],
        bmag=pgrids2d[24],
        vmag=pgrids2d[25],
        rmag=pgrids2d[26],
        imag=pgrids2d[27],
    )

    return mistgrid_path
