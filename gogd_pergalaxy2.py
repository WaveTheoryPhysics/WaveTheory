"""
Per-galaxy α₀ fitting for morphology classification
=====================================================

Fits the best α₀ independently for each galaxy by grid search,
then writes a rich CSV for downstream clustering / morphology analysis.

This is a companion diagnostic script — results are NOT for the main
paper (which uses a single global α₀ = κ). The per-galaxy fit shows
how A_morph = α₀_best / κ varies across the sample and whether
natural morphology groups emerge from the data alone.

Physics
-------
Master formula (crossfade, m=1):

    a_eff(r) = (1 - μ(r)) · a_b(r)  +  μ(r) · α₀ · √(a_b · a_*)

    μ(r)  = 1 / (1 + Mb/r²)          [local compactness crossfade]
    a_b   = G Mb(r) / r²              [baryonic acceleration]
    a_*   = c² / (π R)                [S³ curvature scale]
    α₀    = A_morph · κ               [fitted per galaxy]
    κ     = 1 + 1/π²  ≈ 1.1013        [S³ curvature factor]

Output CSV columns
------------------
name, N, alpha0_best, A_morph, chi2_WT_best, chi2_WT_kappa,
chi2_MOND, chi2_LCDM, WT_wins, MOND_wins, LCDM_wins,
Vmax_kms, Rout_kpc, f_gas_mean, f_gas_outer, M200_Msun,
log10_M200, a_char, log10_a_char, m_used

Typical run
-----------
    python3 gogd_pergalaxy.py --chi2csv pergalaxy_alpha_fit.csv

    # Restrict to a vmax range
    python3 gogd_pergalaxy.py \\
        --m 1.0 \\
        --grid 0.05,3.0,0.02 \\
        --vmin 0 --vmax 120 \\
        --chi2csv pergalaxy_dwarfs.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import tempfile
import urllib.request
import zipfile
from io import StringIO
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. Constants
# =====================================================================
PI      = math.pi
C_SI    = 299_792_458.0
G_SI    = 6.67430e-11
A0_MOND = 1.2e-10
MSUN    = 1.98847e30
MPC_M   = 3.085677581e22
KPC2M   = 3.085677581491367e19
G_KPC   = 4.30091e-6

R_UNIVERSE_M = 4.286332662e26
A_STAR       = C_SI**2 / (PI * R_UNIVERSE_M)
KAPPA        = 1.0 + 1.0 / PI**2   # S³ curvature factor


# =====================================================================
# 2. Data loading  (identical to gogd.py)
# =====================================================================
def loadtxt_safe(fp: pathlib.Path) -> np.ndarray:
    try:
        return np.loadtxt(fp, comments="#")
    except (UnicodeDecodeError, ValueError):
        pass
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(fp, "r", encoding=enc, errors="ignore") as fh:
                txt = fh.read()
            txt = txt.replace(" .", " nan").replace(". ", "nan ")
            data = np.genfromtxt(StringIO(txt), comments="#")
            if data.ndim == 1:
                data = data.reshape(-1, data.shape[0])
            return data
        except Exception:
            continue
    raise RuntimeError(f"Could not load {fp}")


def load_rotmod(path: pathlib.Path) -> Tuple[np.ndarray, ...]:
    arr = loadtxt_safe(path)
    r_kpc, vobs, verr = arr[:, 0], arr[:, 1], arr[:, 2]
    vgas, vdisk, vbul = arr[:, 3], arr[:, 4], arr[:, 5]
    vstar = np.sqrt(vdisk**2 + vbul**2)
    return r_kpc, vobs, verr, vgas, vstar


# =====================================================================
# 3. WT velocity — crossfade master formula
# =====================================================================
def fit_wt_2d_fast(r_kpc, v_obs, vgas_kms, vstar_kms, sigma,
                   alpha_grid, m_exp,
                   upsilon_min=0.1, upsilon_max=2.0, n_ups=50):
    """Vectorized 2D grid search — loops over Υ★, vectorizes over α₀."""
    ups_grid = np.linspace(upsilon_min, upsilon_max, n_ups)
    
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg = np.asarray(vgas_kms, dtype=float) * 1e3
    
    best_chi2 = np.inf
    best_a0 = alpha_grid[0]
    best_ups = 1.0
    
    for ups in ups_grid:
        vs = np.asarray(vstar_kms, dtype=float) * 1e3 * math.sqrt(ups)
        safe_r = np.where(r_m > 0.0, r_m, np.nan)
        a_b = np.nan_to_num(
            np.where(r_m > 0.0, vg**2 / safe_r, 0.0) +
            np.where(r_m > 0.0, vs**2 / safe_r, 0.0),
            nan=0.0, posinf=0.0, neginf=0.0)
        
        u = np.clip(a_b / A_STAR, 0.0, np.inf)
        mu = 1.0 / (1.0 + np.power(u, m_exp))
        sqrt_ab_astar = np.sqrt(np.clip(a_b * A_STAR, 0.0, np.inf))
        
        # Vectorize over alpha_grid
        for a0 in alpha_grid:
            a_eff = np.clip((1.0 - mu) * a_b + mu * a0 * sqrt_ab_astar, 0.0, np.inf)
            v_pred = np.sqrt(a_eff * r_m) / 1e3
            chi2 = np.sum(((v_obs - v_pred) / sigma)**2) / len(v_obs)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_a0 = a0
                best_ups = ups
    
    return float(best_a0), float(best_ups), float(best_chi2)
    
def v_wt_with_ml(r_kpc, vgas_kms, vstar_kms, alpha0, m_exp, upsilon):
    """WT master formula with variable stellar M/L."""
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg = np.asarray(vgas_kms, dtype=float) * 1e3
    vs = np.asarray(vstar_kms, dtype=float) * 1e3 * math.sqrt(upsilon)

    safe_r = np.where(r_m > 0.0, r_m, np.nan)
    a_g = np.where(r_m > 0.0, vg**2 / safe_r, 0.0)
    a_s = np.where(r_m > 0.0, vs**2 / safe_r, 0.0)
    a_b = np.nan_to_num(a_g + a_s, nan=0.0, posinf=0.0, neginf=0.0)

    u = np.clip(a_b / A_STAR, 0.0, np.inf)
    mu = 1.0 / (1.0 + np.power(u, m_exp))

    a_shell = alpha0 * np.sqrt(np.clip(a_b * A_STAR, 0.0, np.inf))
    a_eff = np.clip((1.0 - mu) * a_b + mu * a_shell, 0.0, np.inf)
    return np.sqrt(a_eff * r_m) / 1e3


def fit_wt_upsilon(r_kpc, v_obs, vgas_kms, vstar_kms, sigma, alpha0, m_exp,
                   upsilon_min=0.1, upsilon_max=2.0, n_grid=500):
    """Grid search over Υ★ for WT at fixed alpha0=kappa."""
    grid = np.linspace(upsilon_min, upsilon_max, n_grid)
    best_chi2 = np.inf
    best_ups = 1.0
    for ups in grid:
        v_pred = v_wt_with_ml(r_kpc, vgas_kms, vstar_kms, alpha0, m_exp, ups)
        resid = (v_obs - v_pred) / sigma
        chi2 = np.sum(resid**2) / len(v_obs)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_ups = ups
    return best_ups, best_chi2
    
def v_wt(r_kpc: np.ndarray, vgas_kms: np.ndarray, vstar_kms: np.ndarray,
         alpha0: float, m_exp: float) -> np.ndarray:
    """
    Crossfade master formula:
        a_eff = (1 - μ) · a_b  +  μ · α₀ · √(a_b · a_*)
        μ(r)  = 1 / (1 + (a_b/a_*)^m)
    """
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg  = np.asarray(vgas_kms,  dtype=float) * 1e3
    vs  = np.asarray(vstar_kms, dtype=float) * 1e3

    safe_r = np.where(r_m > 0.0, r_m, np.nan)
    a_g = np.where(r_m > 0.0, vg**2 / safe_r, 0.0)
    a_s = np.where(r_m > 0.0, vs**2 / safe_r, 0.0)
    a_b = np.nan_to_num(a_g + a_s, nan=0.0, posinf=0.0, neginf=0.0)

    u   = np.clip(a_b / A_STAR, 0.0, np.inf)
    mu  = 1.0 / (1.0 + np.power(u, m_exp))

    a_shell = alpha0 * np.sqrt(np.clip(a_b * A_STAR, 0.0, np.inf))
    a_eff   = np.clip((1.0 - mu) * a_b + mu * a_shell, 0.0, np.inf)
    return np.sqrt(a_eff * r_m) / 1e3

def v_mond_with_ml(r_kpc, vgas_kms, vstar_kms, upsilon):
    """
    MOND prediction with variable stellar mass-to-light ratio.
    upsilon scales the stellar contribution: Vstar -> sqrt(upsilon) * Vstar.
    Uses the RAR interpolation function nu(y) = 1/(1 - exp(-sqrt(y))).
    """
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg = np.asarray(vgas_kms, dtype=float) * 1.0e3
    vs = np.asarray(vstar_kms, dtype=float) * 1.0e3 * math.sqrt(upsilon)

    safe_r = np.where(r_m > 0.0, r_m, np.nan)
    a_bar = (np.nan_to_num(vg**2 / safe_r, nan=0.0)
             + np.nan_to_num(vs**2 / safe_r, nan=0.0))

    y = np.where(a_bar > 0.0, a_bar / A0_MOND, 1e-30)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(y)))
    a_mond = a_bar * nu

    return np.sqrt(np.clip(a_mond, 0.0, np.inf) * r_m) / 1.0e3


def fit_mond_pergalaxy(r_kpc, v_obs, vgas_kms, vstar_kms, sigma,
                       upsilon_min=0.1, upsilon_max=2.0,
                       n_grid=500):
    """
    Grid search over stellar M/L ratio to minimise chi2.
    Returns best upsilon, best chi2_nu.
    """
    grid = np.linspace(upsilon_min, upsilon_max, n_grid)
    best_chi2 = np.inf
    best_ups = 1.0

    for ups in grid:
        v_pred = v_mond_with_ml(r_kpc, vgas_kms, vstar_kms, ups)
        resid = (v_obs - v_pred) / sigma
        chi2 = np.sum(resid**2) / len(v_obs)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_ups = ups

    return best_ups, best_chi2
    
def v_mond(r_kpc: np.ndarray, vgas_kms: np.ndarray, vstar_kms: np.ndarray) -> np.ndarray:
    """MOND circular velocity using the 'simple' interpolation function.

    Implements the McGaugh, Lelli & Schombert (2016) convention:
        nu(y) = 1 / (1 - exp(-sqrt(y))),  y = a_N / a_0
        a_eff  = a_N * nu(y)

    This is the interpolation function used throughout the SPARC/RAR
    literature and is the standard benchmark for comparison.
    """
    r_m    = np.asarray(r_kpc,    dtype=float) * KPC2M
    vg     = np.asarray(vgas_kms, dtype=float) * 1.0e3
    vs     = np.asarray(vstar_kms, dtype=float) * 1.0e3
    safe_r = np.where(r_m > 0.0, r_m, np.nan)
    a_n    = (np.nan_to_num(vg**2 / safe_r, nan=0.0)
            + np.nan_to_num(vs**2 / safe_r, nan=0.0))
    y      = np.where(a_n > 0.0, a_n / A0_MOND, 0.0)
    nu     = np.where(y > 0.0, 1.0 / (1.0 - np.exp(-np.sqrt(y))), 1.0)
    a_eff  = a_n * nu
    return np.sqrt(np.clip(a_eff * r_m, 0.0, np.inf)) / 1.0e3

# =====================================================================
# 4. ΛCDM NFW benchmark  (identical to gogd.py)
# =====================================================================
def lcdm_velocity_with_ml(r_kpc, vgas_kms, vstar_kms, m200, upsilon, h0=70.0):
    """ΛCDM total velocity with variable stellar M/L."""
    vg = np.asarray(vgas_kms, dtype=float)
    vs = np.asarray(vstar_kms, dtype=float) * math.sqrt(upsilon)
    v_bar = np.sqrt(np.clip(vg**2 + vs**2, 0.0, np.inf))
    v_nfw = nfw_velocity(r_kpc, m200, h0)
    return np.sqrt(v_bar**2 + v_nfw**2)


def fit_lcdm_upsilon(r_kpc, v_obs, vgas_kms, vstar_kms, sigma, m200,
                     upsilon_min=0.1, upsilon_max=2.0, n_grid=500, h0=70.0):
    """Grid search over Υ★ for ΛCDM with fixed M200 from Vmax."""
    grid = np.linspace(upsilon_min, upsilon_max, n_grid)
    best_chi2 = np.inf
    best_ups = 1.0
    for ups in grid:
        v_pred = lcdm_velocity_with_ml(r_kpc, vgas_kms, vstar_kms, m200, ups, h0)
        resid = (v_obs - v_pred) / sigma
        chi2 = np.sum(resid**2) / len(v_obs)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_ups = ups
    return best_ups, best_chi2
    
def lcdm_total_velocity(r_kpc, v_bar, m200, h0=70.0):
    """Total velocity: quadrature sum of baryonic + NFW halo."""
    v_nfw = nfw_velocity(r_kpc, m200, h0)
    return np.sqrt(v_bar**2 + v_nfw**2)


def fit_m200_pergalaxy(r_kpc, v_obs, v_bar, sigma,
                       log_m200_min=8.0, log_m200_max=14.0,
                       n_grid=500, h0=70.0):
    """
    Grid search over log10(M200/Msun) to minimise chi2.
    Returns best M200, best chi2_nu.
    """
    log_grid = np.linspace(log_m200_min, log_m200_max, n_grid)
    best_chi2 = np.inf
    best_m200 = 10.0**log_m200_min

    for lg in log_grid:
        m200 = 10.0**lg
        v_pred = lcdm_total_velocity(r_kpc, v_bar, m200, h0)
        resid = (v_obs - v_pred) / sigma
        chi2 = np.sum(resid**2) / len(v_obs)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_m200 = m200

    return best_m200, best_chi2

def concentration_dutton_maccio(m200: float) -> float:
    return 10.0 ** (0.537 - 0.097 * math.log10(m200 / 1e12))


def r200_from_m200(m200: float, h0: float = 70.0) -> float:
    return 206.0 * (m200 / 1e12) ** (1.0 / 3.0) * (h0 / 70.0) ** (-2.0 / 3.0)


def nfw_velocity(r_kpc: np.ndarray, m200: float, h0: float = 70.0) -> np.ndarray:
    r = np.asarray(r_kpc, dtype=float)
    if m200 <= 0.0:
        return np.zeros_like(r)
    c200 = concentration_dutton_maccio(m200)
    rs   = r200_from_m200(m200, h0) / c200
    def f(x): return np.log(1.0 + x) - x / (1.0 + x)
    xs   = np.where(r > 0.0, r / rs, 1e-12)
    m_enc = m200 * f(xs) / f(np.array([c200]))[0]
    v2   = np.where(r > 0.0, G_KPC * m_enc / r, 0.0)
    return np.sqrt(np.clip(v2, 0.0, np.inf))


def estimate_M200(fp: pathlib.Path, h0: float = 70.0) -> float:
    vmax = float(np.nanmax(loadtxt_safe(fp)[:, 1])) * 1e3
    h0_s = (h0 * 1e3) / MPC_M
    return (vmax**3) / (10.0 * G_SI * h0_s) / MSUN


# =====================================================================
# 5. χ² helpers
# =====================================================================
def rchi2(vobs, vpred, verr):
    n = len(vobs)
    if n == 0:
        return float("nan")
    return float(np.sum(((vobs - vpred) / verr) ** 2) / n)


def best_alpha0_for_galaxy(fp: pathlib.Path, alpha_grid: np.ndarray,
                            m_exp: float) -> Tuple[float, float]:
    """Grid-search α₀ for a single galaxy. Returns (best_alpha0, best_chi2)."""
    r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)
    best_a, best_c = alpha_grid[0], float("inf")
    for a in alpha_grid:
        c = rchi2(vobs, v_wt(r_kpc, vgas, vstar, a, m_exp), verr)
        if np.isfinite(c) and c < best_c:
            best_c, best_a = c, a
    return float(best_a), float(best_c)


# =====================================================================
# 6. Physical observables from rotmod
# =====================================================================
def galaxy_observables(fp: pathlib.Path, h0: float = 70.0) -> dict:
    r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)

    vmax     = float(np.nanmax(vobs)) if len(vobs) else float("nan")
    rout     = float(r_kpc[-1])       if len(r_kpc) else float("nan")

    # Characteristic acceleration
    if rout > 0 and np.isfinite(vmax):
        a_char = (vmax * 1e3)**2 / (rout * KPC2M)
        log10_a = math.log10(a_char)
    else:
        a_char = float("nan")
        log10_a = float("nan")

    # Gas fraction profile
    vg2 = vgas**2;  vs2 = vstar**2;  tot = vg2 + vs2
    mask = tot > 0
    fgas = np.full_like(vg2, float("nan"))
    fgas[mask] = vg2[mask] / tot[mask]
    f_mean  = float(np.nanmean(fgas))  if np.any(mask) else float("nan")
    f_outer = float(fgas[-1])          if (len(fgas) and np.isfinite(fgas[-1])) else f_mean

    # M200
    try:
        m200 = estimate_M200(fp, h0)
        log10_m200 = math.log10(m200) if m200 > 0 else float("nan")
    except Exception:
        m200 = float("nan")
        log10_m200 = float("nan")

    return dict(vmax=vmax, rout=rout, a_char=a_char, log10_a=log10_a,
                f_mean=f_mean, f_outer=f_outer, m200=m200, log10_m200=log10_m200)


# =====================================================================
# 7. CLI
# =====================================================================
def fmt(x):
    return f"{x:.6g}" if (x is not None and np.isfinite(x)) else ""


def build_parser():
    p = argparse.ArgumentParser(
        description="Per-galaxy α₀ fitting for morphology classification.")
    p.add_argument("--datadir", default="data/Rotmod_LTG")
    p.add_argument("--grid", default="0.05,3.0,0.001",
                   help="α₀ grid: min,max,step  (e.g. 0.05,3.0,0.01)")
    p.add_argument("--m", type=float, default=1.0,
                   help="Transition exponent m (default: 1.0)")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=float("inf"))
    p.add_argument("--chi2csv", default="pergalaxy_alpha_fit.csv")
    p.add_argument("--H0", type=float, default=70.0)
    p.add_argument("--names", default="",
                   help="Comma-separated galaxy names; overrides vmin/vmax")
    return p


def main():
    args = build_parser().parse_args()

    dpath = pathlib.Path(args.datadir)
    if not dpath.exists():
        print("[INFO] downloading SPARC archive…")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        urllib.request.urlretrieve(
            "https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip", tmp.name)
        zipfile.ZipFile(tmp.name).extractall(dpath)

    files = sorted(fp for fp in dpath.glob("*_rotmod.dat")
                   if not fp.name.startswith("._"))
    if not files:
        sys.exit("No rotmod files found.")

    # Sample selection
    names_list = [n.strip() for n in args.names.split(",") if n.strip()]
    if names_list:
        sample = [dpath / f"{n}_rotmod.dat" for n in names_list
                  if (dpath / f"{n}_rotmod.dat").exists()]
    else:
        def vmax_fp(fp):
            return float(np.nanmax(loadtxt_safe(fp)[:, 1]))
        sample = [fp for fp in files if args.vmin <= vmax_fp(fp) <= args.vmax]

    if not sample:
        sys.exit("No galaxies matched the selection criteria.")

    gmin, gmax, gstep = map(float, args.grid.split(","))
    alpha_grid = np.arange(gmin, gmax + 0.5 * gstep, gstep)

    print(f"Per-galaxy fitting: {len(sample)} galaxies")
    print(f"WT α₀ grid: {gmin:.3f} → {gmax:.3f}, step {gstep:.3f}  ({len(alpha_grid)} points)")
    print(f"m = {args.m:.3f},  κ = {KAPPA:.6f}")
    print()

    # -- CSV header --
    header = [
        "name", "N",
        # WT
        "alpha0_best", "A_morph", "chi2_WT_best", "chi2_WT_kappa",
        "ups_WT_best", "chi2_WT_ups",
        "alpha0_2d", "ups_WT_2d", "chi2_WT_2d",
        # MOND
        "ups_MOND_best", "chi2_MOND_fixed", "chi2_MOND_best",
        # LCDM
        "log10_M200_fixed", "log10_M200_best",
        "chi2_LCDM_fixed", "chi2_LCDM_best",
        "ups_LCDM_best", "chi2_LCDM_ups",
        # Winners
        "win_0param", "win_1param","win_1param_ups",
        # Observables
        "Vmax_kms",
    ]

    out_path = pathlib.Path(args.chi2csv)
    rows = []

    for i, fp in enumerate(sample, 1):
        name = fp.stem.replace("_rotmod", "")
        if i % 25 == 0 or i == 1 or i == len(sample):
            print(f"  [{i:3d}/{len(sample)}]  {name}", flush=True)

        r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)
        N = len(vobs)
        vmax_kms = float(np.nanmax(vobs)) if N > 0 else float("nan")

        # Baryonic velocity (for LCDM quadrature sum)
        v_bar = np.sqrt(np.clip(vgas**2 + vstar**2, 0.0, np.inf))

        # Safe errors (avoid division by zero)
        sigma = np.where(verr > 0.0, verr, 1.0)

        # =============================================================
        # WT: 0-param (kappa) and 1-param (best alpha0)
        # =============================================================
        a0_best, chi2_wt_best = best_alpha0_for_galaxy(fp, alpha_grid, args.m)
        A_morph = a0_best / KAPPA
        chi2_wt_kappa = rchi2(vobs, v_wt(r_kpc, vgas, vstar, KAPPA, args.m), verr)
        
        # =============================================================
        # WT 2D diagnostic: (α₀, Υ★) joint fit
        # =============================================================
        a0_2d, ups_2d, chi2_wt_2d = fit_wt_2d_fast(
            r_kpc, vobs, vgas, vstar, sigma, alpha_grid, args.m,
            upsilon_min=0.1, upsilon_max=2.0, n_ups=50)

        # =============================================================
        # MOND: 0-param (fixed a0, Υ★=1) and 1-param (fitted Υ★)
        # =============================================================
        chi2_mond_fixed = rchi2(vobs, v_mond(r_kpc, vgas, vstar), verr)
        ups_mond_best, chi2_mond_best = fit_mond_pergalaxy(
            r_kpc, vobs, vgas, vstar, sigma)

        # =============================================================
        # ΛCDM: 0-param (M200 from Vmax) and 1-param (fitted M200)
        # =============================================================
        try:
            m200_fixed = estimate_M200(fp, args.H0)
            log10_m200_fixed = math.log10(m200_fixed) if m200_fixed > 0 else float("nan")
            v_lcdm_fixed = lcdm_total_velocity(r_kpc, v_bar, m200_fixed, args.H0)
            chi2_lcdm_fixed = rchi2(vobs, v_lcdm_fixed, verr)
        except Exception:
            m200_fixed = float("nan")
            log10_m200_fixed = float("nan")
            chi2_lcdm_fixed = float("nan")

        try:
            m200_best, chi2_lcdm_best = fit_m200_pergalaxy(
                r_kpc, vobs, v_bar, sigma, h0=args.H0)
            log10_m200_best = math.log10(m200_best) if m200_best > 0 else float("nan")
        except Exception:
            m200_best = float("nan")
            log10_m200_best = float("nan")
            chi2_lcdm_best = float("nan")

        # =============================================================
        # TIER 1b: All three fit Υ★ (same single parameter)
        # =============================================================
        ups_wt_best, chi2_wt_ups = fit_wt_upsilon(r_kpc, vobs, vgas, vstar, sigma, KAPPA, args.m)

        # MOND Υ★ already computed above:
        # ups_mond_best, chi2_mond_best  (from fit_mond_pergalaxy)

        ups_lcdm_best, chi2_lcdm_ups = fit_lcdm_upsilon(r_kpc, vobs, vgas, vstar, sigma, m200_fixed, h0=args.H0)
    
        # =============================================================
        # Winners
        # =============================================================
        def pick_winner(c_wt, c_mond, c_lcdm):
            vals = {"WT": c_wt, "MOND": c_mond, "LCDM": c_lcdm}
            finite = {k: v for k, v in vals.items() if np.isfinite(v)}
            if not finite:
                return "---"
            best_key = min(finite, key=finite.get)
            return best_key

        win_0 = pick_winner(chi2_wt_kappa, chi2_mond_fixed, chi2_lcdm_fixed)
        win_1 = pick_winner(chi2_wt_best, chi2_mond_best, chi2_lcdm_best)
        win_2 = pick_winner(chi2_wt_ups, chi2_mond_best, chi2_lcdm_ups)
        rows.append([
            name, N,
            fmt(a0_best), fmt(A_morph), fmt(chi2_wt_best), fmt(chi2_wt_kappa),
            fmt(ups_wt_best), fmt(chi2_wt_ups),
            fmt(a0_2d), fmt(ups_2d), fmt(chi2_wt_2d),
            fmt(ups_mond_best), fmt(chi2_mond_fixed), fmt(chi2_mond_best),
            fmt(log10_m200_fixed), fmt(log10_m200_best),
            fmt(chi2_lcdm_fixed), fmt(chi2_lcdm_best),
            fmt(ups_lcdm_best), fmt(chi2_lcdm_ups),
            win_0, win_1,win_2,
            fmt(vmax_kms),
        ])

    # -- Write CSV --
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow(row)

    print(f"\n[INFO] written: {out_path}")

    # =================================================================
    # Summary statistics
    # =================================================================
    import pandas as pd
    try:
        df = pd.read_csv(out_path)
    except ImportError:
        print("[WARN] pandas not available; skipping summary")
        return

    n_gal = len(df)
    amorph = df["A_morph"].dropna()

    print(f"\n{'='*65}")
    print(f"  SUMMARY — {n_gal} galaxies,  m = {args.m}")
    print(f"{'='*65}")

    print(f"\nA_morph distribution:")
    print(f"  min={amorph.min():.3f}  p25={amorph.quantile(0.25):.3f}  "
          f"median={amorph.median():.3f}  p75={amorph.quantile(0.75):.3f}  "
          f"max={amorph.max():.3f}")
    print(f"  κ reference = {KAPPA:.4f}")

    # --- Tier 0: zero free parameters ---
    print(f"\n--- TIER 0: Zero free parameters ---")
    for label, col in [("WT(κ)", "chi2_WT_kappa"),
                       ("MOND(fixed)", "chi2_MOND_fixed"),
                       ("ΛCDM(fixed)", "chi2_LCDM_fixed")]:
        vals = df[col].dropna()
        wins = (df["win_0param"] == label.split("(")[0].replace("κ","").strip()
                if "WT" not in label
                else df["win_0param"] == "WT").sum()
        # Simpler: count from the column directly
        print(f"  {label:16s}  median χ² = {vals.median():7.2f}  "
              f"mean χ² = {vals.mean():7.2f}")

    w0_wt   = (df["win_0param"] == "WT").sum()
    w0_mond = (df["win_0param"] == "MOND").sum()
    w0_lcdm = (df["win_0param"] == "LCDM").sum()
    print(f"  Wins:  WT={w0_wt}  MOND={w0_mond}  ΛCDM={w0_lcdm}")

    # WT simultaneous wins (beats BOTH)
    simul_0 = ((df["chi2_WT_kappa"] < df["chi2_MOND_fixed"]) &
               (df["chi2_WT_kappa"] < df["chi2_LCDM_fixed"])).sum()
    print(f"  WT simultaneous wins (vs both): {simul_0}/{n_gal}")

    # --- Tier 1: one free parameter per galaxy ---
    print(f"\n--- TIER 1: One parameter per galaxy ---")
    for label, col in [("WT(α₀)", "chi2_WT_best"),
                       ("MOND(Υ★)", "chi2_MOND_best"),
                       ("ΛCDM(M200)", "chi2_LCDM_best")]:
        vals = df[col].dropna()
        print(f"  {label:16s}  median χ² = {vals.median():7.2f}  "
              f"mean χ² = {vals.mean():7.2f}")

    w1_wt   = (df["win_1param"] == "WT").sum()
    w1_mond = (df["win_1param"] == "MOND").sum()
    w1_lcdm = (df["win_1param"] == "LCDM").sum()
    print(f"  Wins:  WT={w1_wt}  MOND={w1_mond}  ΛCDM={w1_lcdm}")

    simul_1 = ((df["chi2_WT_best"] < df["chi2_MOND_best"]) &
               (df["chi2_WT_best"] < df["chi2_LCDM_best"])).sum()
    print(f"  WT simultaneous wins (vs both): {simul_1}/{n_gal}")

    # --- Tier 1b: same parameter (Υ★) for all three ---
    print(f"\n--- TIER 1b: All three fit Υ★ ---")
    for label, col in [("WT(κ+Υ★)", "chi2_WT_ups"),
                       ("MOND(Υ★)", "chi2_MOND_best"),
                       ("ΛCDM(Vmax+Υ★)", "chi2_LCDM_ups")]:
        vals = df[col].dropna()
        print(f"  {label:16s}  median χ² = {vals.median():7.2f}  "
              f"mean χ² = {vals.mean():7.2f}")

    w1b_wt   = (df["win_1param_ups"] == "WT").sum()
    w1b_mond = (df["win_1param_ups"] == "MOND").sum()
    w1b_lcdm = (df["win_1param_ups"] == "LCDM").sum()
    print(f"  Wins:  WT={w1b_wt}  MOND={w1b_mond}  ΛCDM={w1b_lcdm}")
    
    # --- Diagnostic: WT 2D (α₀ + Υ★) ---
    print(f"\n--- DIAGNOSTIC: WT 2-parameter (α₀ + Υ★) ---")
    vals = df["chi2_WT_2d"].dropna()
    print(f"  median χ² = {vals.median():.2f}  mean χ² = {vals.mean():.2f}")
    print(f"  Catastrophic (χ² > 100): {(vals > 100).sum()}")
    ups_2d = df["ups_WT_2d"].dropna()
    print(f"  Υ★: median={ups_2d.median():.3f}  p25={ups_2d.quantile(0.25):.3f}  p75={ups_2d.quantile(0.75):.3f}")
    a0_2d = df["alpha0_2d"].dropna()
    print(f"  α₀: median={a0_2d.median():.3f}  p25={a0_2d.quantile(0.25):.3f}  p75={a0_2d.quantile(0.75):.3f}")
    
    # --- Υ★ distribution ---
    ups = df["ups_MOND_best"].dropna()
    print(f"\nMOND Υ★ distribution:")
    print(f"  min={ups.min():.3f}  p25={ups.quantile(0.25):.3f}  "
          f"median={ups.median():.3f}  p75={ups.quantile(0.75):.3f}  "
          f"max={ups.max():.3f}")

    # --- M200 distribution ---
    lm = df["log10_M200_best"].dropna()
    print(f"\nΛCDM log₁₀(M200) distribution (fitted):")
    print(f"  min={lm.min():.2f}  p25={lm.quantile(0.25):.2f}  "
          f"median={lm.median():.2f}  p75={lm.quantile(0.75):.2f}  "
          f"max={lm.max():.2f}")

    # --- Catastrophic failures (chi2 > 100) ---
    print(f"\nCatastrophic failures (χ² > 100):")
    print(f"  WT(α₀):    {(df['chi2_WT_best'] > 100).sum()}")
    print(f"  MOND(Υ★):  {(df['chi2_MOND_best'] > 100).sum()}")
    print(f"  ΛCDM(M200):{(df['chi2_LCDM_best'] > 100).sum()}")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
