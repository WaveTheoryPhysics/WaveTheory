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

'''
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
'''
def v_mond(r_kpc: np.ndarray, vgas_kms: np.ndarray, vstar_kms: np.ndarray) -> np.ndarray:
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg = np.asarray(vgas_kms, dtype=float) * 1.0e3
    vs = np.asarray(vstar_kms, dtype=float) * 1.0e3
    safe_r = np.where(r_m > 0.0, r_m, np.nan)
    a_bar = np.nan_to_num(vg**2 / safe_r, nan=0.0) + np.nan_to_num(vs**2 / safe_r, nan=0.0)
    # McGaugh interpolation: nu(y) = 1/(1 - exp(-sqrt(y))), y = a_bar/a0
    y = a_bar / A0_MOND
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.clip(y, 1e-30, None))))
    a_mond = a_bar * nu
    return np.sqrt(np.clip(a_mond, 0.0, np.inf) * r_m) / 1.0e3

# =====================================================================
# 4. ΛCDM NFW benchmark  (identical to gogd.py)
# =====================================================================
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

    print(f"Per-galaxy α₀ fit: {len(sample)} galaxies")
    print(f"Grid: {gmin:.3f} → {gmax:.3f}, step {gstep:.3f}  ({len(alpha_grid)} points)")
    print(f"m = {args.m:.3f},  κ = {KAPPA:.6f}")
    print()

    out_path = pathlib.Path(args.chi2csv)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "name", "N",
            "alpha0_best", "A_morph",
            "chi2_WT_best", "chi2_WT_kappa",
            "chi2_MOND", "chi2_LCDM",
            "WT_wins","WT_wins_kappa", "MOND_wins", "LCDM_wins",
            "delta_chi2_WT_MOND", "delta_chi2_WT_LCDM",
            "Vmax_kms", "Rout_kpc",
            "f_gas_mean", "f_gas_outer",
            "M200_Msun", "log10_M200",
            "a_char", "log10_a_char",
            "m_used",
        ])

        for i, fp in enumerate(sample, 1):
            name = fp.stem.replace("_rotmod", "")
            #print(f"  [{i:3d}/{len(sample)}]  {name}", end="  ", flush=True)

            r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)
            N = len(vobs)

            # Per-galaxy best α₀
            a0_best, chi2_best = best_alpha0_for_galaxy(fp, alpha_grid, args.m)
            A_morph = a0_best / KAPPA

            # χ² at universal κ (for comparison)
            chi2_kappa = rchi2(vobs, v_wt(r_kpc, vgas, vstar, KAPPA, args.m), verr)

            # MOND
            chi2_mond = rchi2(vobs, v_mond(r_kpc, vgas, vstar), verr)

            # ΛCDM
            try:
                m200 = estimate_M200(fp, args.H0)
                v_halo = nfw_velocity(r_kpc, m200, args.H0)
                v_lcdm = np.sqrt(np.clip(vstar**2 + vgas**2 + v_halo**2,
                                         0.0, np.inf))
                chi2_lcdm = rchi2(vobs, v_lcdm, verr)
            except Exception:
                chi2_lcdm = float("nan")

            # Winners (at best WT fit)
            best = min(chi2_best,
                       chi2_mond  if np.isfinite(chi2_mond)  else 1e9,
                       chi2_lcdm if np.isfinite(chi2_lcdm) else 1e9)
            best_kappa = min(chi2_kappa,
                       chi2_mond  if np.isfinite(chi2_mond)  else 1e9,
                       chi2_lcdm if np.isfinite(chi2_lcdm) else 1e9)
            wt_w   = 1 if chi2_best == best else 0
            wt_w_kappa = 1 if chi2_kappa == best_kappa else 0
            mond_w = 1 if (np.isfinite(chi2_mond)  and chi2_mond  == best) else 0
            lcdm_w = 1 if (np.isfinite(chi2_lcdm) and chi2_lcdm == best) else 0

            obs = galaxy_observables(fp, args.H0)

            #print(f"α₀={a0_best:.3f}  A_morph={A_morph:.3f}  "
            #      f"χ²_WT={chi2_best:.2f}  χ²_MOND={chi2_mond:.2f}  "
            #      f"χ²_LCDM={fmt(chi2_lcdm)}  "
            #      f"{'WT✓' if wt_w else ('MOND✓' if mond_w else 'LCDM✓')}")

            w.writerow([
                name, N,
                fmt(a0_best), fmt(A_morph),
                fmt(chi2_best), fmt(chi2_kappa),
                fmt(chi2_mond), fmt(chi2_lcdm),
                wt_w, wt_w_kappa, mond_w, lcdm_w,
                fmt(chi2_best - chi2_mond)  if np.isfinite(chi2_mond)  else "",
                fmt(chi2_best - chi2_lcdm) if np.isfinite(chi2_lcdm) else "",
                fmt(obs["vmax"]), fmt(obs["rout"]),
                fmt(obs["f_mean"]), fmt(obs["f_outer"]),
                fmt(obs["m200"]), fmt(obs["log10_m200"]),
                fmt(obs["a_char"]), fmt(obs["log10_a"]),
                fmt(args.m),
            ])

    print()
    print(f"[INFO] written: {out_path}")

    # Quick summary
    import pandas as pd
    try:
        df = pd.read_csv(out_path)
        amorph = df["A_morph"].dropna()
        print(f"\nA_morph distribution across {len(amorph)} galaxies:")
        print(f"  min={amorph.min():.3f}  p25={amorph.quantile(0.25):.3f}  "
              f"median={amorph.median():.3f}  p75={amorph.quantile(0.75):.3f}  "
              f"max={amorph.max():.3f}")
        print(f"  κ reference = {KAPPA:.4f}  (A_morph=1.000)")
        wt_wins = df["WT_wins"].sum()
        wt_wins_kappa = df["WT_wins_kappa"].sum()
        print(f"\nWT wins (at best per-galaxy α₀): {wt_wins}/{len(df)}")
        print(f"\nWT_kappa wins (at κ reference): {wt_wins_kappa}/{len(df)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
