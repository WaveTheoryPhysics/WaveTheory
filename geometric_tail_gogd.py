"""
Geometry of Galaxy Dynamics rotation-curve analysis
==================================================

This script keeps the data-loading, MOND / ΛCDM comparison, χ² summaries,
and panel plotting workflow of the older `geometric_tail.py`, but replaces the
Wave Theory predictor with the two-channel model from:

    Geometry of Galactic Dynamics
    Local Curvature, Shell Transport, and the Emergence of Flat Rotation Curves
    Daniel Banasik, March 26, 2026

Implemented WT model
--------------------
The Wave Theory prediction is now based on the paper's master acceleration law:

    a_eff(r) = a_b(r)
             + μ(r) α0 L(χ) [4π G M_b(r) / r]

with

    a_b(r) = G M_b(r) / r^2
    μ(r)   = 1 / (1 + (a_b/a_*)^m)
    a_*    = c^2 / (π R_universe)
    χ      = r / R_universe
    L(χ)   = A_boundary(χ) / V_cap(χ)

and

    V_cap(χ) = 2 π R^3 [χ - 1/2 sin(2χ)]
    A_boundary(χ) = 4 π R^2 sin^2(χ)

Important modelling note
------------------------
The new paper treats α0 and m as empirical calibration parameters. The older
WT gas-coupling mass law α(M) and the velocity-weighted β boost are therefore
removed from the WT prediction here.

Typical runs
~~~~~~~~~~~~
    # 1) Use a fixed morphology factor α0 and transition exponent m
    python geometric_tail_gogd.py \
        --alpha 0.03 \
        --m 2.0 \
        --plotN 6 \
        --fname gogd_fixed

    # 2) Fit a single global α0 on a grid for the chosen sample
    python geometric_tail_gogd.py \
        --alpha -1 \
        --alpha-mode fit \
        --grid 0,0.2,0.002 \
        --m 2.0 \
        --plotN 6 \
        --fname gogd_fit

    # 3) Restrict to a vmax range and compare WT vs MOND vs ΛCDM
    python geometric_tail_gogd.py \
        --alpha 0.03 \
        --m 2.0 \
        --vmin 80 --vmax 220 \
        --plotN 9 \
        --lcdm 1 \
        --fname gogd_intermediate
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
import sys
import tempfile
import urllib.request
import zipfile
from io import StringIO
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. Global constants
# =====================================================================
PI = math.pi
C_SI = 299_792_458.0                    # m/s
G_SI = 6.67430e-11                      # m^3 kg^-1 s^-2
A0_MOND = 1.2e-10                       # m/s^2
MSUN = 1.98847e30                       # kg
MPC_M = 3.085677581e22                  # m
KPC2M = 3.085677581491367e19            # m
G_KPC = 4.30091e-6                      # kpc (km/s)^2 / Msun

# Wave Theory / Geometry of Galactic Dynamics cosmic radius
R_UNIVERSE_M = 4.286332662e26           # m
A_STAR = C_SI**2 / (PI * R_UNIVERSE_M)  # c^2 / (πR)


# =====================================================================
# 2. Robust data loading
# =====================================================================
def loadtxt_safe(fp: pathlib.Path) -> np.ndarray:
    """Robust text loader for SPARC / THINGS rotmod tables."""
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

    with open(fp, "rb") as fh:
        raw = fh.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            txt = raw.decode(enc, errors="ignore")
            txt = txt.replace(" .", " nan").replace(". ", "nan ")
            data = np.genfromtxt(StringIO(txt), comments="#")
            if data.ndim == 1:
                data = data.reshape(-1, data.shape[0])
            return data
        except Exception:
            continue

    raise RuntimeError(f"Could not load file {fp} with any fallback.")


def load_rotmod(path: pathlib.Path) -> Tuple[np.ndarray, ...]:
    """Load one *_rotmod.dat file.

    Returns radius R_k [kpc], observed velocity, uncertainty, gas velocity,
    and combined stellar velocity.
    """
    arr = loadtxt_safe(path)
    r_kpc, vobs, verr = arr[:, 0], arr[:, 1], arr[:, 2]
    vgas, vdisk, vbul = arr[:, 3], arr[:, 4], arr[:, 5]
    vstar = np.sqrt(vdisk**2 + vbul**2)
    return r_kpc, vobs, verr, vgas, vstar


# =====================================================================
# 3. Optional ΛCDM helper functions (unchanged benchmarking scaffold)
# =====================================================================
def concentration_dutton_maccio(m200_msun: float) -> float:
    m12 = m200_msun / 1.0e12
    a = 0.537
    b = -0.097
    log10_c = a + b * math.log10(m12)
    return 10.0 ** log10_c


def r200_from_m200(m200_msun: float, h0_km_s_mpc: float = 70.0) -> float:
    m12 = m200_msun / 1.0e12
    return 206.0 * (m12 ** (1.0 / 3.0)) * ((h0_km_s_mpc / 70.0) ** (-2.0 / 3.0))


def nfw_halo_velocity(r_kpc: np.ndarray, m200_msun: float, h0_km_s_mpc: float = 70.0) -> np.ndarray:
    r_arr = np.asarray(r_kpc, dtype=float)
    if m200_msun <= 0.0:
        return np.zeros_like(r_arr)

    c200 = concentration_dutton_maccio(m200_msun)
    r200 = r200_from_m200(m200_msun, h0_km_s_mpc)
    rs = r200 / c200

    def f(x: np.ndarray) -> np.ndarray:
        return np.log(1.0 + x) - x / (1.0 + x)

    x = r_arr / rs
    x_safe = np.where(x > 0.0, x, 1.0e-12)
    m_enclosed = m200_msun * f(x_safe) / f(np.array([c200]))[0]
    v2 = np.where(r_arr > 0.0, G_KPC * m_enclosed / r_arr, 0.0)
    return np.sqrt(np.clip(v2, 0.0, np.inf))


def estimate_M200_from_vmax(fp: pathlib.Path, h0_km_s_mpc: float) -> float:
    """Conventional M200 estimate used only for the optional NFW benchmark."""
    data = loadtxt_safe(fp)
    vmax_kms = float(np.nanmax(data[:, 1]))
    v_m_s = vmax_kms * 1.0e3
    h0_s = (h0_km_s_mpc * 1000.0) / MPC_M
    m_kg = (v_m_s**3) / (10.0 * G_SI * h0_s)
    return m_kg / MSUN


# =====================================================================
# 4. Geometry of Galactic Dynamics WT predictor
# =====================================================================

def v_wt_gogd(
    r_kpc: np.ndarray,
    vgas_kms: np.ndarray,
    vstar_kms: np.ndarray,
    alpha0: float,
    m_exp: float,
    r_universe_m: float = R_UNIVERSE_M,
) -> np.ndarray:
    """
    WT circular velocity from the newest GOGD master formula.

    New paper formula:
        a_eff(r) = (1 - sigma(r)) * a_b(r)
                 + sigma(r) * alpha0 * sqrt(a_b(r) * a_star)

        sigma(r) = 1 / (1 + (a_b/a_star)^m)
        a_star   = c^2 / (pi * R_universe)

    where
        a_b(r) = a_g(r) + a_s(r)
        a_g    = V_gas^2 / r
        a_s    = V_star^2 / r

    Inputs
    ------
    r_kpc      : radius array [kpc]
    vgas_kms   : gas circular-speed contribution [km/s]
    vstar_kms  : stellar circular-speed contribution [km/s]
    alpha0     : morphology-dependent projection efficiency
    m_exp      : transition sharpness exponent m
    r_universe_m : cosmic radius R [m]

    Returns
    -------
    v_wt_kms : predicted total circular velocity [km/s]
    """
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg_ms = np.asarray(vgas_kms, dtype=float) * 1.0e3
    vs_ms = np.asarray(vstar_kms, dtype=float) * 1.0e3

    # Safe radius handling
    safe_r = np.where(r_m > 0.0, r_m, np.nan)

    # Local baryonic acceleration pieces
    a_g = np.where(r_m > 0.0, vg_ms**2 / safe_r, 0.0)
    a_s = np.where(r_m > 0.0, vs_ms**2 / safe_r, 0.0)
    a_b = np.nan_to_num(a_g + a_s, nan=0.0, posinf=0.0, neginf=0.0)

    # Cosmological curvature acceleration a_* = c^2 / (pi R)
    a_star = (C_SI ** 2) / (PI * r_universe_m)

    # Crossfade weight sigma(r) = 1 / (1 + (a_b / a_*)^m)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = np.where(a_star > 0.0, a_b / a_star, 0.0)
        sigma = 1.0 / (1.0 + np.power(u, m_exp))

    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # 2D shell channel
    a_shell = alpha0 * np.sqrt(np.clip(a_b * a_star, 0.0, np.inf))

    # Master acceleration formula
    a_eff = (1.0 - sigma) * a_b + sigma * a_shell
    a_eff = np.clip(a_eff, 0.0, np.inf)

    # Circular speed: v = sqrt(a r)
    v_wt_ms = np.sqrt(a_eff * np.clip(r_m, 0.0, np.inf))
    return v_wt_ms / 1.0e3


def v_mond(r_kpc: np.ndarray, vgas_kms: np.ndarray, vstar_kms: np.ndarray) -> np.ndarray:
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg = np.asarray(vgas_kms, dtype=float) * 1.0e3
    vs = np.asarray(vstar_kms, dtype=float) * 1.0e3
    safe_r = np.where(r_m > 0.0, r_m, np.nan)
    a_n = np.nan_to_num(vg**2 / safe_r, nan=0.0) + np.nan_to_num(vs**2 / safe_r, nan=0.0)
    a = np.where(a_n < A0_MOND, np.sqrt(a_n * A0_MOND), a_n)
    return np.sqrt(np.clip(a, 0.0, np.inf) * r_m) / 1.0e3


# =====================================================================
# 5. χ² helpers
# =====================================================================
def reduced_chi2(v_obs: np.ndarray, v_pred: np.ndarray, v_err: np.ndarray) -> float:
    n = len(v_obs)
    if n == 0:
        return float("nan")
    return float(np.sum(((v_obs - v_pred) / v_err) ** 2) / n)


def chi2_global(alpha0: float, m_exp: float, files: List[pathlib.Path]) -> float:
    chi = 0.0
    dof = 0.0
    for fp in files:
        r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)
        v_wt = v_wt_gogd(r_kpc, vgas, vstar, alpha0, m_exp)
        chi += np.sum(((vobs - v_wt) / verr) ** 2)
        dof += len(vobs)
    return float(chi / dof) if dof else float("nan")


def galaxy_chi2(fp: pathlib.Path, args: argparse.Namespace, alpha0_best: float) -> Tuple[str, int, float, float, float, float]:
    r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)

    alpha0_local = args.alpha if args.alpha >= 0.0 else alpha0_best
    v_wt = v_wt_gogd(r_kpc, vgas, vstar, alpha0_local, args.m)
    chi_wt = reduced_chi2(vobs, v_wt, verr)

    if args.mond:
        v_mnd = v_mond(r_kpc, vgas, vstar)
        chi_mnd = reduced_chi2(vobs, v_mnd, verr)
    else:
        chi_mnd = float("nan")

    if args.lcdm:
        m200 = estimate_M200_from_vmax(fp, args.H0)
        v_halo = nfw_halo_velocity(r_kpc, m200, args.H0)
        v_lcdm = np.sqrt(np.clip(vstar**2 + vgas**2 + v_halo**2, 0.0, np.inf))
        chi_lcdm = reduced_chi2(vobs, v_lcdm, verr)
    else:
        chi_lcdm = float("nan")

    name = fp.stem.replace("_rotmod", "")
    return name, len(vobs), chi_wt, chi_mnd, chi_lcdm, alpha0_local


# =====================================================================
# 6. CLI / analysis driver
# =====================================================================
def vmax_fp(fp: pathlib.Path) -> float:
    return float(np.nanmax(loadtxt_safe(fp)[:, 1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", default="data/Rotmod_LTG")
    p.add_argument("--grid", default="0,0.2,0.002", help="α0 grid for fit mode: min,max,step")
    p.add_argument("--alpha", type=float, default=1.10132118,
                   help="Fixed morphology factor α0. Use a negative value with --alpha-mode fit to grid-fit α0.")
    p.add_argument("--alpha-mode", choices=["const", "fit"], default="const",
                   help="How to choose α0 for the WT predictor.")
    p.add_argument("--m", type=float, default=1.0,
                   help="Transition sharpness exponent m in μ(r) = [1 + (a_b/a_*)^m]^-1.")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=float("inf"))
    p.add_argument("--plotN", type=int, default=0)
    p.add_argument("--page", type=int, default=-1,
                   help="0-based page index; if ≥0 use deterministic blocks of plotN instead of random sampling")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fname", default="rotation_samples_gogd")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--chi2csv", default="chi2_summary_GeometryOfGalaxyDynamics.csv")
    p.add_argument("--names", default="",
                   help="Comma-separated list of specific galaxy names (no random sampling / vmax cuts)")
    p.add_argument("--mond", type=int, default=1,
                   help="Compute and plot MOND curve (1=on, 0=off).")
    p.add_argument("--lcdm", type=int, default=1,
                   help="Compute and plot ΛCDM NFW curve (1=on, 0=off).")
    p.add_argument("--H0", type=float, default=70.0,
                   help="Hubble constant for the optional M200 estimate in the NFW benchmark (km/s/Mpc).")
    return p

def fmt(x):
    """Safely format floats for CSV."""
    return f"{x:.6g}" if (x is not None and np.isfinite(x)) else ""

def main() -> None:
    args = build_parser().parse_args()

    names_list = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else []

    dpath = pathlib.Path(args.datadir)
    if not dpath.exists():
        print("[INFO] downloading SPARC archive…")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        urllib.request.urlretrieve("https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip", tmp.name)
        zipfile.ZipFile(tmp.name).extractall(dpath)

    files = sorted(fp for fp in dpath.glob("*_rotmod.dat") if not fp.name.startswith("._"))
    if not files:
        sys.exit("No rotmod files found.")

    if names_list:
        sample: List[pathlib.Path] = []
        for name in names_list:
            fp = dpath / f"{name}_rotmod.dat"
            if fp.exists():
                sample.append(fp)
            else:
                print(f"[WARN] galaxy {name} not found at {fp}")
    else:
        sample = [fp for fp in files if args.vmin <= vmax_fp(fp) <= args.vmax]

    print(f"Sample: {len(sample)} galaxies   m={args.m:.3f}   mode={args.alpha_mode}\n")
    print(f"Using a_* = c^2/(πR) = {A_STAR:.6e} m/s² with R = {R_UNIVERSE_M:.9e} m")

    if args.alpha >= 0.0:
        alpha0_best = args.alpha
        redchi = chi2_global(alpha0_best, args.m, sample)
        best_label = f"const α0={alpha0_best:.4f} (χ²_red={redchi:.3f})"
        print(f"Using constant α0 = {alpha0_best:.6f}   (χ²_red={redchi:.3f})")
    elif args.alpha_mode == "fit":
        gmin, gmax, gstep = map(float, args.grid.split(","))
        alpha_grid = np.arange(gmin, gmax + 0.5 * gstep, gstep)
        chis = [chi2_global(a, args.m, sample) for a in alpha_grid]
        best_i = int(np.argmin(chis))
        alpha0_best = float(alpha_grid[best_i])
        chi_best = float(chis[best_i])
        best_label = f"fit α0={alpha0_best:.4f} (χ²_red={chi_best:.3f})"
        print(f"Best α0 = {alpha0_best:.6f}   (χ²_red={chi_best:.3f})")

        plt.figure(figsize=(6, 4))
        plt.plot(alpha_grid, chis, "o-")
        plt.axvline(alpha0_best, color="r", ls="--", label=f"best α0={alpha0_best:.4f}")
        plt.xlabel("α0")
        plt.ylabel("Reduced χ²")
        plt.title(f"Geometry of Galaxy Dynamics fit, m={args.m:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("alpha0_fit_diagnostic_gogd.png", dpi=140)
    else:
        sys.exit("Set either a non-negative --alpha or use --alpha-mode fit with a negative --alpha.")

    if args.chi2csv:
        out_path = pathlib.Path(args.chi2csv)
        total_wt_wins = 0
        total_mond_wins = 0
        total_lcdm_wins = 0
        with out_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "name", "N",
                "chi2_WT", "chi2_MOND", "chi2_LCDM",
                "alpha0_used", "m_used",
                "catastrophic_WT", "catastrophic_MOND", "catastrophic_LCDM",
                "MOND < WT","ΛCDM < WT", "ΛCDM < MOND",
                "MOND Win", "ΛCDM Win", "WT Win",
            ])
            for fp in sample:
                name, npts, chi_wt, chi_mnd, chi_lcdm, alpha0_local = galaxy_chi2(fp, args, alpha0_best)
                
                catastrophic_WT  = 1 if (chi_wt > 100) else 0
                catastrophic_mond = 1 if (np.isfinite(chi_mnd) and chi_mnd > 100) else 0
                catastrophic_lcdm = 1 if (np.isfinite(chi_lcdm) and chi_lcdm > 100) else 0

                mond_better_WT = 1 if (np.isfinite(chi_mnd) and chi_mnd < chi_wt) else 0
                lcdm_better_WT = 1 if (np.isfinite(chi_lcdm) and chi_lcdm < chi_wt) else 0
                lcdm_better_mond = 1 if (np.isfinite(chi_lcdm) and chi_lcdm < chi_mnd) else 0

                mond_win = 1 if (np.isfinite(chi_mnd) and chi_mnd < chi_wt and chi_mnd < chi_lcdm) else 0
                lcdm_win = 1 if (np.isfinite(chi_lcdm) and chi_lcdm < chi_wt and chi_lcdm < chi_mnd) else 0
                wt_win = 1 if (np.isfinite(chi_wt) and chi_wt < chi_lcdm and chi_wt < chi_mnd) else 0

                total_wt_wins += wt_win
                total_mond_wins += mond_win
                total_lcdm_wins += lcdm_win
                
                w.writerow([
                    name,
                    npts,
                    f"{chi_wt:.6g}",
                    f"{chi_mnd:.6g}" if np.isfinite(chi_mnd) else "",
                    f"{chi_lcdm:.6g}" if np.isfinite(chi_lcdm) else "",
                    f"{alpha0_local:.6g}",
                    f"{args.m:.6g}",
                    catastrophic_WT,
                    catastrophic_mond,
                    catastrophic_lcdm,
                    mond_better_WT,
                    lcdm_better_WT,
                    lcdm_better_mond,
                    fmt(mond_win),
                    fmt(lcdm_win),
                    fmt(wt_win)
                ])
        
        print(f"\n\nWT Wins: {total_wt_wins}\nMOND Wins: {total_mond_wins}\nLCDM Wins: {total_lcdm_wins}\n\n")
        print(f"[INFO] χ² CSV written: {out_path}")

    if names_list:
        picks = sample
        print(f"[INFO] explicitly plotting {len(picks)} named galaxies: {names_list}")
    elif args.plotN:
        if args.page is not None and args.page >= 0:
            start_idx = args.page * args.plotN
            end_idx = start_idx + args.plotN
            picks = sample[start_idx:end_idx]
            print(f"[INFO] plotting page {args.page}, indices [{start_idx}:{end_idx})")
        else:
            random.seed(args.seed)
            picks = random.sample(sample, min(args.plotN, len(sample)))
    else:
        picks = []

    if not picks:
        return

    n_gal = len(picks)
    rows = min(3, n_gal)
    cols = int(math.ceil(n_gal / rows))

    gal_names = [fp.stem.replace("_rotmod", "") for fp in picks]
    name_part = "_".join(gal_names)
    if len(name_part) > 120:
        name_part = "_".join(gal_names[:6]) + "_etc"
    outname = f"{args.fname}_{name_part}.png" if args.fname else f"{name_part}.png"

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharey=True)
    axs = np.atleast_1d(axs).flatten()

    for ax, fp in zip(axs, picks):
        name, _, chi_wt, chi_mnd, chi_lcdm, alpha0_local = galaxy_chi2(fp, args, alpha0_best)
        r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)

        v_wt = v_wt_gogd(r_kpc, vgas, vstar, alpha0_local, args.m)
        v_mnd = v_mond(r_kpc, vgas, vstar) if args.mond else None

        if args.lcdm:
            m200 = estimate_M200_from_vmax(fp, args.H0)
            v_halo = nfw_halo_velocity(r_kpc, m200, args.H0)
            v_lcdm = np.sqrt(np.clip(vstar**2 + vgas**2 + v_halo**2, 0.0, np.inf))
        else:
            v_lcdm = None

        ax.errorbar(r_kpc, vobs, yerr=verr, fmt="o", ms=3, label="data", alpha=0.8)
        ax.plot(r_kpc, v_wt, lw=1.4, label="WT two-channel")
        if v_mnd is not None:
            ax.plot(r_kpc, v_mnd, ls="--", lw=1.2, label="MOND")
        if v_lcdm is not None:
            ax.plot(r_kpc, v_lcdm, ls=":", lw=1.2, label=r"$\Lambda$CDM (NFW)")

        ax.set_title(name)
        ax.set_xlabel("R [kpc]")
        if ax is axs[0]:
            ax.set_ylabel("V [km/s]")
        ax.grid(alpha=0.3)

        chi_text = [f"χ²_WT={chi_wt:.2f}"]#, f"α0={alpha0_local:.4f}", f"m={args.m:.2f}"]
        if np.isfinite(chi_mnd):
            chi_text.append(f"χ²_MOND={chi_mnd:.2f}")
        if np.isfinite(chi_lcdm):
            chi_text.append(f"χ²_LCDM={chi_lcdm:.2f}")

        ax.text(0.97, 0.05, "\n".join(chi_text), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7)

    for ax in axs[n_gal:]:
        ax.axis("off")

    axs[0].legend(fontsize=7)
    #fig.suptitle(
    #    f"Rotation curves: Geometry of Galaxy Dynamics WT (m={args.m:.2f})"
    #    + (" vs MOND" if args.mond else "")
    #    + (" vs ΛCDM (NFW)" if args.lcdm else "")
    #    + f"\n{best_label}"
    #)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(outname, dpi=args.dpi)
    print(f"[INFO] {outname} saved.")


if __name__ == "__main__":
    main()
