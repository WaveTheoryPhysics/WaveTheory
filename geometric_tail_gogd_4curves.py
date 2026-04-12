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
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

"""
Geometry of Galaxy Dynamics rotation-curve analysis
==================================================

Variant with four comparison curves per panel:
    - WT pure κ
    - WT adjusted α
    - MOND
    - ΛCDM (NFW)

The pure-κ branch keeps α0 fixed to
    κ = 1 + 1/π²,
while the adjusted-α branch uses either a user-supplied α0 or a globally fitted α0.

Color convention used in the plots:
    WT pure κ      -> orange solid
    WT adjusted α  -> red solid
    MOND           -> green dashed
    ΛCDM (NFW)     -> red dotted
    data           -> blue points
"""

# =====================================================================
# 1. Global constants
# =====================================================================
PI = math.pi
C_SI = 299_792_458.0
G_SI = 6.67430e-11
A0_MOND = 1.2e-10
MSUN = 1.98847e30
MPC_M = 3.085677581e22
KPC2M = 3.085677581491367e19
G_KPC = 4.30091e-6

R_UNIVERSE_M = 4.286332662e26
A_STAR = C_SI**2 / (PI * R_UNIVERSE_M)
KAPPA_ALPHA0 = 1.0 + 1.0 / (PI**2)

# Fixed visual mapping
COLOR_DATA = "tab:blue"
COLOR_WT_KAPPA = "tab:orange"
COLOR_WT_ALPHA = "red"
COLOR_MOND = "tab:green"
COLOR_LCDM = "tab:red"


# =====================================================================
# 2. Robust data loading
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
    arr = loadtxt_safe(path)
    r_kpc, vobs, verr = arr[:, 0], arr[:, 1], arr[:, 2]
    vgas, vdisk, vbul = arr[:, 3], arr[:, 4], arr[:, 5]
    vstar = np.sqrt(vdisk**2 + vbul**2)
    return r_kpc, vobs, verr, vgas, vstar


# =====================================================================
# 3. Optional ΛCDM helper functions
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
    data = loadtxt_safe(fp)
    vmax_kms = float(np.nanmax(data[:, 1]))
    v_m_s = vmax_kms * 1.0e3
    h0_s = (h0_km_s_mpc * 1000.0) / MPC_M
    m_kg = (v_m_s**3) / (10.0 * G_SI * h0_s)
    return m_kg / MSUN


# =====================================================================
# 4. GOGD WT predictor
# =====================================================================
def v_wt_gogd(
    r_kpc: np.ndarray,
    vgas_kms: np.ndarray,
    vstar_kms: np.ndarray,
    alpha0: float,
    m_exp: float,
    r_universe_m: float = R_UNIVERSE_M,
) -> np.ndarray:
    r_m = np.asarray(r_kpc, dtype=float) * KPC2M
    vg_ms = np.asarray(vgas_kms, dtype=float) * 1.0e3
    vs_ms = np.asarray(vstar_kms, dtype=float) * 1.0e3

    safe_r = np.where(r_m > 0.0, r_m, np.nan)
    a_g = np.where(r_m > 0.0, vg_ms**2 / safe_r, 0.0)
    a_s = np.where(r_m > 0.0, vs_ms**2 / safe_r, 0.0)
    a_b = np.nan_to_num(a_g + a_s, nan=0.0, posinf=0.0, neginf=0.0)

    a_star = (C_SI**2) / (PI * r_universe_m)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = np.where(a_star > 0.0, a_b / a_star, 0.0)
        sigma = 1.0 / (1.0 + np.power(u, m_exp))
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    a_shell = alpha0 * np.sqrt(np.clip(a_b * a_star, 0.0, np.inf))
    a_eff = (1.0 - sigma) * a_b + sigma * a_shell
    a_eff = np.clip(a_eff, 0.0, np.inf)

    v_wt_ms = np.sqrt(a_eff * np.clip(r_m, 0.0, np.inf))
    return v_wt_ms / 1.0e3


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


def galaxy_metrics(
    fp: pathlib.Path,
    args: argparse.Namespace,
    alpha0_adjusted: float,
) -> Tuple[str, int, float, float, float, float, float]:
    r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)

    v_wt_kappa = v_wt_gogd(r_kpc, vgas, vstar, args.alpha_kappa, args.m)
    v_wt_alpha = v_wt_gogd(r_kpc, vgas, vstar, alpha0_adjusted, args.m)
    chi_wt_kappa = reduced_chi2(vobs, v_wt_kappa, verr)
    chi_wt_alpha = reduced_chi2(vobs, v_wt_alpha, verr)

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
    return name, len(vobs), chi_wt_kappa, chi_wt_alpha, chi_mnd, chi_lcdm, alpha0_adjusted


# =====================================================================
# 6. CLI / driver
# =====================================================================
def vmax_fp(fp: pathlib.Path) -> float:
    return float(np.nanmax(loadtxt_safe(fp)[:, 1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", default="data/Rotmod_LTG")
    p.add_argument("--grid", default="0,2.0,0.002", help="Adjusted α0 grid for fit mode: min,max,step")
    p.add_argument(
        "--alpha",
        type=float,
        default=KAPPA_ALPHA0,
        help="Adjusted α0 for the WT adjusted-α branch. Use a negative value with --alpha-mode fit to grid-fit it.",
    )
    p.add_argument(
        "--alpha-kappa",
        type=float,
        default=KAPPA_ALPHA0,
        help="Fixed α0 used by the WT pure-κ branch. Default: κ = 1 + 1/π².",
    )
    p.add_argument("--alpha-mode", choices=["const", "fit"], default="const")
    p.add_argument("--m", type=float, default=1.0)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=float("inf"))
    p.add_argument("--plotN", type=int, default=0)
    p.add_argument("--page", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fname", default="rotation_samples_gogd_4curves")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--chi2csv", default="chi2_summary_GeometryOfGalaxyDynamics_4curves.csv")
    p.add_argument("--names", default="")
    p.add_argument("--mond", type=int, default=1)
    p.add_argument("--lcdm", type=int, default=1)
    p.add_argument("--H0", type=float, default=70.0)
    return p


def fmt(x: float) -> str:
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
    print(f"Using κ branch α0 = {args.alpha_kappa:.6f}")

    if args.alpha >= 0.0:
        alpha0_adjusted = args.alpha
        redchi = chi2_global(alpha0_adjusted, args.m, sample)
        print(f"Using adjusted α0 = {alpha0_adjusted:.6f}   (χ²_red={redchi:.3f})")
    elif args.alpha_mode == "fit":
        gmin, gmax, gstep = map(float, args.grid.split(","))
        alpha_grid = np.arange(gmin, gmax + 0.5 * gstep, gstep)
        chis = [chi2_global(a, args.m, sample) for a in alpha_grid]
        best_i = int(np.argmin(chis))
        alpha0_adjusted = float(alpha_grid[best_i])
        chi_best = float(chis[best_i])
        print(f"Best adjusted α0 = {alpha0_adjusted:.6f}   (χ²_red={chi_best:.3f})")

        plt.figure(figsize=(6, 4))
        plt.plot(alpha_grid, chis, "o-")
        plt.axvline(alpha0_adjusted, color=COLOR_WT_ALPHA, ls="--", label=f"best α0={alpha0_adjusted:.4f}")
        plt.xlabel("adjusted α0")
        plt.ylabel("Reduced χ²")
        plt.title(f"GOGD adjusted α fit, m={args.m:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("alpha0_fit_diagnostic_gogd_4curves.png", dpi=140)
    else:
        sys.exit("Set either a non-negative --alpha or use --alpha-mode fit with a negative --alpha.")

    if args.chi2csv:
        name_p = "".join(args.names)
        out_path = pathlib.Path(f"{args.fname}_{name_p}.csv")
        with out_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "name", "N",
                "chi2_WT_kappa", "chi2_WT_adjusted_alpha", "chi2_MOND", "chi2_LCDM",
                "alpha0_kappa", "alpha0_adjusted",
                "WT_adjusted_best", "WT_kappa_best", "MOND_best", "LCDM_best",
            ])
            for fp in sample:
                name, npts, chi_kappa, chi_alpha, chi_mnd, chi_lcdm, alpha_used = galaxy_metrics(fp, args, alpha0_adjusted)
                finite_pool = {
                    "WT adjusted": chi_alpha,
                    "WT κ": chi_kappa,
                    "MOND": chi_mnd,
                    "ΛCDM": chi_lcdm,
                }
                finite_pool = {k: v for k, v in finite_pool.items() if np.isfinite(v)}
                best_name = min(finite_pool, key=finite_pool.get) if finite_pool else ""
                w.writerow([
                    name, npts,
                    f"{chi_kappa:.6g}",
                    f"{chi_alpha:.6g}",
                    fmt(chi_mnd),
                    fmt(chi_lcdm),
                    f"{args.alpha_kappa:.6g}",
                    f"{alpha_used:.6g}",
                    1 if best_name == "WT adjusted" else 0,
                    1 if best_name == "WT κ" else 0,
                    1 if best_name == "MOND" else 0,
                    1 if best_name == "ΛCDM" else 0,
                ])
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
        name, _, chi_kappa, chi_alpha, chi_mnd, chi_lcdm, alpha_used = galaxy_metrics(fp, args, alpha0_adjusted)
        r_kpc, vobs, verr, vgas, vstar = load_rotmod(fp)

        v_wt_kappa = v_wt_gogd(r_kpc, vgas, vstar, args.alpha_kappa, args.m)
        v_wt_alpha = v_wt_gogd(r_kpc, vgas, vstar, alpha_used, args.m)
        v_mnd = v_mond(r_kpc, vgas, vstar) if args.mond else None

        if args.lcdm:
            m200 = estimate_M200_from_vmax(fp, args.H0)
            v_halo = nfw_halo_velocity(r_kpc, m200, args.H0)
            v_lcdm = np.sqrt(np.clip(vstar**2 + vgas**2 + v_halo**2, 0.0, np.inf))
        else:
            v_lcdm = None

        ax.errorbar(r_kpc, vobs, yerr=verr, fmt="o", ms=3, color=COLOR_DATA, label="", alpha=0.8)
        
        ax.plot(r_kpc, v_wt_kappa, lw=1.4, color=COLOR_WT_KAPPA, label="")
        ax.plot(r_kpc, v_wt_alpha, lw=1.4, color=COLOR_WT_ALPHA, label="")
        if v_mnd is not None:
            ax.plot(r_kpc, v_mnd, ls="--", lw=1.2, color=COLOR_MOND, label="")
        if v_lcdm is not None:
            ax.plot(r_kpc, v_lcdm, ls=":", lw=1.6, color=COLOR_LCDM, label=rf"")
            
        #ax.plot(r_kpc, v_wt_kappa, lw=1.4, color=COLOR_WT_KAPPA, label=f"WT pure κ, χ²={chi_kappa:.2f}")
        #ax.plot(r_kpc, v_wt_alpha, lw=1.4, color=COLOR_WT_ALPHA, label=f"WT adjusted α, χ²={chi_alpha:.2f}")
        #if v_mnd is not None:
        #    ax.plot(r_kpc, v_mnd, ls="--", lw=1.2, color=COLOR_MOND, label=f"MOND, χ²={chi_mnd:.2f}")
        #if v_lcdm is not None:
        #    ax.plot(r_kpc, v_lcdm, ls=":", lw=1.6, color=COLOR_LCDM, label=rf"$\Lambda$CDM (NFW), χ²={chi_lcdm:.2f}")

        ax.set_title(name)
        ax.set_xlabel("R [kpc]")
        if ax is axs[0]:
            ax.set_ylabel("V [km/s]")
        ax.grid(alpha=0.3)

        #chi_text = [
        #    f"χ²_κ={chi_kappa:.2f}",
        #    f"χ²_α={chi_alpha:.2f}",
        #]
        #if np.isfinite(chi_mnd):
        #    chi_text.append(f"χ²_MOND={chi_mnd:.2f}")
        #if np.isfinite(chi_lcdm):
        #    chi_text.append(f"χ²_LCDM={chi_lcdm:.2f}")

        #ax.text(0.97, 0.05, "\n".join(chi_text), transform=ax.transAxes,
        #        ha="right", va="bottom", fontsize=7)

    for ax in axs[n_gal:]:
        ax.axis("off")

    axs[0].legend(fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(outname, dpi=args.dpi)
    print(f"[INFO] {outname} saved.")


if __name__ == "__main__":
    main()
