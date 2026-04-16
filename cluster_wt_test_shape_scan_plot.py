#!/usr/bin/env python3
"""
Wave Theory cluster-scale test + global alpha/mass scan + panel plotting
========================================================================

Backwards-compatible single run:
    python3 cluster_wt_test_shape_scan_plot.py 1.0 1 4e11 2.2026

Grid scan mode:
    python3 cluster_wt_test_shape_scan_plot.py 1.0 1 4e11 2.2026 \
        --scan \
        --alpha-grid 0.5,5.0,0.1 \
        --mass-grid 1e11,2e15,25 \
        --top 12

Panel plot mode:
    python3 cluster_wt_test_shape_scan_plot.py 1.0 1 5.2e11 4.8 \
        --plot-panels 4 \
        --plot-select best \
        --plot-file cluster_match_4panel.pdf

Notes
-----
- alpha-grid is linear: start,stop,step
- mass-grid is logarithmic: start,stop,num_points
- "balanced score" is lower-is-better and combines:
      mean_shape_rms
    + |mean_delta_s|
    + 0.5 * |mean_offset|
  so shape dominates, slope is next, amplitude offset still matters.
"""

import argparse
import math
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vikhlinin2006_data import (
    get_cluster_list, CLUSTER_PROPERTIES,
    acceleration_profiles, KPC_M
)

# =====================================================================
# Constants
# =====================================================================
C_LIGHT = 2.998e8
R_HUBBLE = 4.286332662e26
A_STAR = C_LIGHT**2 / (np.pi * R_HUBBLE)
A0_MOND = 1.2e-10
MSUN = 1.98847e30
MW_MASS_MSUN = 5.3e11

# =====================================================================
# Core model
# =====================================================================
def crossfade(x, m=1.0):
    return 1.0 / (1.0 + x**m)

def wt_predicted_acceleration(a_bar, m=1.0, alpha=1.0):
    x = a_bar / A_STAR
    return a_bar + A_STAR * crossfade(x, m) * alpha

def balance_radius_m(m_gal_msun):
    m_kg = float(m_gal_msun) * MSUN
    return (2.0 * m_kg * R_HUBBLE) ** (1.0 / 3.0) / np.pi

def expansion_suppression(r_kpc, m_gal_msun=MW_MASS_MSUN):
    r_m = np.asarray(r_kpc, dtype=float) * KPC_M
    r_bal = balance_radius_m(m_gal_msun)
    return np.exp(-r_m / r_bal)

def wt_predicted_acceleration_with_suppression(a_bar, r_kpc, m=1.0, m_gal_msun=MW_MASS_MSUN, alpha=1.0):
    a_wt = wt_predicted_acceleration(a_bar, m=m, alpha=alpha)
    s = expansion_suppression(r_kpc, m_gal_msun=m_gal_msun)
    return a_wt * s**2

def mond_predicted_acceleration(a_bar):
    return a_bar / (1.0 - np.exp(-np.sqrt(a_bar / A0_MOND)))

# =====================================================================
# Diagnostics
# =====================================================================
def fit_log_slope(r_kpc, a_vals):
    r = np.asarray(r_kpc, dtype=float)
    a = np.asarray(a_vals, dtype=float)
    mask = np.isfinite(r) & np.isfinite(a) & (r > 0.0) & (a > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan
    x = np.log10(r[mask])
    y = np.log10(a[mask])
    s, b = np.polyfit(x, y, 1)
    return float(s), float(b)

def shape_rms_dex(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0.0) & (pred > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan
    resid = np.log10(obs[mask] / pred[mask])
    mean_offset = float(np.mean(resid))
    centered = resid - mean_offset
    rms_dex = float(np.sqrt(np.mean(centered**2)))
    return rms_dex, mean_offset

def classify_shape_rms(rms_dex):
    if not np.isfinite(rms_dex):
        return "n/a"
    if rms_dex < 0.20:
        return "good"
    if rms_dex < 0.28:
        return "salvageable"
    return "quite off"

def classify_shape_regime(delta_s, tol=0.10):
    if not np.isfinite(delta_s):
        return "n/a"
    if delta_s > tol:
        return "too shallow"
    if delta_s < -tol:
        return "too steep"
    return "matched"

def classify_amplitude_regime(offset_dex, tol=0.15):
    if not np.isfinite(offset_dex):
        return "n/a"
    if offset_dex > tol:
        return "low amplitude"
    if offset_dex < -tol:
        return "high amplitude"
    return "matched"

def combine_regimes(shape_regime, amplitude_regime):
    if shape_regime == "n/a" or amplitude_regime == "n/a":
        return "n/a"
    return f"{shape_regime} / {amplitude_regime}"

# =====================================================================
# Evaluation helpers
# =====================================================================
def get_cluster_profiles():
    clusters = []
    for name in get_cluster_list():
        props = CLUSTER_PROPERTIES[name]
        r500 = props[0]
        if r500 is None:
            continue
        r_min = 50.0
        r_max = min(float(r500), 1500.0)
        radii = np.logspace(np.log10(r_min), np.log10(r_max), 12)
        r, a_obs, a_bar, M_tot, M_gas = acceleration_profiles(radii, name)
        clusters.append({
            "name": name,
            "props": props,
            "r": r,
            "a_obs": a_obs,
            "a_bar": a_bar,
        })
    return clusters

def evaluate_params(cluster_cache, m=1.0, apply_suppression=False, m_gal_msun=MW_MASS_MSUN, alpha=1.0):
    wt_residuals = []
    mond_residuals = []
    cluster_shape_stats = []

    for item in cluster_cache:
        name = item["name"]
        props = item["props"]
        r = item["r"]
        a_obs = item["a_obs"]
        a_bar = item["a_bar"]

        a_wt = wt_predicted_acceleration(a_bar, m=m, alpha=alpha)
        if apply_suppression:
            a_wt = wt_predicted_acceleration_with_suppression(
                a_bar, r, m=m, m_gal_msun=m_gal_msun, alpha=alpha
            )
        a_mond = mond_predicted_acceleration(a_bar)

        for j in range(len(r)):
            if a_wt[j] > 0 and a_obs[j] > 0 and a_bar[j] > 0 and a_mond[j] > 0:
                wt_residuals.append(np.log10(a_obs[j] / a_wt[j]))
                mond_residuals.append(np.log10(a_obs[j] / a_mond[j]))

        s_obs, _ = fit_log_slope(r, a_obs)
        s_wt, _ = fit_log_slope(r, a_wt)
        s_mond, _ = fit_log_slope(r, a_mond)
        delta_s_wt = s_wt - s_obs
        delta_s_mond = s_mond - s_obs
        rms_wt, off_wt = shape_rms_dex(a_obs, a_wt)
        rms_mond, off_mond = shape_rms_dex(a_obs, a_mond)

        wt_shape_regime = classify_shape_regime(delta_s_wt)
        wt_amplitude_regime = classify_amplitude_regime(off_wt)

        cluster_shape_stats.append({
            'name': name,
            'Tspec': props[1],
            'r500': props[0],
            'r': r,
            'a_obs': a_obs,
            'a_bar': a_bar,
            'a_wt': a_wt,
            'a_mond': a_mond,
            's_obs': s_obs,
            's_wt': s_wt,
            's_mond': s_mond,
            'delta_s_wt': delta_s_wt,
            'delta_s_mond': delta_s_mond,
            'shape_rms_wt': rms_wt,
            'shape_rms_mond': rms_mond,
            'offset_wt': off_wt,
            'offset_mond': off_mond,
            'wt_shape_regime': wt_shape_regime,
            'wt_amplitude_regime': wt_amplitude_regime,
            'wt_combined_regime': combine_regimes(wt_shape_regime, wt_amplitude_regime),
            'drop_obs': float(a_obs[0] / a_obs[-1]),
            'drop_wt': float(a_wt[0] / a_wt[-1]),
            'drop_mond': float(a_mond[0] / a_mond[-1]),
        })

    wt_res = np.array(wt_residuals, dtype=float)
    mond_res = np.array(mond_residuals, dtype=float)
    wt_shape = np.array([row['shape_rms_wt'] for row in cluster_shape_stats], dtype=float)
    wt_deltas = np.array([row['delta_s_wt'] for row in cluster_shape_stats], dtype=float)
    wt_offsets = np.array([row['offset_wt'] for row in cluster_shape_stats], dtype=float)

    summary = {
        "cluster_shape_stats": cluster_shape_stats,
        "wt_res": wt_res,
        "mond_res": mond_res,
        "mean_log_resid": float(np.nanmean(wt_res)),
        "median_log_resid": float(np.nanmedian(wt_res)),
        "std_log_resid": float(np.nanstd(wt_res)),
        "mean_obs_pred": float(10**np.nanmean(wt_res)),
        "median_obs_pred": float(10**np.nanmedian(wt_res)),
        "mean_delta_s": float(np.nanmean(wt_deltas)),
        "median_delta_s": float(np.nanmedian(wt_deltas)),
        "mean_shape_rms": float(np.nanmean(wt_shape)),
        "median_shape_rms": float(np.nanmedian(wt_shape)),
        "mean_offset": float(np.nanmean(wt_offsets)),
        "median_offset": float(np.nanmedian(wt_offsets)),
        "frac_rms_lt_020": float(np.mean(wt_shape < 0.20) * 100.0),
        "frac_rms_lt_028": float(np.mean(wt_shape < 0.28) * 100.0),
    }
    return summary

def balanced_score(summary):
    return (
        summary["mean_shape_rms"]
        + abs(summary["mean_delta_s"])
        + 0.5 * abs(summary["mean_offset"])
    )

# =====================================================================
# Plotting
# =====================================================================
def choose_plot_rows(rows, n_panels=4, select_mode="best"):
    rows_sorted = sorted(rows, key=lambda d: d["shape_rms_wt"])
    n_panels = max(1, int(n_panels))
    if select_mode == "best":
        return rows_sorted[:n_panels]
    if select_mode == "worst":
        return rows_sorted[-n_panels:]
    if select_mode == "mixed":
        if n_panels == 1:
            return [rows_sorted[0]]
        n_best = int(math.ceil(n_panels / 2))
        n_worst = n_panels - n_best
        picked = rows_sorted[:n_best] + rows_sorted[-n_worst:]
        # preserve unique order if overlap
        seen = set()
        out = []
        for row in picked:
            if row["name"] not in seen:
                out.append(row)
                seen.add(row["name"])
        return out[:n_panels]
    raise ValueError("plot-select must be one of: best, worst, mixed")

def plot_cluster_panels(summary, n_panels=4, plot_file="cluster_match_panels.pdf",
                        select_mode="best", m=1.0, apply_suppression=False,
                        m_gal_msun=MW_MASS_MSUN, alpha=1.0):
    rows = choose_plot_rows(summary["cluster_shape_stats"], n_panels=n_panels, select_mode=select_mode)
    n = len(rows)
    if n == 0:
        return None

    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, row in zip(axes_flat, rows):
        r = row["r"]
        a_obs = row["a_obs"]
        a_wt = row["a_wt"]
        a_mond = row["a_mond"]
        a_bar = row["a_bar"]

        ax.loglog(r, a_obs, marker='o', linewidth=1.8, label='Observed')
        ax.loglog(r, a_wt, marker='s', linewidth=1.8, label='WT')
        ax.loglog(r, a_mond, marker='^', linewidth=1.4, label='MOND')
        ax.loglog(r, a_bar, linestyle='--', linewidth=1.2, label='Baryonic')

        ax.set_title(row["name"])
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("a [m s$^{-2}$]")
        ax.grid(True, which='both', alpha=0.25)

        info = (
            f"Tspec={row['Tspec']:.1f} keV\n"
            f"RMS={row['shape_rms_wt']:.3f} dex\n"
            f"Δs={row['delta_s_wt']:.3f}\n"
            f"offset={row['offset_wt']:.3f} dex\n"
            #f"{row['wt_combined_regime']}"
        )
        ax.text(
            0.03, 0.03, info,
            transform=ax.transAxes,
            fontsize=8,
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75)
        )

    for ax in axes_flat[n:]:
        ax.axis('off')

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
    sup = (
        f"WT cluster match panels | m={m:g} | suppression={'on' if apply_suppression else 'off'} | "
        f"M={m_gal_msun:.3e} Msun | alpha={alpha:.4g} | select={select_mode} | "
        f"meanRMS={summary['mean_shape_rms']:.4f} dex | meanΔs={summary['mean_delta_s']:.4f}"
    )
    #fig.suptitle(sup, y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)
    return plot_file

# =====================================================================
# Output
# =====================================================================
def print_single_run(summary, m=1.0, apply_suppression=False, m_gal_msun=MW_MASS_MSUN, alpha=1.0):
    wt_res = summary["wt_res"]
    cluster_shape_stats = summary["cluster_shape_stats"]

    print("=" * 95)
    print("WAVE THEORY CLUSTER TEST")
    print(f"a_star = {A_STAR:.4e} m/s^2   (c^2 / pi*R_H)")
    print(f"crossfade m = {m}")
    print(f"a_0(MOND) = {A0_MOND:.4e} m/s^2")
    print(f"alpha (A_morph*κ) = {alpha:.2e}")
    if apply_suppression:
        r_bal_m = balance_radius_m(m_gal_msun)
        print(f"suppression = ON   S(r)=exp(-r/r_bal),  M_gal = {m_gal_msun:.3e} Msun")
        print(f"r_bal = {r_bal_m:.4e} m = {r_bal_m / KPC_M:.2f} kpc")
    else:
        print("suppression = OFF")
    print("=" * 95)

    print()
    print("=" * 95)
    print("SUMMARY STATISTICS (all radial points, all clusters)")
    print("=" * 95)
    print(f"N data points: {len(wt_res)}")
    print()
    print(f"{'Metric':34s} {'WT':>12s}")
    print("-" * 50)
    print(f"{'mean log10(obs/pred)':34s} {summary['mean_log_resid']:12.4f}")
    print(f"{'median log10(obs/pred)':34s} {summary['median_log_resid']:12.4f}")
    print(f"{'std log10(obs/pred)':34s} {summary['std_log_resid']:12.4f}")
    print(f"{'mean obs/pred':34s} {summary['mean_obs_pred']:12.4f}")
    print(f"{'median obs/pred':34s} {summary['median_obs_pred']:12.4f}")

    for fac in [1.5, 2.0, 3.0]:
        wt_frac = np.mean(np.abs(wt_res) < np.log10(fac)) * 100
        print(f"{('within factor ' + str(fac)):34s} {wt_frac:11.1f}%")

    print()
    print("=" * 230)
    print("PER-CLUSTER SHAPE DIAGNOSTICS")
    print("=" * 230)
    print(
        f"{'Cluster':16s} {'s_obs':>8s} {'s_WT':>8s} {'Δs_WT':>9s} {'RMS_WT':>8s} {'offset_WT':>10s} "
        f"{'shape_regime':>14s} {'amp_regime':>15s} {'combined':>28s} {'drop_obs':>10s} {'drop_WT':>10s} {'grade':>12s}"
    )
    print("-" * 230)
    for row in sorted(cluster_shape_stats, key=lambda d: d['shape_rms_wt']):
        print(
            f"{row['name']:16s} "
            f"{row['s_obs']:8.3f} {row['s_wt']:8.3f} {row['delta_s_wt']:9.3f} {row['shape_rms_wt']:8.3f} {row['offset_wt']:10.3f} "
            f"{row['wt_shape_regime']:>14s} {row['wt_amplitude_regime']:>15s} {row['wt_combined_regime']:>28s} "
            f"{row['drop_obs']:10.2f} {row['drop_wt']:10.2f} {classify_shape_rms(row['shape_rms_wt']):>12s}"
        )

    print()
    print("=" * 95)
    print("GLOBAL SHAPE SUMMARY (per-cluster, normalization-free)")
    print("=" * 95)
    print(f"{'Metric':34s} {'WT':>12s}")
    print("-" * 50)
    print(f"{'mean slope mismatch Δs':34s} {summary['mean_delta_s']:12.4f}")
    print(f"{'median slope mismatch Δs':34s} {summary['median_delta_s']:12.4f}")
    print(f"{'mean shape RMS [dex]':34s} {summary['mean_shape_rms']:12.4f}")
    print(f"{'median shape RMS [dex]':34s} {summary['median_shape_rms']:12.4f}")
    print(f"{'mean offset [dex]':34s} {summary['mean_offset']:12.4f}")
    print(f"{'median offset [dex]':34s} {summary['median_offset']:12.4f}")
    print(f"{'clusters with RMS < 0.20 dex':34s} {summary['frac_rms_lt_020']:11.1f}%")
    print(f"{'clusters with RMS < 0.28 dex':34s} {summary['frac_rms_lt_028']:11.1f}%")

def parse_alpha_grid(spec):
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("alpha-grid must be start,stop,step")
    start, stop, step = parts
    n = int(np.floor((stop - start) / step + 0.5)) + 1
    vals = start + step * np.arange(n)
    return vals[(vals >= min(start, stop) - 1e-12) & (vals <= max(start, stop) + 1e-12)]

def parse_mass_grid(spec):
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("mass-grid must be start,stop,num_points")
    start, stop, num = parts
    num = int(num)
    if start <= 0 or stop <= 0 or num < 2:
        raise ValueError("mass-grid requires positive start/stop and num_points >= 2")
    return np.geomspace(start, stop, num)

def print_scan_results(results, top=10):
    print()
    print("=" * 140)
    print("GLOBAL ALPHA / MASS SCAN")
    print("=" * 140)
    print(f"Scanned {len(results)} parameter combinations")
    print()
    print("Balanced score = mean_shape_rms + |mean_delta_s| + 0.5*|mean_offset|")
    print("Lower is better.")
    print()

    ranked = sorted(results, key=lambda d: d["score"])
    print(f"{'rank':>4s} {'alpha':>8s} {'mass[Msun]':>14s} {'score':>9s} {'meanRMS':>9s} {'meanΔs':>9s} {'meanOff':>9s} {'<0.20dex':>9s}")
    print("-" * 90)
    for i, row in enumerate(ranked[:top], start=1):
        print(f"{i:4d} {row['alpha']:8.4f} {row['mass']:14.4e} {row['score']:9.4f} "
              f"{row['mean_shape_rms']:9.4f} {row['mean_delta_s']:9.4f} {row['mean_offset']:9.4f} {row['frac_rms_lt_020']:8.1f}%")

    best_shape = min(results, key=lambda d: d["mean_shape_rms"])
    best_slope = min(results, key=lambda d: abs(d["mean_delta_s"]))
    best_amp = min(results, key=lambda d: abs(d["mean_offset"]))

    print()
    print("Single-metric winners")
    print("-" * 90)
    print(f"Best shape     : alpha={best_shape['alpha']:.4f} mass={best_shape['mass']:.4e} "
          f"meanRMS={best_shape['mean_shape_rms']:.4f}")
    print(f"Best slope     : alpha={best_slope['alpha']:.4f} mass={best_slope['mass']:.4e} "
          f"meanΔs={best_slope['mean_delta_s']:.4f}")
    print(f"Best amplitude : alpha={best_amp['alpha']:.4f} mass={best_amp['mass']:.4e} "
          f"meanOff={best_amp['mean_offset']:.4f}")

# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="WT cluster test with optional global alpha/mass scan and panel plotting")
    parser.add_argument("m", nargs="?", type=float, default=1.0, help="crossfade exponent m")
    parser.add_argument("apply_suppression", nargs="?", type=int, default=0, help="0/1")
    parser.add_argument("m_gal_msun", nargs="?", type=float, default=MW_MASS_MSUN, help="transition mass scale")
    parser.add_argument("alpha", nargs="?", type=float, default=1.0, help="global alpha")

    parser.add_argument("--scan", action="store_true", help="scan one global alpha and one global mass")
    parser.add_argument("--alpha-grid", default="0.5,5.0,0.1",
                        help="linear alpha grid: start,stop,step")
    parser.add_argument("--mass-grid", default="1e11,2e15,25",
                        help="log mass grid: start,stop,num_points")
    parser.add_argument("--top", type=int, default=12, help="number of best scan rows to show")
    parser.add_argument("--no-single", action="store_false",
                        help="when --scan is used, skip the initial single-run report")

    parser.add_argument("--plot-panels", type=int, default=0,
                        help="save a multi-panel cluster match figure with this many panels")
    parser.add_argument("--plot-select", default="best", choices=["best", "worst", "mixed"],
                        help="which clusters to show in the panel plot")
    parser.add_argument("--plot-file", default="cluster_match_panels.pdf",
                        help="output figure filename (.pdf or .png)")

    args = parser.parse_args()
    apply_suppression = bool(args.apply_suppression)

    cluster_cache = get_cluster_profiles()
    single = None

    if not (args.scan and args.no_single):
        single = evaluate_params(
            cluster_cache,
            m=args.m,
            apply_suppression=apply_suppression,
            m_gal_msun=args.m_gal_msun,
            alpha=args.alpha,
        )
        print_single_run(
            single,
            m=args.m,
            apply_suppression=apply_suppression,
            m_gal_msun=args.m_gal_msun,
            alpha=args.alpha,
        )

    if args.plot_panels > 0:
        if single is None:
            single = evaluate_params(
                cluster_cache,
                m=args.m,
                apply_suppression=apply_suppression,
                m_gal_msun=args.m_gal_msun,
                alpha=args.alpha,
            )
        outfile = plot_cluster_panels(
            single,
            n_panels=args.plot_panels,
            plot_file=args.plot_file,
            select_mode=args.plot_select,
            m=args.m,
            apply_suppression=apply_suppression,
            m_gal_msun=args.m_gal_msun,
            alpha=args.alpha,
        )
        if outfile is not None:
            print()
            print(f"Saved panel figure: {outfile}")

    if args.scan:
        alphas = parse_alpha_grid(args.alpha_grid)
        masses = parse_mass_grid(args.mass_grid)
        results = []

        for alpha in alphas:
            for mass in masses:
                summary = evaluate_params(
                    cluster_cache,
                    m=args.m,
                    apply_suppression=apply_suppression,
                    m_gal_msun=float(mass),
                    alpha=float(alpha),
                )
                results.append({
                    "alpha": float(alpha),
                    "mass": float(mass),
                    "score": float(balanced_score(summary)),
                    "mean_shape_rms": summary["mean_shape_rms"],
                    "mean_delta_s": summary["mean_delta_s"],
                    "mean_offset": summary["mean_offset"],
                    "frac_rms_lt_020": summary["frac_rms_lt_020"],
                })

        print_scan_results(results, top=args.top)

if __name__ == "__main__":
    main()
