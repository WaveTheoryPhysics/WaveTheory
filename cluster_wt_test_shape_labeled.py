"""
Wave Theory cluster-scale test
================================
Apply the EXACT same master formula used for galaxy rotation curves
to cluster acceleration profiles from Vikhlinin et al. (2006).

NO modifications to the geometric mechanism.
The formula is:

  a_obs = a_bar + a_star * kappa(a_bar / a_star)

where kappa is the crossfade function and a_star = c^2 / (pi * R)
with R = c/H0 (Hubble radius) giving a_star ~ 6.59e-10 m/s^2.

This is exactly what produced the galaxy rotation curves.

Extended version:
- adds slope mismatch in log(a) vs log(r)
- adds pure shape RMS in dex after removing mean normalization offset
- ranks clusters by shape quality, so suppression / envelope ideas can be iterated quickly
"""

import numpy as np
import sys

# Import the Vikhlinin data module
from vikhlinin2006_data import (
    get_cluster_list, CLUSTER_PROPERTIES,
    acceleration_profiles, KPC_M
)

# =====================================================================
# Wave Theory master formula — IDENTICAL to galaxy code
# =====================================================================
C_LIGHT = 2.998e8       # m/s
H0      = 70.0          # km/s/Mpc
H0_SI   = H0 * 1e3 / (3.0857e22)  # s^-1

# Geometric acceleration scale
R_HUBBLE = 4.286332662 * 10**26               # Hubble radius in meters
A_STAR   = C_LIGHT**2 / (np.pi * R_HUBBLE)   # ~ 6.59e-11 m/s^2
A0_MOND  = 1.2e-10  # for comparison
MSUN      = 1.98847e30
MW_MASS_MSUN = 5.3e11   # Milky-Way-like galaxy mass for suppression scale


def crossfade(x, m=1.0):
    """
    Crossfade function kappa(x) with steepness m.
    kappa(x) = 1 / (1 + x^m)

    This is the SAME function used for galaxies.
    """
    return 1.0 / (1.0 + x**m)


def wt_predicted_acceleration(a_bar, m=1.0, alpha=1.0):
    """
    Wave Theory master formula:
      a_obs = a_bar + a_star * kappa(a_bar / a_star)

    IDENTICAL to galaxy rotation curve formula.
    """
    x = a_bar / A_STAR
    kappa = crossfade(x, m)
    return a_bar + A_STAR * kappa * alpha


def balance_radius_m(m_gal_msun):
    """Gravity-expansion balance radius for a characteristic galaxy mass.

    r_bal = (2 G M / H^2)^(1/3) = (2 M R_HUBBLE)^(1/3) / pi
    with M in kg and R_HUBBLE in m.
    """
    m_kg = float(m_gal_msun) * MSUN
    return (2.0 * m_kg * R_HUBBLE) ** (1.0 / 3.0) / np.pi


def expansion_suppression(r_kpc, m_gal_msun=MW_MASS_MSUN):
    """Optional expansion suppression applied at the velocity level.

    S(r;M_gal) = exp(-r / r_bal(M_gal))

    Since this script predicts accelerations, multiplying velocity by S
    corresponds to multiplying acceleration by S^2.
    """
    r_m = np.asarray(r_kpc, dtype=float) * KPC_M
    r_bal = balance_radius_m(m_gal_msun)
    return np.exp(-r_m / r_bal)


def wt_predicted_acceleration_with_suppression(a_bar, r_kpc, m=1.0, m_gal_msun=MW_MASS_MSUN, alpha=1.0):
    """Original WT acceleration multiplied by S(r;M_gal)^2.

    The core formula is left unchanged; this only adds the optional
    expansion-suppression envelope.
    """
    a_wt = wt_predicted_acceleration(a_bar, m=m, alpha=alpha)
    s = expansion_suppression(r_kpc, m_gal_msun=m_gal_msun)
    return a_wt * s**2


def mond_predicted_acceleration(a_bar):
    """Standard MOND (simple interpolation) for comparison."""
    return a_bar / (1.0 - np.exp(-np.sqrt(a_bar / A0_MOND)))


# =====================================================================
# Shape diagnostics
# =====================================================================
def fit_log_slope(r_kpc, a_vals):
    """Best-fit slope s of log10(a) = s log10(r) + b."""
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
    """Pure shape error in dex after removing mean normalization offset.

    Returns:
      rms_dex      : RMS of centered log residuals
      mean_offset  : mean log10(obs/pred), the removed normalization offset
    """
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
    """Signed slope regime from Δs = s_pred - s_obs.

    Positive Δs means prediction is too shallow.
    Negative Δs means prediction is too steep.
    """
    if not np.isfinite(delta_s):
        return "n/a"
    if delta_s > tol:
        return "too shallow"
    if delta_s < -tol:
        return "too steep"
    return "matched"


def classify_amplitude_regime(offset_dex, tol=0.15):
    """Signed amplitude regime from mean log10(obs/pred).

    Positive offset means prediction is systematically too low.
    Negative offset means prediction is systematically too high.
    """
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
# Run the test
# =====================================================================
def run_cluster_test(m=1.0, apply_suppression=False, m_gal_msun=MW_MASS_MSUN, alpha=1.0):
    """
    Apply WT master formula to all 12 Vikhlinin clusters.
    No free parameters — a_star is fixed, m is the same as galaxy fits.
    """

    clusters = get_cluster_list()

    print("=" * 95)
    print("WAVE THEORY CLUSTER TEST")
    print(f"a_star = {A_STAR:.4e} m/s^2   (c^2 / pi*R_H)")
    print(f"crossfade m = {m}")
    print(f"a_0(MOND) = {A0_MOND:.4e} m/s^2")
    print(f"alpha (A_morph*κ) = {alpha:.2e} ")
    if apply_suppression:
        r_bal_m = balance_radius_m(m_gal_msun)
        print(f"suppression = ON   S(r)=exp(-r/r_bal),  M_gal = {m_gal_msun:.3e} Msun")
        print(f"r_bal = {r_bal_m:.4e} m = {r_bal_m / KPC_M:.2f} kpc")
    else:
        print("suppression = OFF")
    print("=" * 95)

    print(f"\n{'Cluster':16s} {'r(kpc)':>7s} {'a_obs':>11s} {'a_bar':>11s} "
          f"{'a_WT':>11s} {'a_MOND':>11s} {'obs/WT':>7s} {'obs/MOND':>8s}")
    print("-" * 95)

    # Collect residuals for summary
    wt_residuals = []
    mond_residuals = []
    cluster_shape_stats = []

    for name in clusters:
        props = CLUSTER_PROPERTIES[name]
        r500 = props[0]
        if r500 is None:
            continue

        # Radial grid: 50 kpc to r500, log-spaced
        r_min = 50.0
        r_max = min(float(r500), 1500.0)
        radii = np.logspace(np.log10(r_min), np.log10(r_max), 12)

        r, a_obs, a_bar, M_tot, M_gas = acceleration_profiles(radii, name)

        # Apply WT and MOND predictions
        a_wt = wt_predicted_acceleration(a_bar, m=m, alpha=alpha)
        if apply_suppression:
            a_wt = wt_predicted_acceleration_with_suppression(
                a_bar, r, m=m, m_gal_msun=m_gal_msun, alpha=alpha
            )
        a_mond = mond_predicted_acceleration(a_bar)

        # Print selected radii
        show_idx = [0, 3, 6, 9, 11]  # 5 representative radii
        for j in show_idx:
            if j >= len(r):
                continue
            ratio_wt = a_obs[j] / a_wt[j] if a_wt[j] > 0 else float('nan')
            ratio_mond = a_obs[j] / a_mond[j] if a_mond[j] > 0 else float('nan')

            marker = " " if 0.7 < ratio_wt < 1.4 else "*"
            print(f"{name:16s} {r[j]:7.0f} {a_obs[j]:11.3e} {a_bar[j]:11.3e} "
                  f"{a_wt[j]:11.3e} {a_mond[j]:11.3e} {ratio_wt:7.3f} "
                  f"{ratio_mond:8.3f}{marker}")
        print()

        # Collect all points for global statistics
        for j in range(len(r)):
            if a_wt[j] > 0 and a_obs[j] > 0 and a_bar[j] > 0 and a_mond[j] > 0:
                wt_residuals.append(np.log10(a_obs[j] / a_wt[j]))
                mond_residuals.append(np.log10(a_obs[j] / a_mond[j]))

        # Per-cluster shape diagnostics
        s_obs, _ = fit_log_slope(r, a_obs)
        s_wt, _ = fit_log_slope(r, a_wt)
        s_mond, _ = fit_log_slope(r, a_mond)
        delta_s_wt = s_wt - s_obs
        delta_s_mond = s_mond - s_obs
        rms_wt, off_wt = shape_rms_dex(a_obs, a_wt)
        rms_mond, off_mond = shape_rms_dex(a_obs, a_mond)

        wt_shape_regime = classify_shape_regime(delta_s_wt)
        wt_amplitude_regime = classify_amplitude_regime(off_wt)
        mond_shape_regime = classify_shape_regime(delta_s_mond)
        mond_amplitude_regime = classify_amplitude_regime(off_mond)

        cluster_shape_stats.append({
            'name': name,
            'Tspec': props[1],
            'r_min': float(np.min(r)),
            'r_max': float(np.max(r)),
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
            'mond_shape_regime': mond_shape_regime,
            'mond_amplitude_regime': mond_amplitude_regime,
            'mond_combined_regime': combine_regimes(mond_shape_regime, mond_amplitude_regime),
            'drop_obs': float(a_obs[0] / a_obs[-1]),
            'drop_wt': float(a_wt[0] / a_wt[-1]),
            'drop_mond': float(a_mond[0] / a_mond[-1]),
        })

    # Summary statistics
    wt_res = np.array(wt_residuals)
    mond_res = np.array(mond_residuals)

    print("=" * 95)
    print("SUMMARY STATISTICS (all radial points, all clusters)")
    print("=" * 95)
    print(f"N data points: {len(wt_res)}")
    print()
    print(f"{'Metric':30s} {'WT':>12s} {'MOND':>12s}")
    print("-" * 55)
    print(f"{'mean log10(obs/pred)':30s} {np.mean(wt_res):12.4f} "
          f"{np.mean(mond_res):12.4f}")
    print(f"{'median log10(obs/pred)':30s} {np.median(wt_res):12.4f} "
          f"{np.median(mond_res):12.4f}")
    print(f"{'std log10(obs/pred)':30s} {np.std(wt_res):12.4f} "
          f"{np.std(mond_res):12.4f}")
    print(f"{'mean obs/pred':30s} {10**np.mean(wt_res):12.4f} "
          f"{10**np.mean(mond_res):12.4f}")
    print(f"{'median obs/pred':30s} {10**np.median(wt_res):12.4f} "
          f"{10**np.median(mond_res):12.4f}")

    # Fraction within factors
    for fac in [1.5, 2.0, 3.0]:
        wt_frac = np.mean(np.abs(wt_res) < np.log10(fac)) * 100
        mond_frac = np.mean(np.abs(mond_res) < np.log10(fac)) * 100
        print(f"{'within factor ' + str(fac):30s} {wt_frac:11.1f}% "
              f"{mond_frac:11.1f}%")

    # Per-cluster summary at r = 500 kpc
    print()
    print("=" * 95)
    print("PER-CLUSTER SUMMARY at r = 500 kpc")
    print("=" * 95)
    print(f"{'Cluster':16s} {'Tspec':>5s} {'a_obs':>11s} {'a_bar':>11s} "
          f"{'a_WT':>11s} {'obs/WT':>7s} {'a_bar/a*':>9s} {'regime':>10s}")
    print("-" * 95)

    for name in clusters:
        props = CLUSTER_PROPERTIES[name]
        r500 = props[0]
        if r500 is None or r500 < 500:
            continue

        r_arr = np.array([500.0])
        _, a_obs, a_bar, _, _ = acceleration_profiles(r_arr, name)
        a_wt = wt_predicted_acceleration(a_bar, m=m, alpha=alpha)
        if apply_suppression:
            a_wt = wt_predicted_acceleration_with_suppression(
                a_bar, r_arr, m=m, m_gal_msun=m_gal_msun, alpha=alpha
            )

        x = a_bar[0] / A_STAR
        ratio = a_obs[0] / a_wt[0]

        if x < 0.1:
            regime = "deep-MOND"
        elif x < 1.0:
            regime = "crossfade"
        else:
            regime = "Newtonian"

        print(f"{name:16s} {props[1]:5.1f} {a_obs[0]:11.3e} {a_bar[0]:11.3e} "
              f"{a_wt[0]:11.3e} {ratio:7.3f} {x:9.4f} {regime:>10s}")

    # New: cluster-by-cluster shape diagnostics
    print()
    print("=" * 220)
    print("PER-CLUSTER SHAPE DIAGNOSTICS")
    print("=" * 220)
    print(
        f"{'Cluster':16s} {'s_obs':>8s} {'s_WT':>8s} {'Δs_WT':>9s} {'RMS_WT':>8s} {'offset_WT':>10s} "
        f"{'shape_regime':>14s} {'amp_regime':>15s} {'combined':>28s} "
        f"{'s_MOND':>8s} {'Δs_MOND':>10s} {'RMS_MOND':>10s} {'drop_obs':>10s} {'drop_WT':>10s} {'grade':>12s}"
    )
    print("-" * 220)

    for row in sorted(cluster_shape_stats, key=lambda d: d['shape_rms_wt']):
        print(
            f"{row['name']:16s} "
            f"{row['s_obs']:8.3f} {row['s_wt']:8.3f} {row['delta_s_wt']:9.3f} {row['shape_rms_wt']:8.3f} {row['offset_wt']:10.3f} "
            f"{row['wt_shape_regime']:>14s} {row['wt_amplitude_regime']:>15s} {row['wt_combined_regime']:>28s} "
            f"{row['s_mond']:8.3f} {row['delta_s_mond']:10.3f} {row['shape_rms_mond']:10.3f} "
            f"{row['drop_obs']:10.2f} {row['drop_wt']:10.2f} {classify_shape_rms(row['shape_rms_wt']):>12s}"
        )

    # Global shape summary
    wt_shape = np.array([row['shape_rms_wt'] for row in cluster_shape_stats], dtype=float)
    mond_shape = np.array([row['shape_rms_mond'] for row in cluster_shape_stats], dtype=float)
    wt_deltas = np.array([row['delta_s_wt'] for row in cluster_shape_stats], dtype=float)
    mond_deltas = np.array([row['delta_s_mond'] for row in cluster_shape_stats], dtype=float)

    print()
    print("=" * 95)
    print("GLOBAL SHAPE SUMMARY (per-cluster, normalization-free)")
    print("=" * 95)
    print(f"{'Metric':34s} {'WT':>12s} {'MOND':>12s}")
    print("-" * 62)
    print(f"{'mean slope mismatch Δs':34s} {np.nanmean(wt_deltas):12.4f} {np.nanmean(mond_deltas):12.4f}")
    print(f"{'median slope mismatch Δs':34s} {np.nanmedian(wt_deltas):12.4f} {np.nanmedian(mond_deltas):12.4f}")
    print(f"{'mean shape RMS [dex]':34s} {np.nanmean(wt_shape):12.4f} {np.nanmean(mond_shape):12.4f}")
    print(f"{'median shape RMS [dex]':34s} {np.nanmedian(wt_shape):12.4f} {np.nanmedian(mond_shape):12.4f}")
    print(f"{'clusters with RMS < 0.20 dex':34s} {np.mean(wt_shape < 0.20) * 100:11.1f}% {np.mean(mond_shape < 0.20) * 100:11.1f}%")
    print(f"{'clusters with RMS < 0.28 dex':34s} {np.mean(wt_shape < 0.28) * 100:11.1f}% {np.mean(mond_shape < 0.28) * 100:11.1f}%")

    # The critical question: what does WT predict vs what we see?
    print()
    print("=" * 95)
    print("INTERPRETATION")
    print("=" * 95)
    print(f"a_star = {A_STAR:.4e} m/s^2")
    print(f"At r=500 kpc in clusters, a_bar ~ 1e-11 m/s^2")
    print(f"So a_bar/a_star ~ {1e-11/A_STAR:.4f}")
    print(f"kappa(a_bar/a_star) ~ {crossfade(1e-11/A_STAR, m):.6f}")
    print(f"WT boost = a_star * kappa ~ {A_STAR * crossfade(1e-11/A_STAR, m):.4e} m/s^2")
    print(f"This means a_WT ~ a_bar + a_star*kappa ~ {1e-11 + A_STAR * crossfade(1e-11/A_STAR, m):.4e}")
    print(f"But observed a_obs ~ 1e-10 m/s^2")
    print()
    print("The ratio a_obs/a_WT tells us if the formula works at cluster scale.")
    print("Positive Δs means WT is too shallow; shape RMS isolates shape error after removing normalization.")


if __name__ == "__main__":
    m_val = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    apply_suppression = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    m_gal_msun = float(sys.argv[3]) if len(sys.argv) > 3 else MW_MASS_MSUN
    m_alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    run_cluster_test(m=m_val, apply_suppression=apply_suppression, m_gal_msun=m_gal_msun, alpha=m_alpha)
