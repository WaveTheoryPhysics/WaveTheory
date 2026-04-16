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
"""

import numpy as np
import sys

# Import the Vikhlinin data module
from vikhlinin2006_data import (
    get_cluster_list, CLUSTER_PROPERTIES, GAS_DENSITY_PARAMS,
    acceleration_profiles, KPC_M, G_SI, MSUN_KG
)

# =====================================================================
# Wave Theory master formula — IDENTICAL to galaxy code
# =====================================================================
C_LIGHT = 2.998e8       # m/s
H0      = 70.0          # km/s/Mpc
H0_SI   = H0 * 1e3 / (3.0857e22)  # s^-1

# Geometric acceleration scale
R_HUBBLE = 4.286332662 * 10**26               # Hubble radius in meters
A_STAR   = C_LIGHT**2 / (np.pi * R_HUBBLE)   # ~ 6.59e-10 m/s^2
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
# Run the test
# =====================================================================
def run_cluster_test(m=1.0, apply_suppression=False, m_gal_msun=MW_MASS_MSUN, alpha=1.0):
    """
    Apply WT master formula to all 12 Vikhlinin clusters.
    No free parameters — a_star is fixed, m is the same as galaxy fits.
    """
    
    clusters = get_cluster_list()
    
    print("=" * 95)
    print(f"WAVE THEORY CLUSTER TEST")
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
    
    # Radial grid for each cluster
    all_results = {}
    
    print(f"\n{'Cluster':16s} {'r(kpc)':>7s} {'a_obs':>11s} {'a_bar':>11s} "
          f"{'a_WT':>11s} {'a_MOND':>11s} {'obs/WT':>7s} {'obs/MOND':>8s}")
    print("-" * 95)
    
    # Collect residuals for summary
    wt_residuals = []
    mond_residuals = []
    wt_data_all = []
    
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
            a_wt = wt_predicted_acceleration_with_suppression(a_bar, r, m=m, m_gal_msun=m_gal_msun, alpha=alpha)
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
        
        # Collect all points for statistics
        for j in range(len(r)):
            if a_wt[j] > 0 and a_obs[j] > 0 and a_bar[j] > 0:
                wt_residuals.append(np.log10(a_obs[j] / a_wt[j]))
                mond_residuals.append(np.log10(a_obs[j] / a_mond[j]))
                wt_data_all.append({
                    'name': name, 'r': r[j],
                    'a_obs': a_obs[j], 'a_bar': a_bar[j],
                    'a_wt': a_wt[j], 'a_mond': a_mond[j],
                    'Tspec': props[1]
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
        a_wt = wt_predicted_acceleration(a_bar, m=m)
        if apply_suppression:
            a_wt = wt_predicted_acceleration_with_suppression(a_bar, r_arr, m=m, m_gal_msun=m_gal_msun)
        
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


if __name__ == "__main__":
    m_val = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    apply_suppression = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    m_gal_msun = float(sys.argv[3]) if len(sys.argv) > 3 else MW_MASS_MSUN
    m_alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    run_cluster_test(m=m_val, apply_suppression=apply_suppression, m_gal_msun=m_gal_msun, alpha=m_alpha)

