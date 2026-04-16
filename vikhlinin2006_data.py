"""
Vikhlinin et al. (2006) — Chandra cluster mass profiles
=========================================================

Data extracted from Tables 1, 2, and 3 of:
  Vikhlinin, Kravtsov, Forman, Jones, Markevitch, Murray & Van Speybroeck
  "Chandra Sample of Nearby Relaxed Galaxy Clusters: Mass, Gas Fraction,
   and Mass-Temperature Relation"
  ApJ, 2006 (astro-ph/0507092)

Cosmology used: Omega_M=0.3, Lambda=0.7, h=0.71

This module provides:
  - Raw table data for 13 clusters
  - Analytic gas density model (eq. 3)
  - Analytic temperature model (eqs. 4-6)
  - Hydrostatic mass derivation (eq. 7)
  - Radial profiles of a_obs(r) and a_bar(r)
"""

import numpy as np
import math

# =====================================================================
# Constants
# =====================================================================
G_CGS    = 6.6743e-8       # cm^3 g^-1 s^-2
G_SI     = 6.6743e-11      # m^3 kg^-1 s^-2
MSUN     = 1.98847e33      # g
MSUN_KG  = 1.98847e30      # kg
KPC_CM   = 3.0857e21       # cm
MPC_CM   = 3.0857e24       # cm
KPC_M    = 3.0857e19       # m
KEV_ERG  = 1.60218e-9      # erg per keV
MP_G     = 1.6726e-24      # proton mass in g
KB_CGS   = 1.3807e-16      # Boltzmann constant erg/K
MU       = 0.6             # mean molecular weight (fully ionized, cosmic abundance)
MU_E     = 1.17            # mean molecular weight per electron


# =====================================================================
# Table 1: Cluster sample (redshifts)
# =====================================================================
CLUSTER_REDSHIFTS = {
    "A133":          0.0569,
    "A262":          0.0162,
    "A383":          0.1883,
    "A478":          0.0881,
    "A907":          0.1603,
    "A1413":         0.1429,
    "A1795":         0.0622,
    "A1991":         0.0592,
    "A2029":         0.0779,
    "A2390":         0.2302,
    "RXJ1159+5531":  0.0810,
    "MKW4":          0.0199,
    "USGCS152":      0.0153,
}


# =====================================================================
# Table 2: Gas density model parameters (eq. 3)
#
#   n_p n_e = n0^2 * (r/rc)^(-alpha) / (1 + r^2/rc^2)^(3*beta - alpha/2)
#             * 1/(1 + r^gamma/rs^gamma)^(epsilon/gamma)
#             + n02^2 / (1 + r^2/rc2^2)^(3*beta2)
#
# All fits use gamma = 3 (fixed).
# Units: n0 in 10^-3 cm^-3, rc/rs in kpc, n02 in 10^-1 cm^-3
# =====================================================================
GAS_DENSITY_PARAMS = {
    # name:       (rdet_kpc, n0_1e-3, rc_kpc, rs_kpc, alpha, beta, epsilon,
    #              n02_1e-1, rc2_kpc, beta2)
    # n02=0, rc2=0, beta2=0 means no second component
    "A133":         (1100, 2.968, 142.7, 1423.3, 0.996, 0.575, 5.000,
                     0.276, 33.44, 0.980),
    "A262":         (450,  3.434, 45.2,  350.8,  1.674, 0.333, 1.806,
                     0.0,  0.0,   0.0),
    "A383":         (800,  7.000, 115.2, 422.3,  2.018, 0.583, 0.740,
                     1.014, 0.08, 1.000),
    "A478":         (2000, 8.169, 177.1, 3148.2, 1.493, 0.715, 5.000,
                     0.584, 24.00, 1.000),
    "A907":         (1300, 6.257, 136.9, 1885.1, 1.554, 0.594, 4.986,
                     0.0,  0.0,   0.0),
    "A1413":        (1800, 5.526, 186.3, 2077.1, 1.217, 0.651, 4.991,
                     0.0,  0.0,   0.0),
    "A1795":        (1500, 14.993, 72.8, 1030.8, 1.060, 0.545, 3.474,
                     0.0,  0.0,   0.0),
    "A1991":        (1000, 9.373, 44.4,  998.2,  1.516, 0.501, 5.000,
                     0.999, 5.00, 1.165),
    "A2029":        (2250, 16.469, 80.7, 870.6,  1.131, 0.539, 1.650,
                     3.741, 5.00, 1.000),
    "A2390":        (2500, 3.069, 353.6, 1200.0, 1.917, 0.696, 0.240,
                     0.0,  0.0,   0.0),
    "RXJ1159+5531": (600,  0.198, 613.8, 961.5,  1.762, 1.215, 4.939,
                     0.416, 12.66, 1.000),
    "MKW4":         (550,  0.280, 488.6, 1081.6, 1.628, 1.224, 0.000,
                     0.189, 11.08, 0.661),
    "USGCS152":     (300,  17.450, 8.1,  467.5,  2.644, 0.453, 3.280,
                     0.0,  0.0,   0.0),
}


# =====================================================================
# Table 3: Derived cluster properties
#   r500 (kpc), Tspec (keV), Tmg (keV), c500,
#   M2500 (1e14 Msun), M500 (1e14 Msun),
#   fg_2500, fg_500
# =====================================================================
CLUSTER_PROPERTIES = {
    # name:      (r500, Tspec, Tmg, c500, M2500, M500, fg2500, fg500)
    "A133":      (998,  4.15, 3.68, 3.31, 1.24, 3.14, 0.065, 0.084),
    "A262":      (663,  2.08, 1.92, 3.58, 0.34, None, 0.067, None),
    "A383":      (956,  4.80, 4.36, 4.36, 1.68, 3.10, 0.090, 0.122),
    "A478":      (1359, 7.95, 7.34, 3.75, 4.23, 7.83, 0.096, 0.118),
    "A907":      (1117, 5.96, 5.44, 3.56, 2.30, 4.71, 0.090, 0.121),
    "A1413":     (1339, 7.38, 6.76, 3.06, 3.08, 7.78, 0.092, 0.104),
    "A1795":     (1283, 6.10, 5.52, 3.73, 2.66, 6.57, 0.088, 0.098),
    "A1991":     (753,  2.59, 2.23, 4.66, 0.60, 1.28, 0.068, 0.099),
    "A2029":     (1380, 8.46, 7.59, 4.13, 4.41, 8.29, 0.090, 0.121),
    "A2390":     (1448, 8.90, 9.35, 1.65, 3.50, 10.88, 0.127, 0.138),
    "RXJ1159+5531": (695, 1.80, 1.58, 1.89, 0.31, None, 0.042, None),
    "MKW4":      (635,  1.65, 1.40, 2.69, 0.30, 0.74, 0.044, 0.063),
    "USGCS152":  (None, 0.69, 0.59, None, 0.07, None, 0.043, None),
}


# =====================================================================
# Temperature model parameters
# =====================================================================
# Vikhlinin uses T_3D(r) = T0 * t_cool(r) * t(r)   (eq. 6)
# where:
#   t(r) = (r/r_t)^(-a) / (1 + (r/r_t)^b)^(c/b)    (eq. 4)
#   t_cool(r) = (x + Tmin/T0) / (x + 1),  x = (r/r_cool)^a_cool  (eq. 5)
#
# These parameters are NOT tabulated in the paper — they are fitted
# individually per cluster in Paper I (Vikhlinin et al. 2005).
# For a fully self-contained implementation, we provide approximate
# temperature profiles using the UNIVERSAL scaled profile (eq. 9):
#   T(r)/Tmg = 1.35 * (x/0.045)^1.9 + 0.45 / ((x/0.045)^1.9 + 1)
#              * 1/(1 + (x/0.6)^2)^0.45
# where x = r/r500.
#
# This is valid for T > 2.5 keV clusters at r > 0.05 r500.
# For the three cool clusters (A262, RXJ1159, MKW4, USGCS152)
# this is approximate.


def universal_temperature_profile(r_kpc, r500_kpc, Tmg_keV):
    """
    Universal scaled temperature profile from eq. (9).
    T(r)/Tmg = 1.35 * (x/0.045)^1.9 + 0.45
                      / ((x/0.045)^1.9 + 1)
               * 1/(1 + (x/0.6)^2)^0.45
    where x = r/r500.

    Returns T in keV.
    """
    r = np.asarray(r_kpc, dtype=float)
    x = r / r500_kpc
    x = np.clip(x, 1e-6, None)

    t1 = (x / 0.045)**1.9
    t_cool_part = (t1 + 0.45) / (t1 + 1.0)
    t_outer = 1.0 / (1.0 + (x / 0.6)**2)**0.45

    return 1.35 * Tmg_keV * t_cool_part * t_outer


# =====================================================================
# Gas density model (eq. 3)
# =====================================================================
def gas_density_npne(r_kpc, params):
    """
    Compute n_p * n_e (cm^-6) from the Vikhlinin model (eq. 3).

    Parameters
    ----------
    r_kpc : array-like, radius in kpc
    params : tuple from GAS_DENSITY_PARAMS

    Returns
    -------
    npne : array, n_p * n_e in cm^-6
    """
    (rdet, n0_1e3, rc, rs, alpha, beta, epsilon,
     n02_1e1, rc2, beta2) = params

    r = np.asarray(r_kpc, dtype=float)
    r = np.clip(r, 0.1, None)  # avoid r=0

    gamma = 3.0  # fixed

    n0 = n0_1e3 * 1e-3   # cm^-3
    n02 = n02_1e1 * 1e-1  # cm^-3

    # First component
    term1 = n0**2 * (r / rc)**(-alpha)
    term2 = (1.0 + (r / rc)**2)**(3.0 * beta - alpha / 2.0)
    term3 = (1.0 + (r / rs)**gamma)**(epsilon / gamma)
    comp1 = term1 / (term2 * term3)

    # Second component (if present)
    if n02 > 0 and rc2 > 0 and beta2 > 0:
        comp2 = n02**2 / (1.0 + (r / rc2)**2)**(3.0 * beta2)
    else:
        comp2 = 0.0

    return comp1 + comp2


def gas_density_rho(r_kpc, params):
    """
    Gas mass density rho_gas in g/cm^3.
    rho_g = 1.252 * m_p * sqrt(n_p * n_e)
    (for cosmic He abundance, Anders & Grevesse 1989)
    """
    npne = gas_density_npne(r_kpc, params)
    return 1.252 * MP_G * np.sqrt(np.clip(npne, 0, None))


def gas_mass_enclosed(r_kpc, params, n_steps=500):
    """
    Enclosed gas mass M_gas(<r) in solar masses.
    Integrates 4*pi*r^2 * rho_gas(r) dr from 0 to r.
    """
    r = np.asarray(r_kpc, dtype=float)
    results = np.zeros_like(r)

    # Use np.trapezoid (numpy >= 2.0) or np.trapz (older)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    for i, rmax in enumerate(r):
        if rmax <= 0:
            results[i] = 0.0
            continue
        r_grid = np.linspace(0.5, rmax, n_steps)  # kpc
        r_cm = r_grid * KPC_CM
        rho = gas_density_rho(r_grid, params)
        # Integrate in cgs: 4*pi*r^2 * rho * dr
        integrand = 4.0 * np.pi * r_cm**2 * rho
        results[i] = _trapz(integrand, x=r_cm) / MSUN

    return results


# =====================================================================
# Hydrostatic mass (eq. 7)
# =====================================================================
def hydrostatic_mass(r_kpc, r500_kpc, Tmg_keV, params, dr_frac=0.02):
    """
    Total gravitating mass from hydrostatic equilibrium (eq. 7):

    M(r) = -3.71e13 Msun * T(r)[keV] * r[Mpc]
            * (d ln rho_g / d ln r + d ln T / d ln r)

    Parameters
    ----------
    r_kpc : array-like, radii in kpc
    r500_kpc : float, r500 in kpc
    Tmg_keV : float, gas mass-weighted temperature
    params : tuple, gas density params

    Returns
    -------
    M_tot : array, total mass in solar masses (at each r)
    """
    r = np.asarray(r_kpc, dtype=float)
    r = np.clip(r, 1.0, None)
    M = np.zeros_like(r)

    for i, ri in enumerate(r):
        dr = ri * dr_frac
        r_lo = ri - dr / 2.0
        r_hi = ri + dr / 2.0
        if r_lo < 0.5:
            r_lo = 0.5
            r_hi = r_lo + dr

        # Gas density logarithmic derivative
        rho_lo = gas_density_rho(r_lo, params)
        rho_hi = gas_density_rho(r_hi, params)
        if rho_lo > 0 and rho_hi > 0:
            dlnrho_dlnr = np.log(rho_hi / rho_lo) / np.log(r_hi / r_lo)
        else:
            dlnrho_dlnr = 0.0

        # Temperature logarithmic derivative
        T_lo = universal_temperature_profile(r_lo, r500_kpc, Tmg_keV)
        T_hi = universal_temperature_profile(r_hi, r500_kpc, Tmg_keV)
        if T_lo > 0 and T_hi > 0:
            dlnT_dlnr = np.log(T_hi / T_lo) / np.log(r_hi / r_lo)
        else:
            dlnT_dlnr = 0.0

        T_r = universal_temperature_profile(ri, r500_kpc, Tmg_keV)
        r_mpc = ri / 1000.0  # kpc -> Mpc

        M[i] = -3.71e13 * T_r * r_mpc * (dlnrho_dlnr + dlnT_dlnr)

    return M  # in solar masses


# =====================================================================
# Acceleration profiles
# =====================================================================
def acceleration_profiles(r_kpc, cluster_name, n_r=100):
    """
    Compute observed (total) and baryonic acceleration profiles.

    Returns
    -------
    r : array, radii in kpc
    a_obs : array, total gravitational acceleration (m/s^2)
    a_bar : array, baryonic (gas) acceleration (m/s^2)
    M_tot : array, total enclosed mass (Msun)
    M_gas : array, gas enclosed mass (Msun)
    """
    props = CLUSTER_PROPERTIES[cluster_name]
    gas_params = GAS_DENSITY_PARAMS[cluster_name]

    r500 = props[0]  # kpc
    Tmg = props[2]   # keV

    if r500 is None:
        raise ValueError(f"r500 not available for {cluster_name}")

    r = np.asarray(r_kpc, dtype=float)

    # Total mass from hydrostatic equilibrium
    M_tot = hydrostatic_mass(r, r500, Tmg, gas_params)
    M_tot = np.clip(M_tot, 0, None)

    # Gas mass
    M_gas = gas_mass_enclosed(r, gas_params)

    # Accelerations in m/s^2
    # a = G * M / r^2
    r_m = r * KPC_M
    a_obs = G_SI * M_tot * MSUN_KG / r_m**2
    a_bar = G_SI * M_gas * MSUN_KG / r_m**2

    return r, a_obs, a_bar, M_tot, M_gas


def get_cluster_list():
    """Return list of clusters with complete data for analysis."""
    good = []
    for name in CLUSTER_PROPERTIES:
        props = CLUSTER_PROPERTIES[name]
        if props[0] is not None:  # has r500
            good.append(name)
    return sorted(good)


# =====================================================================
# Quick diagnostic
# =====================================================================
if __name__ == "__main__":
    print("Vikhlinin et al. (2006) cluster sample")
    print("=" * 65)
    print(f"{'Cluster':18s} {'z':>6s} {'r500':>6s} {'Tspec':>6s} {'Tmg':>6s} "
          f"{'M500':>7s} {'fg500':>6s}")
    print(f"{'':18s} {'':>6s} {'kpc':>6s} {'keV':>6s} {'keV':>6s} "
          f"{'1e14':>7s} {'':>6s}")
    print("-" * 65)

    for name in sorted(CLUSTER_PROPERTIES.keys()):
        z = CLUSTER_REDSHIFTS.get(name, float('nan'))
        props = CLUSTER_PROPERTIES[name]
        r500 = props[0] if props[0] else 0
        Tspec = props[1]
        Tmg = props[2]
        M500 = props[5] if props[5] else float('nan')
        fg500 = props[7] if props[7] else float('nan')
        print(f"{name:18s} {z:6.4f} {r500:6.0f} {Tspec:6.2f} {Tmg:6.2f} "
              f"{M500:7.2f} {fg500:6.3f}")

    print()
    print("Clusters with complete profiles:", get_cluster_list())

    # Quick test: compute profiles for A1795
    print()
    print("Test: A1795 acceleration profile")
    r_test = np.array([50, 100, 200, 500, 1000])
    r, a_obs, a_bar, M_tot, M_gas = acceleration_profiles(r_test, "A1795")
    print(f"{'r(kpc)':>8s} {'a_obs(m/s2)':>12s} {'a_bar(m/s2)':>12s} "
          f"{'M_tot(Msun)':>12s} {'M_gas(Msun)':>12s} {'ratio':>6s}")
    for i in range(len(r)):
        ratio = a_obs[i] / a_bar[i] if a_bar[i] > 0 else float('nan')
        print(f"{r[i]:8.0f} {a_obs[i]:12.4e} {a_bar[i]:12.4e} "
              f"{M_tot[i]:12.4e} {M_gas[i]:12.4e} {ratio:6.1f}")
