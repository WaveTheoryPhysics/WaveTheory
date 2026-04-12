"""
wt_vs_brouwer.py
================
Load Brouwer+2021 (KiDS-1000) ESD data and overlay WT prediction.

USAGE
-----
1. Download Sersic-split ESD files from:
     https://kids.strw.leidenuniv.nl/sciencedata.php
   Place Fig-*.txt files in the same folder as this script.

2. Run:  python3 wt_vs_brouwer.py
   Output figures appear in ./out/
"""

import pathlib
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / 'data'          # B21 .txt files go here
OUT_DIR  = pathlib.Path(__file__).parent / 'out'
OUT_DIR.mkdir(exist_ok=True)

# ── Physical constants ─────────────────────────────────────────────────────────
G_SI  = 6.674e-11
G_PC  = 4.52e-30       # pc^3 / (Msun s^2)  -- B21 Eq.7 convention
C_SI  = 2.998e8
PI    = np.pi
KPC2M = 3.0857e19
MPC2M = 3.0857e22
PC2M  = 3.0857e16
MSUN  = 1.989e30

# ── WT constants ───────────────────────────────────────────────────────────────
R_UNIVERSE = 4.286e26
A_STAR     = C_SI**2 / (PI * R_UNIVERSE)
KAPPA      = 1.0 + 1.0/PI**2

# A_morph tertile medians from SPARC 175-galaxy fit
ALPHA_LOW  = KAPPA * 0.630    # compact / ETG-like
ALPHA_MID  = KAPPA * 1.070
ALPHA_HIGH = KAPPA * 1.638    # diffuse / LTG-like

print(f"a_* = {A_STAR:.4e} m/s^2")
print(f"alpha_0: Low={ALPHA_LOW:.4f}, Mid={ALPHA_MID:.4f}, High={ALPHA_HIGH:.4f}")

# ── B21 file loader ─────────────────────────────────────────────────────────
def load_esd(filepath):
    """
    Load B21 ESD file, apply bias correction per README:
      ESD_corrected = ESD_t / bias
      error_corrected = error / bias
    """
    data = np.loadtxt(filepath, comments='#')
    R     = data[:, 0]
    ESD_t = data[:, 1]
    ESD_x = data[:, 2]
    err   = data[:, 3]
    bias  = data[:, 4]
    return dict(
        R      = R,
        ESD_t  = ESD_t / bias,
        ESD_x  = ESD_x / bias,
        error  = np.abs(err / bias),   # abs: errorbar requires non-negative
        bias   = bias,
    )

def ESD_to_gobs(ESD):
    """ESD (Msun/pc^2) -> g_obs (m/s^2) via B21 Eq.7."""
    return 4.0 * G_PC * ESD * PC2M

# ── WT master formula ───────────────────────────────────────────────────────
def g_eff_wt(g_bar, alpha0):
    g_bar = np.asarray(g_bar, dtype=float)
    xi    = np.sqrt(np.clip(g_bar, 0, None) / A_STAR)
    return A_STAR * xi * (xi**3 + alpha0) / (1.0 + xi**2)

# ── File discovery ──────────────────────────────────────────────────────────
def find_esd_files():
    files = sorted(DATA_DIR.glob('Fig-*.txt'))
    return {f.stem: f for f in files}

def extract_bin_number(name):
    """Extract trailing bin number from various filename patterns."""
    # Handles: bin-1, bin_1, _1 at end
    for sep in ('bin-', 'bin_', '_'):
        if sep in name:
            try:
                return int(name.split(sep)[-1])
            except ValueError:
                continue
    return 0

def is_covmatrix(name):
    return 'cov' in name.lower() or 'matrix' in name.lower()

# ── WT RAR curve set ────────────────────────────────────────────────────────
WT_CURVES = [
    (ALPHA_LOW,  'royalblue',   '--', r'Low $A_{\rm morph}$ (ETG-like)'),
    (ALPHA_MID,  'seagreen',    '-.',  r'Mid $A_{\rm morph}$'),
    (ALPHA_HIGH, 'darkorange',  '-',  r'High $A_{\rm morph}$ (LTG-like)'),
]

G_BAR_WT = np.geomspace(1e-14, 1e-9, 300)

# ── Plotting helpers ────────────────────────────────────────────────────────
def add_wt_rar_curves(ax):
    for alpha0, col, ls, lbl in WT_CURVES:
        ax.loglog(G_BAR_WT, g_eff_wt(G_BAR_WT, alpha0),
                  color=col, lw=2.0, ls=ls, label=lbl)
    ax.loglog(G_BAR_WT, G_BAR_WT,
              'k:', lw=0.8, alpha=0.4, label='Newtonian')
    ax.loglog(G_BAR_WT, np.sqrt(G_BAR_WT * A_STAR),
              'k--', lw=0.8, alpha=0.4, label='Deep MOND')
    ax.axvline(A_STAR, color='gray', lw=0.7, ls='--', alpha=0.5)
    ax.text(A_STAR * 1.3, 2e-11, r'$a_*$', fontsize=9, color='gray')

def format_rar_ax(ax, title):
    ax.set_xlabel(r'$g_{\rm bar}$ [m s$^{-2}$]', fontsize=11)
    ax.set_ylabel(r'$g_{\rm obs}$ [m s$^{-2}$]', fontsize=11)
    ax.set_xlim(1e-14, 1e-9)
    ax.set_ylim(1e-13, 1e-8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.2)

# ── Main plots ──────────────────────────────────────────────────────────────
def plot_wt_only():
    fig, ax = plt.subplots(figsize=(7, 6))
    add_wt_rar_curves(ax)
    format_rar_ax(ax, 'WT RAR prediction by morphology\n'
                      '(waiting for Brouwer+2021 data files)')
    out = OUT_DIR / 'wt_rar_prediction_only.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_sersic_comparison(sersic_files):
    # Drop covariance matrix files
    data_files = {k: v for k, v in sersic_files.items() if not is_covmatrix(k)}
    bins = sorted(data_files.items(), key=lambda x: extract_bin_number(x[0]))

    n = len(bins)
    if n == 0:
        print("No non-covariance Sersic files found.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6.5*n, 5.5), squeeze=False)
    axes = axes[0]

    for ax, (name, fpath) in zip(axes, bins):
        d = load_esd(fpath)

        # Detect format: RAR files have R in m/s^2 (< 1e-6), ESD files in Mpc (> 0.01)
        is_rar = d['R'].max() < 1e-6

        bin_num = extract_bin_number(name)
        # B21: bin 1 = low Sersic = LTG; bin 2 = high Sersic = ETG
        type_label = {1: 'Low Sérsic (LTG)', 2: 'High Sérsic (ETG)'}.get(bin_num, f'bin {bin_num}')

        if is_rar:
            g_bar_obs = d['R']
            g_obs     = ESD_to_gobs(d['ESD_t'])
            g_obs_err = ESD_to_gobs(d['error'])

            # Only plot positive g_obs points
            ok = g_obs > 0
            ax.errorbar(g_bar_obs[ok], g_obs[ok], yerr=g_obs_err[ok],
                        fmt='ko', ms=5, capsize=3, elinewidth=1.0,
                        label='B21 data', zorder=5)
            add_wt_rar_curves(ax)
            format_rar_ax(ax, f'B21 Fig-8 Sersic bin {bin_num}: {type_label}')

        else:
            R_kpc = d['R'] * 1e3
            DS    = d['ESD_t']
            DS_err= d['error']
            ok    = DS > 0
            ax.errorbar(R_kpc[ok], DS[ok], yerr=DS_err[ok],
                        fmt='ko', ms=4, capsize=2, elinewidth=0.8,
                        label='B21 data', zorder=5)
            for alpha0, col, ls, lbl in WT_CURVES:
                # Rough WT Delta_Sigma: use g_eff / (2*pi*G) as surface density proxy
                # Full Abel integral is in wt_lensing_abel.py
                pass
            ax.set_xlabel(r'$R_\perp$ [kpc]', fontsize=11)
            ax.set_ylabel(r'$\Delta\Sigma\ [h_{70}\,M_\odot\,{\rm pc}^{-2}]$', fontsize=11)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(f'B21 Sersic bin {bin_num}: {type_label}', fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.suptitle(
        'WT morphology prediction vs Brouwer+2021 (KiDS-1000) Sersic split\n'
        r'Low $A_{\rm morph}$ $\to$ ETG; High $A_{\rm morph}$ $\to$ LTG  |  '
        r'Predicted ratio: $\alpha_0^{\rm High}/\alpha_0^{\rm Low} = 2.60$',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    out = OUT_DIR / 'wt_vs_brouwer_sersic.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_ratio_panel(sersic_files):
    """
    Ratio plot: g_obs(ETG) / g_obs(LTG) vs g_bar.
    WT predicts this ratio ~ ALPHA_LOW / ALPHA_HIGH at fixed g_bar in shell regime.
    """
    data_files = {k: v for k, v in sersic_files.items() if not is_covmatrix(k)}
    by_bin = {extract_bin_number(k): load_esd(v)
              for k, v in data_files.items()}

    if 1 not in by_bin or 2 not in by_bin:
        print("Need both bin 1 and bin 2 for ratio plot.")
        return

    d1 = by_bin[1]   # LTG (low Sersic)
    d2 = by_bin[2]   # ETG (high Sersic)

    is_rar = d1['R'].max() < 1e-6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    if is_rar:
        g_bar1 = d1['R'];  g1 = ESD_to_gobs(d1['ESD_t']); e1 = ESD_to_gobs(d1['error'])
        g_bar2 = d2['R'];  g2 = ESD_to_gobs(d2['ESD_t']); e2 = ESD_to_gobs(d2['error'])

        # Panel 1: both curves on same axes
        ok1 = g1 > 0; ok2 = g2 > 0
        ax1.errorbar(g_bar1[ok1], g1[ok1], yerr=e1[ok1],
                     fmt='bs', ms=5, capsize=2, label='B21: LTG (low Sersic)')
        ax1.errorbar(g_bar2[ok2], g2[ok2], yerr=e2[ok2],
                     fmt='r^', ms=5, capsize=2, label='B21: ETG (high Sersic)')
        add_wt_rar_curves(ax1)
        format_rar_ax(ax1, 'B21 Sersic bins 1 & 2 with WT prediction')

        # Panel 2: ratio ETG/LTG vs g_bar
        # Interpolate onto common g_bar grid
        g_common = np.geomspace(max(g_bar1.min(), g_bar2.min()),
                                min(g_bar1.max(), g_bar2.max()), 20)
        g1_interp = np.interp(g_common, g_bar1, g1)
        g2_interp = np.interp(g_common, g_bar2, g2)
        ratio_obs = g2_interp / g1_interp   # ETG / LTG

        ax2.semilogx(g_common, ratio_obs, 'ko-', ms=5, lw=1.5,
                     label='B21: ETG / LTG ratio')

        # WT predicted ratio: g_eff(ETG) / g_eff(LTG) at each g_bar
        ratio_low_high = g_eff_wt(G_BAR_WT, ALPHA_LOW) / g_eff_wt(G_BAR_WT, ALPHA_HIGH)
        ax2.semilogx(G_BAR_WT, ratio_low_high, 'royalblue', lw=2.0,
                     label=r'WT: Low $A_{\rm morph}$ / High $A_{\rm morph}$')
        ax2.axhline(1.0, color='k', lw=0.8, ls=':')
        ax2.axvline(A_STAR, color='gray', lw=0.7, ls='--', alpha=0.5)
        ax2.text(A_STAR*1.3, 1.05, r'$a_*$', fontsize=9, color='gray')
        ax2.set_xlabel(r'$g_{\rm bar}$ [m s$^{-2}$]', fontsize=11)
        ax2.set_ylabel('ETG / LTG ratio', fontsize=11)
        ax2.set_title('Morphology split ratio\n'
                      r'WT asymptotic: $\alpha_0^{\rm Low}/\alpha_0^{\rm High} = %.2f$'
                      % (ALPHA_LOW/ALPHA_HIGH), fontsize=10)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)
        ax2.set_xlim(1e-14, 1e-9)

    fig.suptitle('Brouwer+2021 KiDS-1000 Sersic split vs Wave Theory prediction',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = OUT_DIR / 'wt_vs_brouwer_ratio.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def main():
    esd_files = find_esd_files()

    if not esd_files:
        print(f"No B21 files found in {DATA_DIR}")
        print("Download from: https://kids.strw.leidenuniv.nl/sciencedata.php")
        plot_wt_only()
        return

    print(f"Found {len(esd_files)} B21 files:")
    for name in sorted(esd_files):
        print(f"  {name}")

    sersic_files = {k: v for k, v in esd_files.items()
                    if ('sersic' in k.lower() or 'Sersic' in k)
                    and not is_covmatrix(k)}
    all_sersic   = {k: v for k, v in esd_files.items()
                    if 'sersic' in k.lower() or 'Sersic' in k}

    if sersic_files:
        print(f"\nPlotting {len(sersic_files)} Sersic data files...")
        plot_sersic_comparison(all_sersic)  # pass all including cov for routing
        plot_ratio_panel(all_sersic)
    else:
        print("No Sersic files found — plotting WT prediction only.")
        plot_wt_only()

    print()
    print("Morphology mapping:")
    print(f"  B21 bin 1 (low Sersic / LTG) <-> High A_morph, alpha0 = {ALPHA_HIGH:.4f}")
    print(f"  B21 bin 2 (high Sersic / ETG) <-> Low A_morph,  alpha0 = {ALPHA_LOW:.4f}")
    print(f"  WT predicted ETG/LTG ratio in shell regime: {ALPHA_LOW/ALPHA_HIGH:.3f}")
    print(f"  Brouwer+2021 observed: ~1.5-2.5x (6sigma detection)")


if __name__ == '__main__':
    plot_wt_only()
    main()
