"""
Shot Noise Example — Applying photon noise to a Prometheus spectrum
===================================================================

Demonstrates all three SNR modes of the shotNoise extension using a
synthetic transmission spectrum (a Gaussian absorption dip).  No actual
Prometheus simulation is required to run this file.

Run from the repository root::

    python -m Prometheus.pythonScripts.examples.shot_noise_example

Output
------
Prints SNR values and noise statistics for each mode, and saves a
three-panel comparison plot to ``shot_noise_example.png``.
"""

import numpy as np
import matplotlib.pyplot as plt

from Prometheus.pythonScripts.shotNoise import (
    SNRModel,
    TransitParams,
    apply_shot_noise,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Create a synthetic "clean" transmission spectrum
# ──────────────────────────────────────────────────────────────────────────────
# Wavelength grid: 300–400 nm with 0.01 nm resolution (UV range)
wavelength_nm = np.arange(300.0, 400.0, 0.01)

# Baseline continuum at 1.0 (no absorption) with a Gaussian dip at 350 nm
# mimicking a single absorption line with ~0.1% depth
dip_center = 350.0   # nm
dip_width  = 0.5     # nm (FWHM ≈ 1.18 nm)
dip_depth  = 1e-3    # fractional depth

clean_spectrum = 1.0 - dip_depth * np.exp(-0.5 * ((wavelength_nm - dip_center) / dip_width)**2)

print("Synthetic spectrum created:")
print(f"  Wavelength range : {wavelength_nm[0]:.1f} – {wavelength_nm[-1]:.1f} nm")
print(f"  Absorption dip   : {dip_depth*100:.2f}% at {dip_center} nm")
print(f"  Grid points      : {len(wavelength_nm)}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Mode 1 — Constant SNR
# ──────────────────────────────────────────────────────────────────────────────
snr_const = SNRModel.constant(snr_per_bin=500.0)

noisy_const, sigma_const = apply_shot_noise(
    wavelength_nm, clean_spectrum, snr_const, seed=42
)

print("Mode 1: Constant SNR")
print(f"  SNR/bin           : {snr_const.snr_at(350.0):.1f}")
print(f"  1σ noise level    : {sigma_const[0]:.2e}")
print(f"  Noise std (actual): {np.std(noisy_const - clean_spectrum):.2e}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Mode 2 — Tabulated SNR with scaling
# ──────────────────────────────────────────────────────────────────────────────
# Suppose an ETC reports SNR at a few blaze wavelengths for a mag-17 star
# with a 900 s exposure.  We want to scale to HD 189733 (mag 9.24) observed
# over a 1.8 hr transit split into 8 phase bins.
etc_wav_nm = [310.0, 330.0, 350.0, 370.0, 390.0]
etc_snr    = [5.0,   12.0,  18.0,  15.0,  8.0]

transit = TransitParams(
    target_mag=9.24,
    transit_duration_hrs=1.8,
    num_bins=8,
)

snr_table = SNRModel.from_table(
    wav_nm=etc_wav_nm,
    snr=etc_snr,
    baseline_mag=17.0,
    baseline_time_hrs=900.0 / 3600.0,
    transit_params=transit,
)

noisy_table, sigma_table = apply_shot_noise(
    wavelength_nm, clean_spectrum, snr_table, seed=42
)

print("Mode 2: Tabulated SNR (scaled)")
print(f"  SNR/bin at 350 nm : {snr_table.snr_at(350.0):.1f}")
print(f"  SNR/bin at 310 nm : {snr_table.snr_at(310.0):.1f}")
print(f"  1σ at dip center  : {sigma_table[np.argmin(np.abs(wavelength_nm - 350))]:.2e}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Mode 3 — UVES JSON (skipped here; shown for reference)
# ──────────────────────────────────────────────────────────────────────────────
# To use Mode 3, point it at a real ETC JSON export:
#
#   snr_json = SNRModel.from_uves_json("uves_etc_output.json", transit)
#   noisy_json, sigma_json = apply_shot_noise(
#       wavelength_nm, clean_spectrum, snr_json, seed=42
#   )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Comparison plot
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- Panel 1: Constant SNR ---
ax = axes[0]
ax.fill_between(wavelength_nm, clean_spectrum - 2*sigma_const,
                clean_spectrum + 2*sigma_const, alpha=0.10, color='C0')
ax.fill_between(wavelength_nm, clean_spectrum - sigma_const,
                clean_spectrum + sigma_const, alpha=0.25, color='C0',
                label='1σ / 2σ bands')
ax.plot(wavelength_nm, noisy_const, linewidth=0.4, color='grey', alpha=0.6, label='Noisy')
ax.plot(wavelength_nm, clean_spectrum, linewidth=1.5, color='C0', label='Clean')
ax.axvline(dip_center, color='red', ls='--', alpha=0.4, label=f'Line at {dip_center} nm')
ax.set_ylabel('Residual Flux dF/F')
ax.set_title(f'Mode 1: Constant SNR = {snr_const._snr_value:.0f}')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 2: Tabulated SNR ---
ax = axes[1]
ax.fill_between(wavelength_nm, clean_spectrum - 2*sigma_table,
                clean_spectrum + 2*sigma_table, alpha=0.10, color='C1')
ax.fill_between(wavelength_nm, clean_spectrum - sigma_table,
                clean_spectrum + sigma_table, alpha=0.25, color='C1',
                label='1σ / 2σ bands')
ax.plot(wavelength_nm, noisy_table, linewidth=0.4, color='grey', alpha=0.6, label='Noisy')
ax.plot(wavelength_nm, clean_spectrum, linewidth=1.5, color='C1', label='Clean')
ax.axvline(dip_center, color='red', ls='--', alpha=0.4, label=f'Line at {dip_center} nm')
ax.set_ylabel('Residual Flux dF/F')
ax.set_xlabel('Wavelength [nm]')
ax.set_title('Mode 2: Tabulated SNR (scaled to HD 189733, 1.8 hr transit, 8 bins)')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('shot_noise_example.png', dpi=150)
print("Plot saved to shot_noise_example.png")
plt.close(fig)
