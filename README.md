# Prometheus-Shot-Noise

> Shot noise injection and SNR estimation for [Prometheus](https://github.com/CrazeXD/Prometheus) transmission spectra.

Prometheus-Shot-Noise is a post-processing extension for the [Prometheus](https://github.com/CrazeXD/Prometheus) radiative transfer code. It adds photon shot noise to simulated transmission spectra and provides flexible signal-to-noise ratio (SNR) modelling sourced from exposure time calculators (ETCs), tabulated curves, or a simple constant value.

The module is **decoupled from Prometheus internals** — it operates on plain NumPy arrays of wavelength and flux, so it slots cleanly into any post-processing pipeline built on top of Prometheus output files.

---

## Features

- **Three SNR source modes:**
  - `constant` — a single SNR/bin applied uniformly across the wavelength grid (e.g. from a quick ETC run at representative conditions)
  - `table` — a user-supplied wavelength–SNR table linearly interpolated onto the simulation grid and rescaled to actual target conditions
  - `json` — a per-pixel SNR curve parsed directly from an ESO ETC JSON export (tested with UVES ETC v2)

- **Automatic SNR scaling** via the standard photon shot-noise law, accounting for target magnitude, stellar flux ratio, transit duration, and number of orbital-phase bins

- **Gaussian noise injection** with optional reproducible seeding

- **Sigma array output** for generating error bars and confidence bands on noisy spectra

- Works with any Prometheus output — no assumptions about absorbing species, wavelength regime, or atmospheric scenario

---

## Installation

Clone this repository alongside your existing Prometheus setup:

```bash
git clone https://github.com/CrazeXD/Prometheus-Shot-Noise.git
```

Place `shotNoise.py` where it can be imported relative to your Prometheus scripts (e.g. inside `Prometheus/pythonScripts/`), or add its directory to your Python path.

**Dependencies:** `numpy` (already required by Prometheus). No additional packages are needed.

---

## Quick Start

```python
from Prometheus.pythonScripts.shotNoise import SNRModel, apply_shot_noise
import numpy as np

# Load your Prometheus output
wavelength_nm = ...   # 1-D array of wavelengths in nanometres
spectrum      = ...   # Corresponding transmission spectrum (dimensionless)

# --- Mode 1: constant SNR ---
snr_model = SNRModel.constant(snr_per_bin=847.0)

noisy_spectrum, sigma = apply_shot_noise(wavelength_nm, spectrum, snr_model, seed=42)
```

---

## SNR Modes in Detail

### 1. Constant SNR

```python
snr_model = SNRModel.constant(snr_per_bin=847.0)
```

A single SNR value is applied at every wavelength point. Useful for quick tests or when your ETC gives a single representative number.

---

### 2. Tabulated SNR

```python
from Prometheus.pythonScripts.shotNoise import SNRModel, TransitParams

transit = TransitParams(
    target_mag=8.5,
    transit_duration_hrs=2.1,
    num_bins=50,
)

snr_model = SNRModel.from_table(
    wav_nm=wav_table,        # array of wavelengths in nm
    snr=snr_table,           # corresponding baseline SNR values
    baseline_mag=10.0,       # magnitude used in ETC run
    baseline_time_hrs=1.0,   # ETC exposure time in hours
    transit_params=transit,
)
```

The table is linearly interpolated onto the simulation grid, then rescaled using the photon shot-noise scaling law:

```
SNR_target = SNR_baseline × √(F_target / F_baseline) × √(t_bin / t_baseline)
```

where the flux ratio is derived from the magnitude difference and the per-bin exposure time is `transit_duration_hrs / num_bins`.

---

### 3. ESO UVES ETC JSON

```python
snr_model = SNRModel.from_uves_json(
    json_path="path/to/uves_etc_output.json",
    transit_params=transit,
)
```

Parses `data.orders[].detectors[].wavelength` and `data.orders[].detectors[].plots.snr.snr` from the ESO ETC v2 JSON schema. Baseline magnitude and exposure time are read automatically from the JSON. The resulting curve is then interpolated and scaled identically to mode 2.

ETC calculators can be found at: https://www.eso.org/observing/etc/

---

## API Reference

### `TransitParams`

```python
TransitParams(target_mag, transit_duration_hrs, num_bins)
```

| Parameter | Type | Description |
|---|---|---|
| `target_mag` | `float` | Apparent magnitude of the science target |
| `transit_duration_hrs` | `float` | Total in-transit duration in hours |
| `num_bins` | `int` | Number of orbital-phase bins |

---

### `SNRModel`

| Method | Description |
|---|---|
| `SNRModel.constant(snr_per_bin)` | Uniform SNR across all wavelengths |
| `SNRModel.from_table(wav_nm, snr, baseline_mag, baseline_time_hrs, transit_params)` | Tabulated wavelength–SNR curve |
| `SNRModel.from_uves_json(json_path, transit_params)` | ESO UVES ETC JSON export |
| `.snr_at(wavelength_nm)` | Returns SNR at a single wavelength |
| `.snr_array(wavelength_nm)` | Returns SNR array over an entire grid |

---

### `apply_shot_noise`

```python
noisy_spectrum, sigma = apply_shot_noise(wavelength_nm, spectrum, snr_model, seed=None)
```

| Parameter | Type | Description |
|---|---|---|
| `wavelength_nm` | `ndarray` | Wavelength grid in nanometres |
| `spectrum` | `ndarray` | Clean transmission spectrum (dimensionless) |
| `snr_model` | `SNRModel` | Noise model |
| `seed` | `int`, optional | Random seed for reproducibility |

Returns `(noisy_spectrum, sigma)` — the noise-injected spectrum and the 1-sigma noise level at each wavelength point.

---

### `scale_snr`

```python
scaled = scale_snr(baseline_snr, baseline_mag, baseline_time_hrs,
                   target_mag, transit_duration_hrs, num_bins)
```

Standalone helper that applies the photon shot-noise scaling law. Useful if you want to rescale an ETC SNR without building a full `SNRModel`.

---

## Relationship to Prometheus

[Prometheus](https://github.com/CrazeXD/Prometheus) (originally by [andreagebek](https://github.com/andreagebek/Prometheus)) is a radiative transfer tool for computing transmission spectra and lightcurves of transiting exoplanets and exomoons. It produces clean, noiseless spectra stored as text output files.

Prometheus-Shot-Noise is a downstream post-processing step: load a Prometheus output file, pass the wavelength and flux arrays into `apply_shot_noise`, and obtain a realistic noisy spectrum ready for comparison with observations or retrieval pipelines.

```
Prometheus simulation
        │
        ▼
  output/*.txt   ──►  apply_shot_noise()  ──►  noisy spectrum + σ
```

---

## License

See the [Prometheus](https://github.com/CrazeXD/Prometheus/blob/main/LICENSE) base repository for license terms (GPL-3.0).
