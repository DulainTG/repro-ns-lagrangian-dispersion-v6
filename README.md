# 3D Solenoidal Turbulence Reproduction: Transverse-Dominant Anisotropic Dispersion and Transient Trapping

This repository provides a standalone reproduction of the numerical experiments and analysis for the study of lagrangian tracer dynamics in 3D solenoidal turbulence, specifically focusing on transverse-dominant anisotropic dispersion and transient trapping.

## Overview

The codebase implements a complete pipeline for:
- Loading and parsing Athena++ VTK fluid snapshots (DS1 dataset).
- Performing high-fidelity Lagrangian tracer integration (RK4/Euler).
- Analyzing Mean Square Displacement (MSD) and anomalous diffusion exponents (EXP1).
- Computing scale-dependent anisotropy in filtered dispersion (EXP2).
- Quantifying vortex residence times and Q-criterion autocorrelation (EXP3).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The canonical entry point is `main.py`.

### Running All Experiments (Full Benchmark)
To run the full reproduction suite as described in the paper:
```bash
python main.py full-benchmark --raw-data-dir /path/to/raw_data --mode full
```

### Running a Single Experiment
You can run experiments individually (`EXP1`, `EXP2`, or `EXP3`):
```bash
python main.py EXP1 --raw-data-dir /path/to/raw_data --mode full
```

### Smoke Mode
For rapid validation or CI, use `--mode smoke`, which reduces the number of tracers and snapshots:
```bash
python main.py full-benchmark --raw-data-dir /path/to/raw_data --mode smoke
```

### CLI Arguments
- `experiment`: `EXP1`, `EXP2`, `EXP3`, or `full-benchmark`.
- `--mode`: `full` (default) or `smoke`.
- `--raw-data-dir`: Path to the directory containing `Turb.hydro_w.#####.vtk` files.
- `--output-dir`: Directory where results (CSV, TXT, plots) will be saved.
- `--dataset`: Alias for the dataset (e.g., `DS1`).

### Parameter Overrides
Optional overrides for experimentation:
- `--number-of-tracers`: Total number of particles.
- `--integration-scheme`: `RK4` or `Euler`.
- `--sub-steps`: Integrator steps between snapshots.
- `--interpolation`: `trilinear` or `nearest`.
- `--filter-min` / `--filter-max`: Wavenumber range for EXP2.

## Testing

Comprehensive tests are provided in the `tests/` directory.

### Running all tests:
```bash
pytest
```

### Running CLI smoke tests:
```bash
pytest tests/test_final_smoke_exp1.py
pytest tests/test_final_smoke_exp2.py
pytest tests/test_final_smoke_exp3.py
pytest tests/test_final_full_benchmark.py
```

## Dataset (DS1)
The reproduction expects Athena++ VTK snapshots. The default sequence for DS1 is indices `18903` to `19893` with a step of 10.

## Artifacts Produced
- **EXP1**: `exp1_msd_alpha_results.csv`
- **EXP2**: `exp2_anisotropy_results.csv`
- **EXP3**: `exp3_vortex_autocorr.csv`, `exp3_timescale_report.txt`
