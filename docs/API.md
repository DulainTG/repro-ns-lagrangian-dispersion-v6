# API Documentation

This document describes the primary modules and classes in the repository.

## `src.dynamics`
Core Lagrangian integration and tracking logic.

### `AthenaVtkLoader`
Implementation for loading and parsing Athena++ VTK files, extracting headers, grid metadata, and binary data.

### `TracerIntegrator` (via `RK4SolverIterationRoutine`)
Handles the integration of particle trajectories within the fluid domain. Supports RK4 and Euler schemes with sub-stepping between snapshots.

### `TracerStateBuffer`
Manages the current state and history of multiple tracers, handling periodic boundary conditions and artifact commit.

## `src.analysis`
Physics-based analysis routines.

### `MsdCalculator`
Computes Mean Square Displacement (MSD) for an ensemble of trajectories.

### `AlphaExponentEstimator`
Extracts diffusion exponents from the MSD time series to identify transport regimes.

### `AnisotropyAnalyzer`
Calculates parallel and perpendicular dispersion relative to the local filtered large-scale velocity field.

### `VorticityTimescaleDeriver`
Calculates the characteristic timescale \(\tau_Q\) from the Lagrangian Q-criterion autocorrelation.

## `src.experiments`
High-level experiment orchestration.

### `ReproductionRunSettings`
Data structure for holding simulation parameters (number of tracers, sub-steps, interpolation, etc.).

### `BaseExperiment`
Abstract base class for all experiments, providing `prepare`, `run`, and `save_artifacts` interfaces.

### `MsdRegimeExperiment` (EXP1)
Reproduces Claim C4 regarding MSD transitions through ballistic and diffusive regimes.

### `FilteredAnisotropyExperiment` (EXP2)
Reproduces Claim C1 regarding transverse-dominant anisotropic dispersion in filtered fields.

### `VortexTrappingExperiment` (EXP3)
Reproduces Claim C2 regarding the transience of vortex trapping events using Q-autocorrelation.

## `src.io`
Input/Output utilities for handling simulation data.

### `VtkPathResolver`
Maps simulation snapshot indices to physical file paths in the dataset directory.

### `AthenaVtkLoader`
Orchestrates the loading of metadata and binary fields from Athena++ VTK files.
