# API Documentation

This document describes the primary modules and classes in the repository.

## `src.dynamics`
Core Lagrangian integration and tracking logic.

### `VtkSnapshotLoader` (Protocol)
Defines the interface for loading fluid fields.

### `AthenaVtkLoader`
Implementation of `VtkSnapshotLoader` for Athena++ VTK files.

### `TracerIntegrator`
Handles the integration of particle trajectories within the fluid domain. Supports RK4 and Euler schemes.

## `src.analysis`
Physics-based analysis routines.

### `mmsd_analysis`
Computes Mean Square Displacement and extracts diffusion exponents.

### `AnisotropyAnalyzer`
Calculates parallel and perpendicular dispersion relative to local or mean magnetic/velocity fields.

### `VortexTrappingAnalyzer`
Analyzes the Q-criterion along trajectories and calculates residence times and autocorrelations.

## `src.experiments`
High-level experiment orchestration.

### `ReproductionRunSettings`
Data structure for holding simulation parameters.

### `BaseExperiment`
Abstract base class for all experiments, providing `prepare`, `run`, and `save_artifacts` interfaces.

### `MsdRegimeExperiment` (EXP1)
Orchestrates the baseline dispersion study.

### `FilteredAnisotropyExperiment` (EXP2)
Orchestrates the scale-dependent anisotropy research.

### `VortexTrappingExperiment` (EXP3)
Orchestrates the study of vortex interaction and trapping.

## `src.io`
Input/Output utilities for handling simulation data.

### `VtkPathResolver`
Maps indices to physical file paths.

### `VelocitySnapshotStream`
Provides a filtered stream of velocity fields for analysis.
