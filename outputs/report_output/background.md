# Background

## Introduction to the Paper
The paper investigates the relationship between large-scale energy injection, coherent structures, and particle transport in three-dimensional solenoidal turbulence. Using high-resolution direct numerical simulations (DNS) of subsonic, isothermal flow, the study analyzes the trajectories of thousands of passive Lagrangian tracers to characterize the statistical properties of their dispersion.

The primary objective is to demonstrate how the rotational nature of solenoidal forcing imposes a directional preference on transport and to determine if temporary trapping within coherent vortices leads to anomalous diffusion. The authors aim to show that while solenoidal forcing induces a persistent transverse-dominant anisotropy, the inherent instabilities of 3D vortices ensure that trapping is too transient to generate long-term memory, leading to an eventual return to classical, albeit anisotropic, diffusion.

## Scope of Reproducibility
The scope of this reproduction is limited to the core claims regarding transport regimes, directional anisotropy, and vortex trapping timescales as exercised by the must-run experiments.

- **C4: Temporal evolution of transport regimes**: This claim states that the Mean-Square Displacement (MSD) transitions from ballistic ($\alpha \approx 2$) to superdiffusive and finally diffusive ($\alpha \approx 1$) regimes. It is tested by **EXP1**, which is expected to show the scaling exponent $\alpha$ descending from approximately 2 to 1.
- **C1: Transverse-dominant anisotropic dispersion**: This claim posits that dispersion perpendicular to the local large-scale velocity field systematically exceeds parallel dispersion ($\lambda < 1$). It is tested by **EXP2**, which expects the anisotropy ratio $\lambda$ to stabilize near $0.52 \pm 0.045$.
- **C2: Transient vortex residence times**: This claim asserts that vortex trapping events are brief, lasting only about 7% of a large-eddy turnover time. It is tested by **EXP3**, which expects the $1/e$ decay time ($\tau_Q$) of the Q-criterion Lagrangian autocorrelation to be approximately 0.200.

To see the full claims and their relationships, see Figure 1 below.

![Figure 1. Scope graph showing the full tested claims and their relationships. To interpret the scope graph, refer to the scope graph key in the Appendix.](../../../../../run_20260508_172703/scoping_output/scoping_visualisation/scope_graph.png)

*Figure 1. Scope graph showing the full tested claims and their relationships. To interpret the scope graph, refer to the scope graph key in the Appendix.*

## Methodology

### Datasets

#### Solenoidal Turbulence DNS Velocity Snapshots (DS1)
This dataset serves as the primary benchmark input for all must-run experiments. It consists of VTK snapshots representing 3D scalar and vector fields of subsonic isothermal turbulence on a periodic $L=1$ cubic grid. The snapshots provide the velocity field $v(x,t)$ used for tracer integration and for deriving auxiliary fields like vorticity, the Q-criterion, and the filtered large-scale velocity field.

**Reproduction Note:** The original paper utilizes a sequence of 200 snapshots. The available reproduction dataset is a reduced subset consisting of 100 snapshots (indices 18903 to 19893 with a step of 10). This 50% reduction in temporal coverage or resolution may impact the convergence of late-time statistics and the observation of geometric saturation artifacts.

### Must-Run Experiments

#### EXP1: Lagrangian Tracer Integration and MSD Regimes
This experiment tests **C4** by reconstructing the trajectories of 8,000 passive tracers. Tracers are seeded at random uniform positions and integrated using a fourth-order Runge-Kutta (RK4) scheme with 10 sub-steps between snapshots. Trilinear interpolation is used to determine velocity at off-grid positions, and periodic boundary conditions are enforced. The required output is a time-series of the Mean-Square Displacement (MSD) and its local scaling exponent $\alpha(t)$. A successful reproduction will show $\alpha(t)$ transitioning from ballistic ($\alpha \approx 2$) to diffusive ($\alpha \approx 1$) behavior.

#### EXP2: Anisotropy of Filtered Dispersion
This experiment tests **C1** by analyzing directional transport relative to the large-scale flow. It requires applying a sharp spectral Fourier filter to the velocity snapshots, retaining only the driving modes ($n=1-3$) to obtain the large-scale velocity field $V_{LS}$. Tracer displacements are then decomposed into components parallel and perpendicular to $V_{LS}$ to calculate the dynamic anisotropy ratio $\lambda(t)$. The required artifact is a table of $\lambda(t)$ values. Success is defined by $\lambda(t)$ dropping below 1.0 after the initial ballistic phase and targeting the paper's value of approximately 0.52.

#### EXP3: Vortex Residence and Q-Autocorrelation
This experiment tests **C2** by quantifying the duration of vortex trapping events. It involves computing the Q-criterion field for all snapshots to identify regions where rotation dominates strain ($Q > 0$). The Q-value is sampled along the tracer trajectories generated in EXP1 to compute the Lagrangian autocorrelation of the Q-signal. The experiment must output the normalized autocorrelation function and the characteristic timescale $\tau_Q$ (where the function decays to $1/e$). A successful reproduction will yield a $\tau_Q$ of approximately 0.200, confirming the transient nature of trapping.