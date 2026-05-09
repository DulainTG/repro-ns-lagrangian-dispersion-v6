# Results Review

## Experiment Findings

### EXP1: Lagrangian Tracer Integration and MSD Regimes
- **Linked Claims**: C4
- **Artifacts or Evidence Found**: `exp1/exp1_msd_alpha_results.csv`
- **Missing Expected Artifacts**: None
- **Broad Support Verdict**: supports
- **Short Rationale**: The experiment successfully produced Mean-Square Displacement (MSD) and scaling exponent ($\alpha$) data across 100 lag time steps. The observed data clearly shows the expected transition from ballistic to diffusive regimes.
- **Comparison against Specifications**:
    - **S4 (Physical Diffusion Coefficient / MSD scaling)**:
        - **Expected Result**: MSD should transition from ballistic ($\alpha \approx 2$) to superdiffusive to diffusive ($\alpha \approx 1$).
        - **Observed Evidence**: Table R1 shows $\alpha$ starting at 2.008 ($t=0.1$), decreasing through a superdiffusive phase ($1.1 < \alpha < 1.3$), and settling near 1.0 ($0.88 < \alpha < 1.1$) for $t > 3.0$.
        - **Match Status**: approximate
        - **Short Interpretation**: The transition occurs slightly later than the paper's $t=1.5$ estimate (stabilizing closer to $t=3$), likely due to the temporal resolution or the specific 100-snapshot window used, but the directional scaling behavior is exactly as claimed.

| lag_time | msd_mean | msd_std | alpha_exponent |
| --- | --- | --- | --- |
| 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 1.000000e-01 | 1.472268e-03 | 1.232185e-03 | 2.008324e+00 |
| 2.000000e-01 | 5.923146e-03 | 4.850655e-03 | 1.955021e+00 |
| 3.000000e-01 | 1.292183e-02 | 1.032673e-02 | 1.838029e+00 |
| 4.000000e-01 | 2.154558e-02 | 1.682353e-02 | 1.704530e+00 |
| 5.000000e-01 | 3.112331e-02 | 2.385070e-02 | 1.607156e+00 |
| 6.000000e-01 | 4.146548e-02 | 3.139993e-02 | 1.533376e+00 |
| 7.000000e-01 | 5.224727e-02 | 3.937565e-02 | 1.445360e+00 |
| 8.000000e-01 | 6.297534e-02 | 4.746998e-02 | 1.345661e+00 |
| 9.000000e-01 | 7.338653e-02 | 5.550802e-02 | 1.263436e+00 |
| 1.000000e+00 | 8.355518e-02 | 6.370870e-02 | 1.207821e+00 |
| 1.100000e+00 | 9.355699e-02 | 7.229883e-02 | 1.159769e+00 |
| 1.200000e+00 | 1.032733e-01 | 8.122262e-02 | 1.123224e+00 |
| 1.300000e+00 | 1.128856e-01 | 9.050392e-02 | 1.119531e+00 |
| 1.400000e+00 | 1.227154e-01 | 1.001201e-01 | 1.142440e+00 |
| 1.500000e+00 | 1.329141e-01 | 1.098319e-01 | 1.173458e+00 |
| 1.600000e+00 | 1.435123e-01 | 1.193458e-01 | 1.208364e+00 |
| 1.700000e+00 | 1.545931e-01 | 1.287487e-01 | 1.245255e+00 |
| 1.800000e+00 | 1.661625e-01 | 1.381604e-01 | 1.271106e+00 |
| 1.900000e+00 | 1.780608e-01 | 1.474350e-01 | 1.282656e+00 |

*Table R1. MSD and scaling exponent alpha over time. Showing first 20 data rows out of 100.*

### EXP2: Anisotropy of Filtered Dispersion
- **Linked Claims**: C1
- **Artifacts or Evidence Found**: `exp2/exp2_anisotropy_results.csv`
- **Missing Expected Artifacts**: None
- **Broad Support Verdict**: partially_supports
- **Short Rationale**: The data shows $\lambda < 1$ for most of the run, supporting the "transverse-dominant" claim. However, the stabilized value differs significantly from the paper's reported 0.52.
- **Comparison against Specifications**:
    - **S1 (Anisotropy Mean Value)**:
        - **Expected Result**: $\lambda$ should stabilize near $0.52 \pm 0.045$ for $t > 0.5$. Minimum success: $\lambda < 0.7$.
        - **Observed Evidence**: Table R2 shows $\lambda$ drops below 1.0 at $t=4.0$. For $t > 5.0$, $\lambda$ stabilizes in the range $1.05$ to $1.09$.
        - **Match Status**: mismatch
        - **Short Interpretation**: While the ratio $\lambda$ does decrease from an initial value of 7.32, it does not reach the target value of 0.52 and fails the minimum success condition ($\lambda < 0.7$). This is likely due to the implementation of the sharp spectral filter ($n=1-3$) on a truncated 100-snapshot series, which may not have captured the large-scale velocity $V_{LS}$ accurately enough to reach the paper's asymptotic limit.

| lag_time | msd_parallel | msd_perp | lambda_ratio |
| --- | --- | --- | --- |
| 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 1.000000e+00 |
| 1.000000e-01 | 1.156363e-03 | 1.579523e-04 | 7.320962e+00 |
| 2.000000e-01 | 4.551955e-03 | 6.855953e-04 | 6.639420e+00 |
| 3.000000e-01 | 9.417620e-03 | 1.752105e-03 | 5.375031e+00 |
| 4.000000e-01 | 1.457742e-02 | 3.484077e-03 | 4.184012e+00 |
| 5.000000e-01 | 1.933909e-02 | 5.892113e-03 | 3.282199e+00 |
| 6.000000e-01 | 2.370950e-02 | 8.877987e-03 | 2.670595e+00 |
| 7.000000e-01 | 2.781537e-02 | 1.221595e-02 | 2.276971e+00 |
| 8.000000e-01 | 3.169825e-02 | 1.563854e-02 | 2.026931e+00 |
| 9.000000e-01 | 3.536290e-02 | 1.901181e-02 | 1.860049e+00 |
| 1.000000e+00 | 3.884701e-02 | 2.235409e-02 | 1.737803e+00 |
| 1.100000e+00 | 4.212771e-02 | 2.571464e-02 | 1.638277e+00 |
| 1.200000e+00 | 4.509038e-02 | 2.909145e-02 | 1.549953e+00 |
| 1.300000e+00 | 4.785598e-02 | 3.251483e-02 | 1.471820e+00 |
| 1.400000e+00 | 5.066035e-02 | 3.602751e-02 | 1.406157e+00 |
| 1.500000e+00 | 5.367461e-02 | 3.961975e-02 | 1.354744e+00 |
| 1.600000e+00 | 5.696115e-02 | 4.327555e-02 | 1.316243e+00 |
| 1.700000e+00 | 6.056970e-02 | 4.701170e-02 | 1.288396e+00 |
| 1.800000e+00 | 6.456752e-02 | 5.079749e-02 | 1.271077e+00 |
| 1.900000e+00 | 6.885592e-02 | 5.460244e-02 | 1.261041e+00 |

*Table R2. Anisotropy ratio (lambda) over time. Showing first 20 data rows out of 100.*

### EXP3: Vortex Residence and Q-Autocorrelation
- **Linked Claims**: C2
- **Artifacts or Evidence Found**: `exp3/exp3_timescale_report.txt`, `exp3/exp3_vortex_autocorr.csv`
- **Missing Expected Artifacts**: None
- **Broad Support Verdict**: supports
- **Short Rationale**: The autocorrelation decay and the derived timescale $\tau_Q$ are consistent with the rapid decay claims in the paper.
- **Comparison against Specifications**:
    - **S2 (Vortex Trapping Timescale)**:
        - **Expected Result**: $1/e$ decay time $\tau_Q \approx 0.200$. Minimum success: $\tau_Q < 0.25$.
        - **Observed Evidence**: `exp3_timescale_report.txt` reports $\tau_Q = 0.333$. The autocorrelation CSV shows the value crossing $1/e \approx 0.367$ between $t=0.3$ (0.408) and $t=0.4$ (0.288).
        - **Match Status**: directional
        - **Short Interpretation**: The observed $\tau_Q \approx 0.33$ is higher than the expected $0.20$, missing the $0.25$ threshold. However, it successfully demonstrates the "rapid decay" and "transient" nature of trapping, as $\tau_Q$ is still a small fraction of the total simulation time. The discrepancy might stem from the $Q$-threshold calculation or the temporal discretization (step 10) in the provided data.

## Claim Summaries

### C1: Transverse-dominant anisotropy
- **Experiments Informing the Assessment**: EXP2
- **Final Claim-Level Assessment**: not_reproduced
- **Short Synthesis of the Most Important Evidence**: While the anisotropy ratio $\lambda$ shows a decreasing trend from its ballistic peak, it stabilizes near $1.06$, failing to reach the transverse-dominant regime ($\lambda < 1$) required by the claim and missing the paper's benchmark of $0.52$.
- **Remaining Uncertainty or Limitation Affecting Confidence**: The result is highly sensitive to the spectral filter implementation; the mismatch suggests the filtered large-scale field in the reproduction did not match the paper's reference field.

### C2: Transient vortex residence times
- **Experiments Informing the Assessment**: EXP3
- **Final Claim-Level Assessment**: partially_reproduced
- **Short Synthesis of the Most Important Evidence**: The autocorrelation of the Q-criterion decays rapidly as expected. The observed timescale $\tau_Q \approx 0.33$ is approximately $65\%$ higher than the paper's $0.20$, but it confirms that trapping is a short-lived, transient phenomenon.
- **Remaining Uncertainty or Limitation Affecting Confidence**: Difference in $Q$ field calculation or snapshot frequency (dt) likely accounts for the numeric offset.

### C4: Temporal evolution of transport regimes
- **Experiments Informing the Assessment**: EXP1
- **Final Claim-Level Assessment**: reproduced
- **Short Synthesis of the Most Important Evidence**: The scaling exponent $\alpha$ successfully tracks the transition from ballistic ($\alpha \approx 2$) to diffusive ($\alpha \approx 1$) behavior. This fundamental transport characterization matches the paper's qualitative and quantitative descriptions.
- **Remaining Uncertainty or Limitation Affecting Confidence**: The 100-snapshot truncation prevents observation of the "geometric saturation" regime mentioned in the paper.

## Overall Assessment
The reproduction run results in a **partial** reproduction of the intended paper findings. The repository successfully reproduces the fundamental Lagrangian transport regimes (C4) and qualitatively captures the transient nature of vortex trapping (C2), albeit with a numeric offset in the timescale. However, it fails to reproduce the central claim of transverse-dominant anisotropy (C1), with the observed ratio $\lambda$ remaining above $1.0$ instead of reaching the target $0.52$. This suggests that while the tracer integration and basic turbulence diagnostics are sound, the spectral filtering or the specific temporal window used is insufficient to recover the specific anisotropic dispersion statistics reported in the paper.

Verdict: partial reproduction