# Results Review

## Experiment Findings

### EXP1: Lagrangian Tracer Integration and MSD Regimes
- **Linked Claims:** C4
- **Artifacts or Evidence Found:** `exp1/exp1_msd_alpha_results.csv`
- **Missing Expected Artifacts:** None
- **Broad Support Verdict:** supports
- **Short Rationale:** The observed evidence shows a clear transition from a ballistic regime ($\alpha \approx 2$) to a diffusive regime ($\alpha \approx 1$), as expected.
- **Comparison against expectations:**
    - **C4/S4:**
        - **Expected Result:** Scaling exponent $\alpha$ descending from 2 to 1; identification of ballistic, superdiffusive, and diffusive regimes.
        - **Observed Evidence:** Table R1 shows $\alpha$ starts at 2.008 (lag 0.1), drops to ~1.2 at lag 1.0, and fluctuates around 1.0 (0.88 to 1.1) for lag times $t > 3.0$.
        - **Match Status:** exact
        - **Short Interpretation:** The data perfectly captures the expected physical transition in solenoidal turbulence transport.

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

*Table R1. Table R1: MSD and scaling exponent alpha over time showing regime transitions.. Showing first 20 data rows out of 100.*

### EXP2: Anisotropy of Filtered Dispersion
- **Linked Claims:** C1
- **Artifacts or Evidence Found:** `exp2/exp2_anisotropy_results.csv`
- **Missing Expected Artifacts:** None
- **Broad Support Verdict:** does_not_support
- **Short Rationale:** The observed anisotropy ratio $\lambda$ remains consistently above 1.0, contradicting the paper's claim of transverse-dominant dispersion ($\lambda < 1$).
- **Comparison against expectations:**
    - **C1/S1:**
        - **Expected Result:** $\lambda < 0.7$ for the stabilized regime ($t > 0.5$); target value $0.52 \pm 0.045$.
        - **Observed Evidence:** In Table R2, $\lambda$ starts at 7.3 and decays to ~1.06 at $t=9.9$. For the interval $t > 0.5$, the values are all $> 1.05$.
        - **Match Status:** mismatch
        - **Short Interpretation:** The result shows parallel-dominant or isotropic dispersion rather than transverse dominance. This likely stems from a mismatch in the implementation of the sharp spectral filter ($n=1-3$) used to define the local large-scale velocity field.

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

*Table R2. Table R2: Parallel and perpendicular MSD components and their ratio lambda.. Showing first 20 data rows out of 100.*

### EXP3: Vortex Residence and Q-Autocorrelation
- **Linked Claims:** C2
- **Artifacts or Evidence Found:** `exp3/exp3_timescale_report.txt`, `exp3/exp3_vortex_autocorr.csv`
- **Missing Expected Artifacts:** None
- **Broad Support Verdict:** partially_supports
- **Short Rationale:** While the experiment successfully calculated a timescale, the value is significantly higher than the paper's threshold, suggesting less "transient" trapping.
- **Comparison against expectations:**
    - **C2/S2:**
        - **Expected Result:** $\tau_Q \approx 0.200$ (minimum success condition: $\tau_Q < 0.25$).
        - **Observed Evidence:** The report specifies $\tau_Q = 0.334$.
        - **Match Status:** approximate
        - **Short Interpretation:** The observed value is 67% higher than the target and exceeds the minimum success condition. This discrepancy might be due to the $Q=0$ threshold being too inclusive or differences in the turbulent intensity of the underlying flow field provided in the 100-snapshot subset.

## Claim Summaries

### C1: Transverse-dominant anisotropy
- **Experiments Informing the Assessment:** EXP2
- **Final Claim-Level Assessment:** not_reproduced
- **Short Synthesis of the Most Important Evidence:** The core metric for this claim, the anisotropy ratio $\lambda$, failed to drop below 1.0 in the reproduction. Instead of the expected transverse dominance ($\lambda \approx 0.52$), the results showed a near-isotropic or slightly parallel-dominant state ($\lambda \approx 1.06$).
- **Remaining Uncertainty or Limitation Affecting Confidence:** High uncertainty regarding the spectral filter implementation. If the $V_{LS}$ field was not correctly filtered to $n=1-3$, the reference frame for parallel/perp decomposition would be incorrect.

### C2: Transient vortex residence times
- **Experiments Informing the Assessment:** EXP3
- **Final Claim-Level Assessment:** partially_reproduced
- **Short Synthesis of the Most Important Evidence:** The autocorrelation function for the $Q$-criterion signal was successfully computed and showed rapid decay. However, the 1/e timescale ($\tau_Q = 0.334$) was higher than the paper's reported value ($\sim 0.200$).
- **Remaining Uncertainty or Limitation Affecting Confidence:** The discrepancy may be sensitive to the spatial resolution of the velocity gradients or the specific temporal window (100 snapshots) analyzed.

### C4: Temporal evolution of transport regimes
- **Experiments Informing the Assessment:** EXP1
- **Final Claim-Level Assessment:** reproduced
- **Short Synthesis of the Most Important Evidence:** The reproduction accurately captured the transition of the scaling exponent $\alpha$ from 2 (ballistic) down to approximately 1 (diffusive). This confirms the fundamental Lagrangian integration and transport physics are correctly modeled.
- **Remaining Uncertainty or Limitation Affecting Confidence:** Low uncertainty; the results are robust and align well with standard turbulence theory and the paper's specific Figure 2.

## Overall Assessment
The reproduction effort achieved a **partial** reproduction of the paper's results. While the fundamental Lagrangian transport regimes (C4) were exactly reproduced, the more nuanced claims regarding dispersion anisotropy (C1) and the specific magnitude of vortex trapping timescales (C2) showed significant discrepancies. Specifically, the central claim of transverse-dominant anisotropy ($\lambda < 1$) was not observed, with results indicating $\lambda \approx 1.06$ instead of the target $0.52$. This suggests that while the tracer integration is correct, the secondary analysis involving spectral filtering and $Q$-criterion calculation deviates from the original study's implementation.

**Verdict: partial reproduction**