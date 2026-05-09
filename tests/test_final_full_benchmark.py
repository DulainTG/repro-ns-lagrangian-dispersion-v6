import subprocess
import pytest
import pandas as pd
from pathlib import Path
import os

@pytest.mark.timeout(120)
def test_full_benchmark_smoke_canonical(tmp_path):
    """
    Validate the canonical full-benchmark smoke-run CLI path.
    Asserts that the complete cross-experiment smoke-mode contract is satisfied.
    """
    output_dir = tmp_path / "benchmark_smoke_output"
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    
    # Guarded real-data input
    if not os.path.exists(raw_data_dir) or not list(Path(raw_data_dir).glob("Turb.hydro_w.*.vtk")):
        pytest.skip(f"Raw data not found at {raw_data_dir}")

    cmd = [
        "python", "main.py", "full-benchmark",
        "--mode", "smoke",
        "--raw-data-dir", str(raw_data_dir),
        "--output-dir", str(output_dir)
    ]
    
    repo_root = Path(__file__).parent.parent
    # Combined timeout for all three experiments in smoke mode
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=100, cwd=repo_root)
    
    assert result.returncode == 0, f"Benchmark failed with stderr: {result.stderr}"
    
    # Verify orchestrated flow log messages
    assert "Executing COMPLETE REPRODUCTION SUITE (EXP1, EXP2, EXP3)" in result.stdout
    assert "SUCCESS: EXP1 reproduction pipeline complete." in result.stdout
    assert "SUCCESS: EXP2 reproduction pipeline complete." in result.stdout
    assert "SUCCESS: EXP3 reproduction pipeline complete." in result.stdout
    
    # Assert Cross-Experiment Artifact Contract
    
    # EXP1 Artifacts
    exp1_csv = output_dir / "exp1_msd_alpha_results.csv"
    assert exp1_csv.exists(), f"EXP1 artifact missing: {exp1_csv}"
    df1 = pd.read_csv(exp1_csv)
    assert len(df1) == 3 # snapshot_limit=3 for smoke
    for col in ['lag_time', 'msd_mean', 'msd_std', 'alpha_exponent']:
        assert col in df1.columns
        
    # EXP2 Artifacts
    exp2_csv = output_dir / "exp2_anisotropy_results.csv"
    assert exp2_csv.exists(), f"EXP2 artifact missing: {exp2_csv}"
    df2 = pd.read_csv(exp2_csv)
    assert len(df2) == 3 # snapshot_limit=3 for smoke
    for col in ['lag_time', 'msd_parallel', 'msd_perp', 'lambda_ratio']:
        assert col in df2.columns
        
    # EXP3 Artifacts
    exp3_csv = output_dir / "exp3_vortex_autocorr.csv"
    exp3_txt = output_dir / "exp3_timescale_report.txt"
    assert exp3_csv.exists(), f"EXP3 artifact missing: {exp3_csv}"
    assert exp3_txt.exists(), f"EXP3 artifact missing: {exp3_txt}"
    
    df3 = pd.read_csv(exp3_csv)
    assert len(df3) == 10 # EXP3 smoke snapshot_limit=10 in main.py
    assert 'autocorrelation_q_normalized' in df3.columns
    
    with open(exp3_txt, 'r') as f:
        content = f.read()
        assert "EXP3: VORTEX TRAPPING TIMESCALE ANALYSIS" in content
