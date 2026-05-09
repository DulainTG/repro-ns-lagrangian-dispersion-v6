import subprocess
import pytest
import pandas as pd
from pathlib import Path
import os

@pytest.mark.timeout(30)
def test_final_smoke_exp3_canonical(tmp_path):
    """
    Validate the canonical EXP3 smoke-mode CLI path.
    """
    output_dir = tmp_path / "exp3_smoke_output"
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    
    # Guarded real-data input
    if not os.path.exists(raw_data_dir) or not list(Path(raw_data_dir).glob("Turb.hydro_w.*.vtk")):
        pytest.skip(f"Raw data not found at {raw_data_dir}")

    cmd = [
        "python", "main.py", "EXP3",
        "--mode", "smoke",
        "--raw-data-dir", str(raw_data_dir),
        "--output-dir", str(output_dir)
    ]
    
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=25, cwd=repo_root)
    
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "SUCCESS: EXP3 reproduction pipeline complete." in result.stdout
    
    # Assert artifact contract
    # 1. Autocorrelation CSV
    csv_path = output_dir / "exp3_vortex_autocorr.csv"
    assert csv_path.exists(), "Smoke artifact 'exp3_vortex_autocorr.csv' was not created."
    
    df = pd.read_csv(csv_path)
    required_columns = ['lag_time', 'autocorrelation_q_normalized']
    for col in required_columns:
        assert col in df.columns, f"Column {col} missing from artifact."
    
    # In smoke mode with snapshot_limit=10, we expect 10 rows.
    assert len(df) == 10

    # 2. Timescale report TXT
    report_path = output_dir / "exp3_timescale_report.txt"
    assert report_path.exists(), "Smoke artifact 'exp3_timescale_report.txt' was not created."
    
    with open(report_path, "r") as f:
        content = f.read()
        assert "EXP3: VORTEX TRAPPING TIMESCALE ANALYSIS" in content
        assert "Lagrangian Auto-correlation Time (tau_Q):" in content

@pytest.mark.timeout(30)
def test_final_smoke_exp3_dataset_alias(tmp_path):
    """
    Validate the EXP3 smoke-mode CLI path with --dataset alias.
    """
    output_dir = tmp_path / "exp3_smoke_output_alias"
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    
    # Guarded real-data input
    if not os.path.exists(raw_data_dir) or not list(Path(raw_data_dir).glob("Turb.hydro_w.*.vtk")):
        pytest.skip(f"Raw data not found at {raw_data_dir}")

    cmd = [
        "python", "main.py", "EXP3",
        "--dataset", "DS1",
        "--mode", "smoke",
        "--raw-data-dir", str(raw_data_dir),
        "--output-dir", str(output_dir)
    ]
    
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=25, cwd=repo_root)
    
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "SUCCESS: EXP3 reproduction pipeline complete." in result.stdout
    
    # Assert artifact contract
    csv_path = output_dir / "exp3_vortex_autocorr.csv"
    assert csv_path.exists(), "Smoke artifact 'exp3_vortex_autocorr.csv' was not created."
    
    df = pd.read_csv(csv_path)
    assert len(df) == 10
