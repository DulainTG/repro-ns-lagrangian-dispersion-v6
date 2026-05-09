import subprocess
import pytest
import pandas as pd
from pathlib import Path
import os

@pytest.mark.timeout(20)
def test_final_smoke_exp1_canonical(tmp_path):
    """
    Validate the canonical EXP1 smoke-mode CLI path.
    """
    output_dir = tmp_path / "exp1_smoke_output"
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    
    # Guarded real-data input
    if not os.path.exists(raw_data_dir) or not list(Path(raw_data_dir).glob("Turb.hydro_w.*.vtk")):
        pytest.skip(f"Raw data not found at {raw_data_dir}")

    cmd = [
        "python", "main.py", "EXP1",
        "--mode", "smoke",
        "--raw-data-dir", str(raw_data_dir),
        "--output-dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "SUCCESS: EXP1 reproduction pipeline complete." in result.stdout
    
    # Assert artifact contract
    artifact_path = output_dir / "exp1_msd_alpha_results.csv"
    assert artifact_path.exists(), "Smoke artifact 'exp1_msd_alpha_results.csv' was not created."
    
    df = pd.read_csv(artifact_path)
    required_columns = ['lag_time', 'msd_mean', 'msd_std', 'alpha_exponent']
    for col in required_columns:
        assert col in df.columns, f"Column {col} missing from artifact."
    
    # In smoke mode with snapshot_limit=3, we expect 3 rows.
    assert len(df) == 3

@pytest.mark.timeout(20)
def test_final_smoke_exp1_dataset_alias(tmp_path):
    """
    Validate the EXP1 smoke-mode CLI path with --dataset alias.
    """
    output_dir = tmp_path / "exp1_smoke_output_alias"
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    
    # Guarded real-data input
    if not os.path.exists(raw_data_dir) or not list(Path(raw_data_dir).glob("Turb.hydro_w.*.vtk")):
        pytest.skip(f"Raw data not found at {raw_data_dir}")

    cmd = [
        "python", "main.py", "EXP1",
        "--dataset", "DS1",
        "--mode", "smoke",
        "--raw-data-dir", str(raw_data_dir),
        "--output-dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "SUCCESS: EXP1 reproduction pipeline complete." in result.stdout
    
    # Assert artifact contract
    artifact_path = output_dir / "exp1_msd_alpha_results.csv"
    assert artifact_path.exists(), "Smoke artifact 'exp1_msd_alpha_results.csv' was not created."
    
    df = pd.read_csv(artifact_path)
    required_columns = ['lag_time', 'msd_mean', 'msd_std', 'alpha_exponent']
    for col in required_columns:
        assert col in df.columns, f"Column {col} missing from artifact."
    
    assert len(df) == 3
