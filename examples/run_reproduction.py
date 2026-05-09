"""
Example script demonstrating how to use the reproduction components programmatically.
This script reflects the workflow orchestrated by main.py.
"""

from pathlib import Path
import os
from src.experiments.outputs import ReproductionRunSettings
from src.experiments.exp1_lagrangian_tracer_msd import MsdRegimeExperiment

def main():
    # 1. Define configuration
    # These settings are the same as used in 'smoke' mode.
    settings = ReproductionRunSettings(
        number_of_tracers=10,
        integration_scheme='RK4',
        sub_steps_per_snapshot=2,
        snapshot_limit=3,
        interpolation='trilinear',
        filter_range=(1, 3),
        analysis_time_threshold=0.5,
        domain_extent=1.0
    )
    
    # 2. Setup paths
    raw_data_dir = Path(os.environ.get("RAW_DATA_DIR", "/raw_data"))
    output_dir = Path("./example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    if not raw_data_dir.exists():
        print(f"Raw data directory not found at {raw_data_dir}. Please set RAW_DATA_DIR.")
        return

    # 3. Instantiate and run experiment
    print("Running a small Lagrangian Tracer MSD experiment...")
    # The experiment class reads RAW_DATA_DIR from env
    os.environ['RAW_DATA_DIR'] = str(raw_data_dir)
    
    # Save the current directory to revert later
    original_cwd = os.getcwd()
    try:
        # Move to the output directory so artifacts land there
        os.chdir(output_dir)
        
        exp = MsdRegimeExperiment(settings)
        
        print("Preparing experiment (loading data)...")
        exp.prepare()
        
        print("Executing integration...")
        results = exp.run()
        
        print("Saving artifacts...")
        exp.save_artifacts(results)
        
        print(f"Done! Results are in {Path('.').absolute()}")
    finally:
        # Always revert back
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
