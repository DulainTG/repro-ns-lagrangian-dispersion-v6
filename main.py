import argparse
import os
import sys
import logging
from pathlib import Path

# Ensure the project root is in the path so we can import from src
# This is usually automatic when running python main.py from the root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.experiments.outputs import ReproductionRunSettings
from src.experiments.exp1_lagrangian_tracer_msd import MsdRegimeExperiment
from src.experiments.exp2_anisotropy_filtered_dispersion import FilteredAnisotropyExperiment
from src.experiments.exp3_vortex_residence_q_autocorr import VortexTrappingExperiment

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def run_experiment(exp_name, mode, raw_data_dir, output_dir, overrides):
    """Orchestrates the execution of a single experiment with defaults and overrides."""
    logger.info("="*80)
    logger.info(f"STARTING REPRODUCTION EXPERIMENT: {exp_name}")
    logger.info(f"Execution Mode: {mode}")
    logger.info(f"Resolved Dataset Path: {raw_data_dir}")
    logger.info(f"Output Directory: {output_dir}")
    
    # Paper-canonical defaults for Solenoidal Turbulence (DS1)
    # Applied to EXP1, EXP2, and EXP3 where relevant.
    settings = {
        'number_of_tracers': 8000,
        'integration_scheme': 'RK4',
        'sub_steps_per_snapshot': 10,
        'interpolation': 'trilinear',
        'filter_range': (1, 3),
        'analysis_time_threshold': 0.5,
        'domain_extent': 1.0
    }

    # Apply smoke mode configuration bounds
    if mode == 'smoke':
        logger.info("Applying 'smoke' mode constraints: reducing sample size and resolution.")
        settings['number_of_tracers'] = 10
        settings['sub_steps_per_snapshot'] = 2

    # Apply explicit CLI overrides
    for key, value in overrides.items():
        if value is not None:
            logger.info(f"Override applied: {key} -> {value}")
            settings[key] = value

    logger.info(f"Final resolved parameters for {exp_name}: {settings}")
    
    # Initialize the standardized setting object
    config = ReproductionRunSettings(**settings)
    
    # Some experiments rely on RAW_DATA_DIR env var
    os.environ['RAW_DATA_DIR'] = str(raw_data_dir)
    
    # Instantiate the requested experiment
    if exp_name == "EXP1":
        exp = MsdRegimeExperiment(config)
    elif exp_name == "EXP2":
        exp = FilteredAnisotropyExperiment(config)
    elif exp_name == "EXP3":
        exp = VortexTrappingExperiment(config)
    else:
        raise ValueError(f"Unknown experiment designation: {exp_name}")

    # Ensure output directory exists before execution
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture current directory to restore later
    original_cwd = os.getcwd()
    
    try:
        # Move execution context to output directory so artifacts (hardcoded as relative) 
        # land in the specified --output-dir.
        os.chdir(output_dir)
        
        logger.info(f"Phase 1: Preparing experiment {exp_name} (loading snapshots and seeding)...")
        exp.prepare()
        
        logger.info(f"Phase 2: Executing numerical integration and analysis routines...")
        results = exp.run()
        
        logger.info(f"Phase 3: Serializing artifacts and generating reports...")
        exp.save_artifacts(results)
        
        logger.info(f"SUCCESS: {exp_name} reproduction pipeline complete.")
        
    except Exception as e:
        logger.error(f"CRITICAL FAILURE during {exp_name} execution: {e}")
        # Re-raise to trigger sys.exit(1) in main
        raise
    finally:
        # Always return to original CWD
        os.chdir(original_cwd)

def main():
    parser = argparse.ArgumentParser(
        description="Reproduction CLI for Transverse-Dominant Anisotropic Dispersion in Solenoidal Turbulence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Primary Command Arguments
    parser.add_argument(
        "experiment", 
        choices=["EXP1", "EXP2", "EXP3", "full-benchmark"],
        help="Target experiment to run or 'full-benchmark' for all three."
    )
    
    parser.add_argument(
        "--mode", 
        choices=["full", "smoke"], 
        default="full",
        help="Run scale. 'full' uses paper defaults; 'smoke' uses minimal bounds for fast CI/validation."
    )
    
    # Dataset and Path Arguments
    parser.add_argument(
        "--raw-data-dir", 
        type=str, 
        default=os.environ.get("RAW_DATA_DIR", "./raw_data"),
        help="Directory containing the VTK snapshot files (DS1)."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Directory where results will be written."
    )

    parser.add_argument(
       "--dataset",
       type=str,
       help="Dataset alias (e.g., DS1). Already defaults to DS1-compatible loader."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Space-separated list of dataset aliases."
    )

    # Individual Reproduction Argument Overrides
    group = parser.add_argument_group("Parameter Overrides (Experimental)")
    group.add_argument("--number-of-tracers", type=int, help="Total number of lagrangian tracers.")
    group.add_argument("--integration-scheme", choices=['RK4', 'Euler'], help="ODE integration scheme.")
    group.add_argument("--sub-steps", type=int, dest="sub_steps_per_snapshot", help="Integrator steps between snapshots.")
    group.add_argument("--interpolation", choices=['trilinear', 'nearest'], help="Field interpolation method.")
    group.add_argument("--filter-min", type=int, help="Min wavenumber for sharp spectral filter (EXP2).")
    group.add_argument("--filter-max", type=int, help="Upper bound for spectral band-pass (EXP2).")
    group.add_argument("--time-threshold", type=float, dest="analysis_time_threshold", help="Threshold time for averaging (EXP2).")

    args = parser.parse_args()

    # Consolidate overrides for passing to run_experiment
    overrides = {}
    if args.number_of_tracers is not None: overrides['number_of_tracers'] = args.number_of_tracers
    if args.integration_scheme is not None: overrides['integration_scheme'] = args.integration_scheme
    if args.sub_steps_per_snapshot is not None: overrides['sub_steps_per_snapshot'] = args.sub_steps_per_snapshot
    if args.interpolation is not None: overrides['interpolation'] = args.interpolation
    if args.analysis_time_threshold is not None: overrides['analysis_time_threshold'] = args.analysis_time_threshold
    
    if args.filter_min is not None or args.filter_max is not None:
        f_min = args.filter_min if args.filter_min is not None else 1
        f_max = args.filter_max if args.filter_max is not None else 3
        overrides['filter_range'] = (f_min, f_max)

    # Normalize paths
    raw_data_dir = Path(args.raw_data_dir).absolute()
    output_dir = Path(args.output_dir).absolute()

    # Pre-flight check: ensure snapshots are present
    if not raw_data_dir.exists() or not any(raw_data_dir.glob("Turb.hydro_w.*.vtk")):
        logger.error(f"Missing required dataset: No VTK snapshots found in {raw_data_dir}")
        print("\n" + "!"*80)
        print("DATASET CHECK FAILED")
        print("!"*80)
        print(f"VTK snapshot files (convention: Turb.hydro_w.#####.vtk) were not found in:")
        print(f"  {raw_data_dir}")
        print("\nPlease provide the correct path using --raw-data-dir or set the RAW_DATA_DIR")
        print("environment variable. Ensure you have the Solenoidal Turbulence (DS1) data.")
        print("!"*80 + "\n")
        sys.exit(1)

    # Execute designated command
    try:
        if args.experiment == "full-benchmark":
            logger.info("Executing COMPLETE REPRODUCTION SUITE (EXP1, EXP2, EXP3)")
            for exp_id in ["EXP1", "EXP2", "EXP3"]:
                run_experiment(exp_id, args.mode, raw_data_dir, output_dir, overrides)
        else:
            run_experiment(args.experiment, args.mode, raw_data_dir, output_dir, overrides)
    except Exception:
        # Exact failure already logged in run_experiment
        sys.exit(1)

if __name__ == "__main__":
    main()
