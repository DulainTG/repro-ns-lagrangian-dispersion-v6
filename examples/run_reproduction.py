"""
Example script demonstrating how to use the reproduction components programmatically.
This script reflects the workflow orchestrated by main.py but allows for more 
granular control over the experiment execution and analysis.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.outputs import ReproductionRunSettings, ReportFormatter
from src.experiments.exp1_lagrangian_tracer_msd import MsdRegimeExperiment, RegimeTransitionValidator
from src.experiments.exp2_anisotropy_filtered_dispersion import FilteredAnisotropyExperiment, TransverseDominanceValidator
from src.experiments.exp3_vortex_residence_q_autocorr import VortexTrappingExperiment, VortexTrappingValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_all_experiments():
    """Demonstrates running all three experiments programmatically."""
    
    # 1. Setup paths and environment
    raw_data_dir = Path(os.environ.get("RAW_DATA_DIR", "/raw_data"))
    output_dir = Path("./example_outputs").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found at {raw_data_dir}. Please set RAW_DATA_DIR environment variable.")
        return

    # Set environment variable for the experiments
    os.environ['RAW_DATA_DIR'] = str(raw_data_dir)

    # 2. Define common configuration (Smoke mode settings for speed)
    settings = ReproductionRunSettings(
        number_of_tracers=20,
        integration_scheme='RK4',
        sub_steps_per_snapshot=2,
        snapshot_limit=5,
        interpolation='trilinear',
        filter_range=(1, 3),
        analysis_time_threshold=0.5,
        domain_extent=1.0
    )
    
    # Save the current directory to revert later
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    try:
        # --- Run EXP1: Mean Square Displacement ---
        logger.info("\n>>> RUNNING EXP1: Lagrangian Tracer MSD")
        exp1 = MsdRegimeExperiment(settings)
        exp1.prepare()
        results1 = exp1.run()
        exp1.save_artifacts(results1)
        
        # Optional validation
        validator1 = RegimeTransitionValidator()
        report1 = validator1.validate_msd_regimes(results1)
        logger.info(f"EXP1 Result: MSD Scaling Exponents calculated. C4 Satisfied: {report1.is_c4_satisfied}")

        # --- Run EXP2: Filtered Anisotropy ---
        logger.info("\n>>> RUNNING EXP2: Filtered Anisotropy")
        exp2 = FilteredAnisotropyExperiment(settings)
        exp2.prepare()
        results2 = exp2.run()
        exp2.save_artifacts(results2)
        
        # Optional validation
        validator2 = TransverseDominanceValidator()
        report2 = validator2.validate_anisotropy_ratio(results2)
        logger.info(f"EXP2 Result: Asymptotic Lambda Ratio: {report2.asymptotic_lambda:.4f}")
        logger.info(f"Transverse Dominance (C1) Satisfied: {report2.is_c1_satisfied}")

        # --- Run EXP3: Vortex Trapping ---
        logger.info("\n>>> RUNNING EXP3: Vortex Trapping")
        exp3 = VortexTrappingExperiment(settings)
        exp3.prepare()
        results3 = exp3.run()
        exp3.save_artifacts(results3)
        
        # Optional validation
        validator3 = VortexTrappingValidator()
        report3 = validator3.validate_trapping_timescale(results3.summary)
        logger.info(f"EXP3 Result: Tau_Q / Te: {report3.tau_q_over_te:.4f}")
        logger.info(f"Vortex Trapping (C2) Satisfied: {report3.is_c2_satisfied}")

        formatter = ReportFormatter()
        logger.info("\nFinal Summary Report:")
        logger.info(formatter.format_timescale_report(results3.summary))
        logger.info("\nSUCCESS: All examples completed. Artifacts are in " + str(output_dir.absolute()))

    except Exception as e:
        logger.error(f"Error during example execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always revert back
        os.chdir(original_cwd)

if __name__ == "__main__":
    run_all_experiments()
