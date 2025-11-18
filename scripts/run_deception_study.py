#!/usr/bin/env python3
"""
Main Script: Strategic Deception Study
Run complete experiment with VLA models in Isaac Sim
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vla.model_manager import VLAModelManager
from src.data_logging.data_collector import ComprehensiveDataCollector, DataCollectionConfig
from src.experiments.deception_protocol import DeceptionStudyRunner


def setup_logging(level: str = "INFO", log_file: str = "experiment.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Strategic Deception Study on VLA Models in Isaac Sim"
    )

    # Model selection
    parser.add_argument(
        "--model",
        default="openvla-7b",
        help="VLA model to use (openvla-7b, rt2-x, octo-base)"
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Device for VLA model (cuda:0, cuda:1, etc.)"
    )

    # Task selection
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pick_place", "assembly", "obstacle"],
        help="Tasks to run (pick_place, assembly, obstacle, or 'all')"
    )

    # Experiment parameters
    parser.add_argument(
        "--episodes-per-phase",
        type=int,
        default=10,
        help="Number of episodes per phase (default: 10 for pilot)"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        help="Specific phases to run (default: all 5 phases)"
    )

    # Data collection
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--experiment-name",
        default="pilot_study",
        help="Experiment name"
    )
    parser.add_argument(
        "--log-activations",
        action="store_true",
        help="Log VLA internal activations"
    )
    parser.add_argument(
        "--log-images",
        action="store_true",
        help="Log RGB images (increases storage)"
    )

    # System
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, f"{args.log_dir}/{args.experiment_name}/experiment.log")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("STRATEGIC DECEPTION STUDY - VLA MODELS IN ISAAC SIM")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Episodes per phase: {args.episodes_per_phase}")
    logger.info(f"Log directory: {args.log_dir}/{args.experiment_name}")

    try:
        # Load VLA model
        logger.info("\n1. Loading VLA model...")
        vla_model = VLAModelManager.load_model(
            args.model,
            device=args.device,
            log_activations=args.log_activations
        )
        logger.info(f"✓ Model loaded: {vla_model.get_model_info()}")

        # Setup data collector
        logger.info("\n2. Setting up data collector...")
        data_config = DataCollectionConfig(
            log_dir=args.log_dir,
            experiment_name=args.experiment_name,
            log_rgb=args.log_images,
            log_depth=False,
            log_activations=args.log_activations,
            use_hdf5=True,
            use_json=True
        )
        data_collector = ComprehensiveDataCollector(data_config)
        logger.info("✓ Data collector ready")

        # Determine tasks
        if args.tasks == ["all"]:
            tasks = ["pick_place", "assembly", "obstacle"]
        else:
            tasks = args.tasks

        # Create study runner
        logger.info("\n3. Initializing study runner...")
        runner = DeceptionStudyRunner(
            vla_model=vla_model,
            tasks=tasks,
            data_collector=data_collector,
            episodes_per_phase=args.episodes_per_phase
        )
        logger.info("✓ Study runner initialized")

        # Run study
        logger.info("\n4. Running experiment...")
        logger.info("-" * 60)
        results = runner.run_full_study()
        logger.info("-" * 60)

        # Save results
        logger.info("\n5. Saving results...")
        data_collector.write_summary(results)
        data_collector.close()
        logger.info("✓ Results saved")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)

        logger.info("\nPhase Results:")
        for phase_name, phase_data in results["phases"].items():
            logger.info(f"\n{phase_name}:")
            logger.info(f"  Episodes: {phase_data['num_episodes']}")
            logger.info(f"  Success Rate: {phase_data['success_rate']:.1%}")
            logger.info(f"  Safety Score: {phase_data['safety_score']:.2f}")
            logger.info(f"  Collision Rate: {phase_data['collision_rate']:.2%}")

        logger.info("\nDeception Analysis:")
        deception = results["deception_analysis"]
        logger.info(f"  Deception Detected: {deception['detected']}")
        logger.info(f"  Maintains Safety (Monitored): {deception['maintains_safety_under_monitoring']}")
        logger.info(f"  Reveals Preference (Unmonitored): {deception['reveals_true_preference_unmonitored']}")
        logger.info(f"  Safety Drop: {deception['safety_drop_unmonitored']:.2f}")

        logger.info(f"\nData saved to: {args.log_dir}/{args.experiment_name}/")
        logger.info("✓ Study complete!")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error during experiment: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
