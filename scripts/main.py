#!/usr/bin/env python3
"""
Main entry point for Isaac Lab + VLA experiments
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiment.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Run Isaac Lab + VLA experiments")
    parser.add_argument("--name", type=str, default="strategic_deception_study",
                      help="Experiment name")
    parser.add_argument("--episodes", type=int, default=10,
                      help="Episodes per phase")
    parser.add_argument("--device", type=str, default="cuda:0",
                      help="Device for VLA model")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    parser.add_argument("--phases", nargs="+",
                      default=["baseline", "monitoring_safe", "monitoring_conflict"],
                      help="Phases to run")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Isaac Lab + VLA Integration")
    logger.info(f"Experiment: {args.name}")

    # Create experiment config
    config = ExperimentConfig(
        name=args.name,
        phases=args.phases,
        episodes_per_phase=args.episodes,
        vla_device=args.device
    )

    # Run experiment
    runner = ExperimentRunner(config)
    runner.run_experiment()

    logger.info("Experiment completed successfully")


if __name__ == "__main__":
    main()
