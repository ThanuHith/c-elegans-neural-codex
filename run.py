from __future__ import annotations

import argparse
from pathlib import Path

from celegans_sim.config import SimulationConfig
from celegans_sim.simulation import run_headless, run_interactive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C. elegans-inspired neural, body, and drug simulation."
    )
    parser.add_argument(
        "--connectome",
        default="prototype",
        choices=["prototype", "surrogate302"],
        help="Connectome preset to load.",
    )
    parser.add_argument(
        "--scenario",
        default="foraging",
        choices=["foraging", "obstacle_course", "toxin_patch"],
        help="Environment preset.",
    )
    parser.add_argument(
        "--drug",
        default="baseline",
        choices=["baseline", "stimulant", "sedative", "neurotoxin"],
        help="Initial drug condition.",
    )
    parser.add_argument(
        "--dose",
        type=float,
        default=0.0,
        help="Initial dose between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without pygame and export metrics after the batch.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Headless simulation steps.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs")),
        help="Directory for exported logs and plots.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = SimulationConfig(
        connectome_mode=args.connectome,
        scenario=args.scenario,
        output_dir=Path(args.output_dir),
    )

    if args.headless:
        run_headless(
            config=config,
            steps=args.steps,
            drug_name=args.drug,
            dose=args.dose,
        )
    else:
        run_interactive(
            config=config,
            initial_drug=args.drug,
            initial_dose=args.dose,
        )


if __name__ == "__main__":
    main()

