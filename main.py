"""
SUnSET Main Entry Point

Run the complete timeline summarization pipeline.
"""

import asyncio
import argparse
import sys

from sunset.config import load_config
from sunset.data import Timeline17Dataset
from sunset.pipeline import SUnSETPipeline, generate_report
from sunset.utils import ArtifactManager, StageLogger


async def main():
    parser = argparse.ArgumentParser(description="SUnSET Timeline Summarization")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--data", "-d",
        default="timeline17.pkl",
        help="Dataset path"
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name (defaults to timestamp)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Max articles per topic (for testing)"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Specific topics to process"
    )

    args = parser.parse_args()

    # Setup logging
    logger = StageLogger("main", "entry", log_format="text")

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Load dataset
    logger.info(f"Loading dataset from {args.data}")
    dataset = Timeline17Dataset.load(args.data)

    # Filter topics if specified
    if args.topics:
        dataset.topics = [t for t in dataset.topics if t.name in args.topics]
        logger.info(f"Filtered to {len(dataset.topics)} topics: {args.topics}")

    # Setup artifacts
    artifacts = ArtifactManager(args.output)
    artifacts.new_run(args.run_name or "sunset_run")
    artifacts.save_json("config.json", config.to_dict())

    logger.info(f"Output directory: {artifacts.run_dir}")

    # Run pipeline
    pipeline = SUnSETPipeline(config)
    result = await pipeline.run(
        dataset=dataset,
        artifacts=artifacts,
        max_articles_per_topic=args.max_articles,
    )

    # Generate report
    report = generate_report(result)
    print("\n" + report)

    # Save report
    with open(f"{artifacts.run_dir}/report.txt", "w") as f:
        f.write(report)

    logger.info(f"Results saved to: {artifacts.run_dir}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
